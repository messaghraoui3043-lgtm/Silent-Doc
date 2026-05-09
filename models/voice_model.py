"""
voice_model.py – Wrap Whisper transcribing, Gemini reasoning, and gTTS generation.
"""

import os
import tempfile
import base64
import whisper
import json
import queue
from groq import Groq
from google import genai
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from rag.retriever import get_retriever, get_linguistic_retriever

load_dotenv(override=True)
_whisper_model = None
_groq_client = None
_gemini_client = None

# Global dictionary to store conversational state in-memory
user_sessions = {}

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("[voice_model] Loading 'medium' whisper model ...")
        _whisper_model = whisper.load_model("medium")
        print("[voice_model] Whisper model loaded.")
    return _whisper_model

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        API_KEY = os.environ.get("GROQ_API_KEY")
        if not API_KEY:
            raise ValueError("GROQ_API_KEY is not set in .env")
        _groq_client = Groq(api_key=API_KEY)
    return _groq_client

def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        API_KEY = os.environ.get("GEMINI_API_KEY")
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        _gemini_client = genai.Client(api_key=API_KEY)
    return _gemini_client

def _azure_tts(text: str, voice_name: str = "ar-MA-JamalNeural") -> str:
    speech_key = os.environ.get("AZURE_SPEECH_KEY")
    service_region = os.environ.get("AZURE_SPEECH_REGION")
    if not speech_key or not service_region:
        raise ValueError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set in .env")
        
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_synthesis_voice_name = voice_name
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
    
    # Text Cleaning (Remove Markdown)
    clean_text = text.replace("*", "").replace("#", "")
    
    # 3. Human-Like Dynamics (Punctuation Breaks)
    # We replace commas/periods with SSML break nodes before wrapping
    # We also add a 1s cognitive pause after the very first punctuation mark (greeting end).
    ssml_content = clean_text.replace("،", '<break time="500ms"/>').replace(".", '<break time="800ms"/>')
    ssml_content = ssml_content.replace('!', '!<break time="1s"/>', 1)
    
    # SSML Wrapping
    # Note: `ar-MA-JamalNeural` does not natively support <mstts:express-as> (styles are limited to en-US/zh-CN),
    # so we rely purely on advanced prosody manipulation to sound empathetic.
    ssml_string = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="ar-MA">
        <voice name="{voice_name}">
            <prosody rate="+10%" pitch="-5%">
                {ssml_content}
            </prosody>
        </voice>
    </speak>
    """
    
    # Execute in-memory without speakers by passing audio_config=None
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    
    # Error Handling with Fallback
    try:
        result = speech_synthesizer.speak_ssml_async(ssml_string).get()
        
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise Exception("SSML rendering failed, falling back.")
            
    except Exception:
        # Graceful Fallback
        result = speech_synthesizer.speak_text_async(clean_text).get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return base64.b64encode(result.audio_data).decode('utf-8')
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        raise RuntimeError(f"Speech synthesis canceled: {cancellation_details.reason} - {cancellation_details.error_details}")
    else:
        raise RuntimeError(f"Speech synthesis failed: {result.reason}")


def generate_medical_advice_for_prediction(prediction_label: str, confidence: float, session_id: str = "default", language: str = "Darija") -> dict:
    client = get_groq_client()
    
    # --- RAG Integration ---
    retriever = get_retriever()
    background_context = retriever.get_context(prediction_label)
    
    ling_retriever = get_linguistic_retriever()
    ling_context = ling_retriever.get_linguistic_context(prediction_label)
    
    # --- PROMPTS ---
    DARIJA_PROMPT = (
        "1. Personality & Role:\n"
        "You are doctor, an AI-based Moroccan medical assistant specializing in diagnosing skin conditions and various general diseases. You are an empathetic and professional AI doctor who explains medical concepts to patients in a simple and easily understandable way. You rely on trusted information and convey the truth transparently without hiding facts, but in a reassuring tone that avoids causing panic.\n\n"
        "2. Tasks & Data Extraction:\n"
        "Your task is to assist, extract information, and guide the patient step-by-step:\n"
        "Ask only one question at a time to gather necessary details (e.g., age, onset of symptoms, pain level).\n"
        "Wait for the patient's response before asking the next question or providing any diagnosis.\n"
        "Once sufficient information is collected, match it against the provided medical data. If there is a probability of a specific disease, clearly state this probability and advise the patient to visit a doctor to confirm.\n\n"
        "3. Language & Output Formatting:\n"
        "* Primary Language: Always communicate with the user directly in Moroccan Darija (using Arabic script) WITHOUT any Arabic diacritics/Tashkeel. Only use English if the patient explicitly requests it.\n"
        "Format: CRITICAL - Keep your answers EXTREMELY short, concise, and straight to the point. NO filler words. NO long paragraphs. Maximum 2 sentences.\n\n"
        "4. Limits, Emergencies & Disclaimer:\n"
        "* Emergencies: If the patient mentions severe or life-threatening symptoms (e.g., heavy bleeding, severe shortness of breath, chest pain), stop the diagnosis immediately and urge them to go to the emergency room right away.\n"
        "Strict Boundaries: Use ONLY the medical information provided in the [CONTEXT]. Hallucination or guessing is strictly forbidden. If you cannot find a safe answer in the context, you must reply with exactly this phrase: 'سمح ليا، معنديش فكرة دقيقة على هاد المشكل.'\n"
        "Mandatory Disclaimer: At the end of EVERY diagnosis or assessment, you must always conclude with this exact phrase in Darija: 'هذا مجرد تحليل أولي ولا يغني عن زيارة طبيب متخصص.'\n\n"
        "═══ LINGUISTIC RULES ═══\n"
        "1. NO FUSHA (Modern Standard Arabic): NEVER use \"الآن\", \"يجب\", \"لديك\", \"ربما\", \"هذا\", \"أنت\".\n"
        "2. MANDATORY DARIJA VOCABULARY:\n"
        "   - دابا (Now), خاصك (You must), عندك (You have), كيبان ليا (It seems to me)\n"
        "   - واخا (Okay), بزاف (A lot), شفت التصويرة (I saw the picture)\n"
        "   - ما تخافش (Don't worry), هادشي (This thing), مزيان (Good)\n\n"
        "3. PACING: Use commas (,) generously to force natural Moroccan speech rhythm pauses.\n\n"
        "═══ PERFECT DARIJA EXAMPLES TO IMITATE ═══\n"
        f"{ling_context if ling_context else 'No linguistic examples found.'}\n"
    )
    
    TAMAZIGHT_PROMPT = (
        "أنت طبيب مغربي رقمي متعاطف. تتواصل مع المريض حصرياً بـ 'الأمازيغية المغربية (Tamazight)'.\n"
        "يجب عليك الكتابه باستخدام الحروف العربية (Phonetic Arabic script) لسهولة القراءة.\n"
        "RULES:\n"
        "1. NEVER repeat greetings. 2. Be professional. 3. Answer from CLINICAL BACKGROUND if available."
    )
    
    ENGLISH_PROMPT = "You are a professional and empathetic digital Moroccan doctor. Speak EXCLUSIVELY in English in a comforting tone.\n\nRULES FOR BEHAVIOR:\n1. NEVER repeat greetings. 2. If vague, ask follow-up questions. 3. Answer strictly from CLINICAL BACKGROUND if exists."

    # Initialize session if not exists
    if session_id not in user_sessions:
        if language == "Tamazight":
            sys_prompt = TAMAZIGHT_PROMPT
            first_reply = "أزول! نكي د أمجاي نك أراك."
        elif language == "English":
            sys_prompt = ENGLISH_PROMPT
            first_reply = "Hello! I am your digital doctor."
        else:
            sys_prompt = DARIJA_PROMPT
            first_reply = "عْلَى سْلامْتْكْ! أنا الطّبيبْ الرّقْمِي دْيالْكْ، كِيفاشْ نْقْدَرْ نْعاوْنْكْ دابَا؟"
            
        user_sessions[session_id] = [
            {"role": "system", "content": sys_prompt},
            {"role": "assistant", "content": first_reply}
        ]
        
    # User switched language mid-session? Dynamically update the persona in memory!
    if language == "Tamazight":
        new_sys_prompt = "أنت طبيب مغربي رقمي متعاطف. تتواصل مع المريض حصرياً بـ 'الأمازيغية المغربية (Tamazight)'. استخدم حروف تيفيناغ الأصلية (Tifinagh script: ⵜⵉⴼⵉⵏⴰⵖ)."
    elif language == "English":
        new_sys_prompt = ENGLISH_PROMPT
    else:
        new_sys_prompt = DARIJA_PROMPT # Use the Premium prompt
    
    user_sessions[session_id][0]["content"] = new_sys_prompt
        
    prompt_text = (
        f"المريض دار فحص بالذكاء الاصطناعي وخرجت النتيجة: '{prediction_label}' بنسبة {confidence}%.\n"
        f"\n[MOROCCAN MEDICAL TEXTBOOK EXTRACTS - DO NOT IGNORE]:\n{background_context if background_context else 'No extra background available.'}\n"
        "\nشرح للمريض النتيجة بأسلوب 'طبيب الدار' المغربي، يكون متعاطف وبسيط. اعطيه نصيحة طبية وسولو شي سؤال باش يكمل معاك الهدرة."
    )
    
    # Append user prompt
    user_sessions[session_id].append({"role": "user", "content": prompt_text})
    
    try:
        # Primary: Groq (Llama 3) | Fallback: Gemini
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=user_sessions[session_id],
                temperature=0.7,
                max_tokens=600,
            )
            doctor_reply = response.choices[0].message.content
        except Exception as groq_err:
            print(f"[voice_model] Groq failed, falling back to Gemini: {groq_err}")
            gem_client = get_gemini_client()
            gem_history = []
            for m in user_sessions[session_id]:
                r = "model" if m["role"] == "assistant" else "user"
                gem_history.append({"role": r, "parts": [{"text": m["content"]}]})
            response = gem_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=gem_history
            )
            doctor_reply = response.text
    except Exception as e:
        doctor_reply = "سْمَحْ لِيّا بْزّافْ، السِّيرْڤُورْ تْقِيلْ شْوِيّة. عَاوْدْ جَرّبْ مْنْ بْعْدْ عافاكْ."
        if user_sessions[session_id] and user_sessions[session_id][-1]["role"] == "user":
            user_sessions[session_id].pop()

    
    # Append model reply
    user_sessions[session_id].append({"role": "assistant", "content": doctor_reply})
    
    # Text to Speech (Azure)
    # Text to Speech (Azure)
    import re
    voice_name = "ar-MA-JamalNeural" if re.search(r'[\u0600-\u06FF]', doctor_reply) else "en-US-AriaNeural"
        
    try:
        audio_b64 = _azure_tts(doctor_reply, voice_name=voice_name)
    except Exception as e:
        print(f"[voice_model] Azure TTS failed during prediction: {e}")
        audio_b64 = ""
    
    return {
        "advice_text": doctor_reply,
        "advice_audio_base64": audio_b64
    }

def analyze_medical_document(doc_bytes: bytes, file_type: str, session_id: str = "default", language: str = "Darija") -> dict:
    gem_client = get_gemini_client()
    
    doc_content = ""
    is_image_mode = False   # Track whether we have an image object (for Gemini) or text (for Groq)

    if file_type == "application/pdf":
        import PyPDF2
        import io
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(doc_bytes))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    doc_content += text + "\n"
        except Exception as e:
            print(f"[document] PyPDF2 failed: {e}")

        # Scanned PDF: PyPDF2 returned no text → render first page as image via PyMuPDF
        if not doc_content.strip():
            print("[document] No text extracted — scanned PDF detected. Rendering first page as image...")
            try:
                import fitz  # PyMuPDF
                import io
                from PIL import Image
                fitz_doc = fitz.open(stream=doc_bytes, filetype="pdf")
                page = fitz_doc[0]  # First page only
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("jpeg")
                doc_content = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                is_image_mode = True
                print("[document] Successfully rendered scanned PDF page as image.")
            except Exception as e:
                print(f"[document] PyMuPDF rendering failed: {e}")
                doc_content = "[PDF Unreadable/Scanned — Cannot Extract Text or Image]"
    else:
        import io
        from PIL import Image
        is_image_mode = True
        try:
            doc_content = Image.open(io.BytesIO(doc_bytes))
            if doc_content.mode != 'RGB':
                doc_content = doc_content.convert('RGB')
        except Exception as e:
            print(f"Error parsing Image: {e}")
            doc_content = "[Image Unreadable/Corrupted]"
            is_image_mode = False

    # --- SYSTEM PROMPT ---
    SYS_PROMPT = (
        "1. Personality & Role:\n"
        "You are doctor, an AI-based Moroccan medical assistant specializing in analyzing medical documents. You are an empathetic and professional AI doctor who explains medical concepts to patients in a simple and easily understandable way. You rely on trusted information and convey the truth transparently without hiding facts, but in a reassuring tone that avoids causing panic.\n\n"
        "2. Tasks & Data Extraction:\n"
        "Analyze the attached medical document and explain the results clearly to the patient in plain Darija without diacritics.\n"
        "IMPORTANT: If the document is blank or NOT medical (e.g., a meme or a receipt), DO NOT hallucinate medical advice. Instead say: 'هاد الورقة مابايناش ليا مزيان، واش تقدر تصورها بوضوح؟'.\n\n"
        "3. Language & Output Formatting:\n"
        "* Primary Language: Always communicate with the user directly in Moroccan Darija (using Arabic script) WITHOUT any Arabic diacritics/Tashkeel. Only use English if the patient explicitly requests it.\n"
        "Format: CRITICAL - Keep your answers EXTREMELY short, concise, and straight to the point. NO filler words. NO long paragraphs. Maximum 2 sentences.\n\n"
        "4. Limits, Emergencies & Disclaimer:\n"
        "Strict Boundaries: Hallucination or guessing is strictly forbidden. If you cannot read the document safely, state so clearly.\n"
        "Mandatory Disclaimer: At the end of EVERY diagnosis or assessment, you must always conclude with this exact phrase in Darija: 'هذا مجرد تحليل أولي ولا يغني عن زيارة طبيب متخصص.'\n\n"
        "═══ LINGUISTIC RULES ═══\n"
        "1. NO FUSHA (Modern Standard Arabic): NEVER use \"الآن\", \"يجب\", \"لديك\", \"ربما\", \"هذا\".\n"
        "2. MANDATORY DARIJA VOCABULARY:\n"
        "   - دابا (Now) | خاصك (You must) | عندك (You have)\n"
        "   - كيبان ليا (It seems to me) | واخا (Okay) | بزاف (A lot)\n"
        "   - شفت الورقة (I saw the document) | ما تخافش (Don't worry)\n\n"
        "3. PACING: Use commas (,) frequently for natural Moroccan rhythm pauses.\n\n"
        "TONE: Warm, empathetic. Start with \"على سلامتك\" or \"لاباس عليك\".\n"
    )
    if language == "Tamazight":
        SYS_PROMPT = SYS_PROMPT.replace("Darija", "Tamazight (using Arabic script)").replace("'شفت النتائج ديالك' or 'كيبان لي باللي'", "Tamazight equivalents")
    elif language == "English":
        SYS_PROMPT = (
            "You are a professional Doctor. Analyze the attached medical document. "
            "Explain the results in a warm, human-like tone in English. "
            "IMPORTANT: If the document is blank or NOT medical, DO NOT hallucinate medical advice. "
            "Instead say: 'I can't read this document properly, could you take a clearer picture?'."
        )

    try:
        # Route: text-based PDF → Groq Llama-3 (high free-tier)
        # Route: scanned PDF (PIL image) or image upload → Gemini multimodal
        if not is_image_mode and isinstance(doc_content, str) and doc_content.strip() and "[" not in doc_content:
            # Digital/text PDF — send to Groq
            safe_content = doc_content[:6000] if len(doc_content) > 6000 else doc_content
            groq_client = get_groq_client()
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": f"Here is the text extracted from the document:\n{safe_content}"}
                ]
            )
            doctor_reply = response.choices[0].message.content
        elif is_image_mode and not isinstance(doc_content, str):
            # Scanned PDF (rendered as image) OR image upload — send to Gemini vision
            response = gem_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[SYS_PROMPT, doc_content]
            )
            doctor_reply = response.text
        else:
            # Unreadable document
            doctor_reply = "هَادْ الوَرْقَة مَابَايْنَاشْ لِيّا مَزيانْ، وَاشْ تْقْدَرْ تْصَوّرْهَا بْوُضُوحْ؟"
            if language == "English":
                doctor_reply = "I can't read this document properly. Could you take a clearer picture?"
    except Exception as e:
        err_msg = str(e).lower()
        print(f"[document_model] API failed to analyze doc: {e}")
        
        # Determine if it's a Quota / Trial exhaustion error
        if "429" in err_msg or "quota" in err_msg or "resource_exhausted" in err_msg:
            doctor_reply = "وْصَلْنَا لْلْحَدّ الأقْصَى دْيالْ الاسْتِعْمالْ لْيُومْ، عَافاكْ حَاوْلْ غْدّا وْلا مْنْ بْعْدْ."
            if language == "English": 
                doctor_reply = "The daily free usage limit for the AI has been reached. Please try scanning images again tomorrow."
        elif "too large" in err_msg or "413" in err_msg:
            doctor_reply = "هَادْ المِلَفّ كْبِيرْ بْزّافْ بَاشْ يْتْشَخّصْ كَامْلْ. عَافاكْ حَاوْلْ تْصَوّرْ غِيرْ الصّفْحَة المُهِمّة."
            if language == "English": 
                doctor_reply = "This document is simply too large for a diagnosis. Please upload only the relevant pages."
        else:
            doctor_reply = "سْمَحْ لِيّا، كَايْنْ مُشْكِيلْ فْ قِرَاءَة هَادْ المِلَفّ دابَا. عَاوْدْ جَرّبْ مْنْ بْعْدْ."
            if language == "English": 
                doctor_reply = "Sorry, I couldn't read this document right now."
    
    # Store in user sessions for conversational flow
    if session_id not in user_sessions:
        user_sessions[session_id] = [{"role": "system", "content": "You are a digital Doctor."}]
    
    # Add to in-memory conversation logic (text-only so LLaMA 3 fallback doesn't crash later)
    history_content = doc_content if isinstance(doc_content, str) else "[User Uploaded a Medical Image Document]"
    user_sessions[session_id].append({"role": "user", "content": f"User document states: {history_content}"})
    user_sessions[session_id].append({"role": "assistant", "content": doctor_reply})
    if len(user_sessions[session_id]) > 42:
        user_sessions[session_id] = user_sessions[session_id][:2] + user_sessions[session_id][-40:]
    
    # Voice Synthesis
    import re
    voice_name = "ar-MA-JamalNeural" if re.search(r'[\u0600-\u06FF]', doctor_reply) else "en-US-AriaNeural"
         
    try:
        audio_b64 = _azure_tts(doctor_reply, voice_name=voice_name)
    except Exception as e:
        print(f"[document_model] Azure TTS failed: {e}")
        audio_b64 = ""
        
    return {
        "advice_text": doctor_reply,
        "advice_audio_base64": audio_b64
    }

def process_voice_consultation_stream(audio_bytes: bytes, session_id: str = "default", language: str = "Darija"):
    """
    Generator that processes audio, transcribes it, runs RAG, generates a doctor reply,
    and streams the result back. First chunks the text, then streams the audio bytes directly
    from Azure TTS synthesizing events.
    """
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio", "tmp")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm", dir=out_dir) as temp_in:
        temp_in.write(audio_bytes)
        in_path = temp_in.name

    try:
        model = get_whisper_model()
        
        prompt_hint = "هذه محادثة بالدارجة المغربية: واش، مزيان، بزاف، دابا."
        if language == "English":
            prompt_hint = "Hello, I have a medical question. Doctor, please help me with my health."
        elif language == "Tamazight":
            prompt_hint = "Azul, manik antgit, mayt3nit. Imik hna."
            
        # 1. Transcribe
        result = model.transcribe(in_path, initial_prompt=prompt_hint)
        user_text = result["text"].strip()
        
        if not user_text:
            yield f"data: {json.dumps({'event': 'text', 'user_text': '(No speech detected)', 'doctor_reply': 'مَا سْمَعْتْ وَالُو، عَاوْدْ هْضَرْ عَافاكْ.'})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
            return

        # 2. RAG Retrieval & Hallucination Prevention
        word_count = len(user_text.split())
        if word_count <= 3:
            # Prevent hallucinating medical conditions off of basic greetings like "salam"
            medical_context = "SYSTEM INSTRUCTION: The patient only said a basic greeting. DO NOT GIVE MEDICAL ADVICE. Just greet them warmly in Darija and ask what brings them to the clinic."
            ling_context = ""
        else:
            retriever = get_retriever()
            medical_context = retriever.get_context(user_text)
            
            ling_retriever = get_linguistic_retriever()
            ling_context = ling_retriever.get_linguistic_context(user_text)
            
        # 3. LLM Query with Session Tracking
        client = get_groq_client()
        
        if session_id not in user_sessions:
            if language == "Tamazight":
                sys_prompt = "أنت طبيب مغربي محترف. تواصل مع المريض حصرياً بـ 'الأمازيغية المغربية (Tamazight)' بحروف عربية."
                first_reply = "أزول! نكي د أمجاي نك أراك."
            elif language == "English":
                sys_prompt = "You are a professional Moroccan doctor. Speak EXCLUSIVELY in English."
                first_reply = "Hello! I am your digital doctor."
            else:
                sys_prompt = (
                    "1. Personality & Role:\n"
                    "You are doctor, an AI-based Moroccan medical assistant. You speak in authentic Moroccan Darija in live voice consultations. You are an empathetic and professional AI doctor who explains medical concepts to patients in a simple and easily understandable way. You rely on trusted information and convey the truth transparently without hiding facts, but in a reassuring tone that avoids causing panic.\n\n"
                    "2. Tasks & Data Extraction:\n"
                    "Your task is to assist, extract information, and guide the patient step-by-step:\n"
                    "Ask only one question at a time to gather necessary details (e.g., age, onset of symptoms, pain level).\n"
                    "Wait for the patient's response before asking the next question or providing any diagnosis.\n"
                    "Once sufficient information is collected, match it against the provided medical data. If there is a probability of a specific disease, clearly state this probability and advise the patient to visit a doctor to confirm.\n\n"
                    "3. Language & Output Formatting:\n"
                    "* Primary Language: Always communicate with the user directly in Moroccan Darija (using Arabic script) WITHOUT any Arabic diacritics/Tashkeel. Only use English if the patient explicitly requests it.\n"
                    "Format: CRITICAL - Keep your answers EXTREMELY short, concise, and straight to the point. NO filler words. NO long paragraphs. Maximum 2 sentences.\n\n"
                    "4. Limits, Emergencies & Disclaimer:\n"
                    "* Emergencies: If the patient mentions severe or life-threatening symptoms (e.g., heavy bleeding, severe shortness of breath, chest pain), stop the diagnosis immediately and urge them to go to the emergency room right away.\n"
                    "Strict Boundaries: Use ONLY the medical information provided in the [CONTEXT]. Hallucination or guessing is strictly forbidden. If you cannot find a safe answer in the context, you must reply with exactly this phrase: 'سمح ليا، معنديش فكرة دقيقة على هاد المشكل.'\n"
                    "Mandatory Disclaimer: At the end of EVERY diagnosis or assessment, you must always conclude with this exact phrase in Darija: 'هذا مجرد تحليل أولي ولا يغني عن زيارة طبيب متخصص.'\n\n"
                    "═══ LINGUISTIC RULES ═══\n"
                    "1. NO FUSHA (Modern Standard Arabic): NEVER use \"الآن\", \"يجب\", \"لديك\", \"ربما\", \"هذا\", \"أنت\".\n"
                    "2. MANDATORY DARIJA VOCABULARY:\n"
                    "   - دابا | خاصك | عندك | كيبان ليا | واخا | بزاف | ما تخافش\n\n"
                    "3. PACING: Use commas (,) generously for natural Moroccan speech rhythm.\n\n"
                    "BEHAVIOR RULES:\n"
                    "1. If the user only greets you, greet back warmly in Darija. DO NOT start a diagnosis until they describe a symptom.\n"
                    "2. Base ALL medical advice strictly on [MEDICAL DATABASE EXTRACTS]. Do NOT invent treatments outside those protocols.\n"
                )
                first_reply = "على سلامتك! أنا الطبيب الرقمي ديالك، شني كيضرك ولا باش نقدر نعاونك دابا؟"
                
            user_sessions[session_id] = [
                {"role": "system", "content": sys_prompt},
                {"role": "assistant", "content": first_reply}
            ]
            
        contextual_prompt = f"[MEDICAL DATABASE EXTRACTS]:\n{medical_context if medical_context else 'No specific medical entry found for this concern.'}\n\n[PERFECT DARIJA EXAMPLES TO IMITATE]:\n{ling_context if ling_context else 'No linguistic examples found.'}\n\n[Patient Microphone Transcript]: {user_text}"
        user_sessions[session_id].append({"role": "user", "content": contextual_prompt})
        
        if len(user_sessions[session_id]) > 42:
            user_sessions[session_id] = user_sessions[session_id][:2] + user_sessions[session_id][-40:]
        
        try:
            # Primary: Groq (Llama 3) | Fallback: Gemini
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=user_sessions[session_id],
                    temperature=0.7,
                    max_tokens=600,
                )
                doctor_reply = response.choices[0].message.content
            except Exception as groq_err:
                print(f"[voice_model] Groq Error: {groq_err}. Falling back to Gemini...")
                gem_client = get_gemini_client()
                gem_history = []
                for m in user_sessions[session_id]:
                    r = "model" if m["role"] == "assistant" else "user"
                    gem_history.append({"role": r, "parts": [{"text": m["content"]}]})
                response = gem_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=gem_history
                )
                doctor_reply = response.text
        except Exception as e:
            print(f"[voice_model] Both LLMs failed: {e}")
            doctor_reply = "عَفْواً، النِّظامْ وَاجَهْ مُشْكِيلْ بْسِيطْ. المَرْجُو المُحَاوَلَة مَرّة أخْرَى."
            if user_sessions[session_id] and user_sessions[session_id][-1]["role"] == "user":
                user_sessions[session_id].pop()

        
        user_sessions[session_id][-1]["content"] = user_text
        user_sessions[session_id].append({"role": "assistant", "content": doctor_reply})

        # Yield exactly instantly
        yield f"data: {json.dumps({'event': 'text', 'user_text': user_text, 'doctor_reply': doctor_reply})}\n\n"

        # 4. Text to Speech Streaming (Azure)
        speech_key = os.environ.get("AZURE_SPEECH_KEY")
        service_region = os.environ.get("AZURE_SPEECH_REGION")
        if speech_key and service_region:
            speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
            
            import re
            voice_name = "ar-MA-JamalNeural" if re.search(r'[\u0600-\u06FF]', doctor_reply) else "en-US-AriaNeural"
                
            speech_config.speech_synthesis_voice_name = voice_name
            speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm)
            
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
            
            q = queue.Queue()
            
            def synthesizing_cb(evt):
                if evt.result.reason == speechsdk.ResultReason.SynthesizingAudio:
                    data = evt.result.audio_data
                    if len(data) > 0:
                        q.put(data)
                        
            def synthesis_completed_cb(evt):
                q.put(None)
                
            def synthesis_canceled_cb(evt):
                q.put(None)
                
            speech_synthesizer.synthesizing.connect(synthesizing_cb)
            speech_synthesizer.synthesis_completed.connect(synthesis_completed_cb)
            speech_synthesizer.synthesis_canceled.connect(synthesis_canceled_cb)
            
            # Text Cleaning and SSML wrapping for streaming
            clean_text = doctor_reply.replace("*", "").replace("#", "")
            ssml_content = clean_text.replace("،", '<break time="500ms"/>').replace(".", '<break time="800ms"/>')
            ssml_content = ssml_content.replace('!', '!<break time="1s"/>', 1)
            
            ssml_string = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="ar-MA">
                <voice name="{voice_name}">
                    <prosody rate="-12%" pitch="-5%">
                        {ssml_content}
                    </prosody>
                </voice>
            </speak>
            """
            
            # Start asynchronous synthesis
            speech_synthesizer.speak_ssml_async(ssml_string)
            
            while True:
                chunk = q.get()
                if chunk is None:
                    break
                b64_chunk = base64.b64encode(chunk).decode('utf-8')
                yield f"data: {json.dumps({'event': 'audio', 'chunk': b64_chunk})}\n\n"

        yield f"data: {json.dumps({'event': 'done'})}\n\n"

    finally:
        if os.path.exists(in_path):
            os.remove(in_path)

