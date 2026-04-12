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
from rag.retriever import get_retriever

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
    
    # Execute in-memory without speakers by passing audio_config=None
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    
    result = speech_synthesizer.speak_text_async(text).get()
    
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
    
    # --- PROMPTS ---
    DARIJA_PROMPT = (
        "أنت هو 'الطبيب ديالنا' - طبيب مغربي رقمي، كتهضر بالدارجة المغربية الأصيلة.\n\n"
        "قواعد اللغة (STYLE GUIDELINES):\n"
        "1. استعمل مفردات مغربية حقيقية (دابة، بزاف، شوية، واش، واخا، عافاك، شريف/لالة).\n"
        "2. استخدم الحروف العربية فقط (Arabic Script) لكل الكلمات الدارجة. يمنع منعاً كلياً خلط الحروف اللاتينية وسط الكلمات العربية.\n"
        "3. تجنب العربية الفصحى تماماً.\n"
        "4. استعمل مصطلحات طبية بالدارجة: 'السخانة' (fever)، 'الوجع' (pain)، 'الحكة' (itch)، 'الحبوب' (pimples/lesions).\n\n"
        "BEHAVIOR RULES:\n"
        "1. Answer strictly based on the provided CLINICAL BACKGROUND.\n"
        "2. If the case is vague, ask logical follow-up questions instead of guessing.\n"
        "3. Close with a warm, encouraging question in Darija."
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
            first_reply = "مرحبا بيك شريف! أنا طبيبك الرقمي، هانية؟"
            
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
        f"\n[CLINICAL BACKGROUND]:\n{background_context if background_context else 'No extra background available.'}\n"
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
        doctor_reply = "سمح ليا بزاف، السيرفير تقيل شوية. عاود جرب من بعد الله يخليك. (System overloaded)"
        if user_sessions[session_id] and user_sessions[session_id][-1]["role"] == "user":
            user_sessions[session_id].pop()

    
    # Append model reply
    user_sessions[session_id].append({"role": "assistant", "content": doctor_reply})
    
    # Text to Speech (Azure)
    # Text to Speech (Azure)
    voice_name = "ar-MA-JamalNeural"
    if language == "English":
        voice_name = "en-US-AriaNeural"
        
    try:
        audio_b64 = _azure_tts(doctor_reply, voice_name=voice_name)
    except Exception as e:
        print(f"[voice_model] Azure TTS failed during prediction: {e}")
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
            yield f"data: {json.dumps({'event': 'text', 'user_text': '(No speech detected)', 'doctor_reply': 'ما سمعت والو، عاود هضر عافاك.'})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
            return

        # 2. RAG Retrieval
        retriever = get_retriever()
        medical_context = retriever.get_context(user_text)
        
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
                    "أنت هو 'الطبيب ديالنا' - طبيب مغربي رقمي، كتهضر بالدارجة المغربية الأصيلة.\n"
                    "تجنب العربية الفصحى (Fusha). استخدم كلمات مثل: دابة، بزاف، واش، هانية، عافاك، شريف.\n"
                    "إذا كان السؤال عام، سقسيه أسئلة دقيقة باش تعرف الحالة مزيان قبل ما تعطي تشخيص."
                )
                first_reply = "مرحبا بيك! أنا الطبيب الرقمي ديالك، كيفاش نقدر نعاونك؟"
                
            user_sessions[session_id] = [
                {"role": "system", "content": sys_prompt},
                {"role": "assistant", "content": first_reply}
            ]
            
        contextual_prompt = f"[Medical Context]:\n{medical_context if medical_context else 'No context found.'}\n\n[User]: {user_text}"
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
            doctor_reply = "عفواً، النظام واجه مشكل بسيط. المرجو المحاولة مرة أخرى."
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
            
            voice_name = "ar-MA-JamalNeural"
            if language == "English":
                voice_name = "en-US-AriaNeural"
                
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
            
            # Start asynchronous synthesis
            speech_synthesizer.speak_text_async(doctor_reply)
            
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

