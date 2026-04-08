"""
voice_model.py – Wrap Whisper transcribing, Gemini reasoning, and gTTS generation.
"""

import os
import tempfile
import base64
import whisper
import json
import queue
from google import genai
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from rag.retriever import get_retriever

load_dotenv(override=True)

_whisper_model = None
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

def process_voice_consultation(audio_bytes: bytes, session_id: str = "default") -> dict:
    """
    Takes audio bytes, transcribes, feeds to Gemini matching session state, 
    converts reply to Azure TTS, and returns a base64 string of the TTS audio.
    """
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio", "tmp")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm", dir=out_dir) as temp_in:
        temp_in.write(audio_bytes)
        in_path = temp_in.name

    try:
        model = get_whisper_model()
        darija_prompt = "هذه محادثة بالدارجة المغربية: واش، مزيان، بزاف، دابا، واخا، عافاك، ديالي، شنو، كيفاش."
        
        print("[DEBUG] 1. Starting Whisper Transcription...")
        result = model.transcribe(in_path, language="ar", initial_prompt=darija_prompt)
        user_text = result["text"].strip()
        print(f"[DEBUG] -> Transcribed: {user_text}")
        
        if not user_text:
            return {
                "user_text": "(No speech detected)",
                "doctor_reply": "ما سمعت والو، عاود هضر عافاك.",
                "audio_base64": ""
            }

        print("[DEBUG] 2. Querying RAG Database...")
        retriever = get_retriever()
        medical_context = retriever.get_context(user_text)
        print("[DEBUG] -> RAG Context retrieved.")
        
        client = get_gemini_client()
        
        if session_id not in user_sessions:
            sys_prompt = (
                "أنت طبيب مغربي محترف ومتعاطف جداً. تواصل مع المريض حصرياً بـ 'الدارجة المغربية' بشكل طبيعي ومريح.\n\n"
                "RULES FOR BEHAVIOR:\n"
                "1. NEVER repeat the same greetings or closings. Vary your conversational flow naturally.\n"
                "2. If the user gives generic symptoms (like fever or headache), DO NOT immediately output a random severe or extreme illness from the RAG context. Instead, ask short, logical follow-up questions to gather more specific symptoms (e.g., 'من إمتى باديك هاد الحريق؟', 'واش كاين شي أعراض خرى؟').\n"
                "3. ONLY suggest a diagnosis when you have collected enough specific symptoms that match the provided medical context.\n"
                "4. Answer ONLY using the provided medical context. If the answer is not in the context, politely say that you don't know in naturally sounding Moroccan Darija: 'سمح ليا، ماعنديش معلومات دقيقة على هاد الحالة دابا.'."
            )
            user_sessions[session_id] = [
                {"role": "user", "parts": [{"text": sys_prompt}]},
                {"role": "model", "parts": [{"text": "مرحبا! أنا طبيبك الرقمي."}]}
            ]
            
        contextual_prompt = f"[Medical Context]:\n{medical_context if medical_context else 'No context found.'}\n\n[User]: {user_text}"
        user_sessions[session_id].append({"role": "user", "parts": [{"text": contextual_prompt}]})
        
        if len(user_sessions[session_id]) > 42:
            user_sessions[session_id] = user_sessions[session_id][:2] + user_sessions[session_id][-40:]
        
        print("[DEBUG] 3. Waiting for Gemini Reply...")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_sessions[session_id],
            )
            doctor_reply = response.text
            print("[DEBUG] -> Gemini replied successfully.")
        except Exception as e:
            error_str = str(e)
            if any(code in error_str for code in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                doctor_reply = "عفواً، النظام عليه ضغط كبير دابا من طرف جوجل. المرجو انتظار بضع ثواني وإعادة المحاولة."
                print(f"[DEBUG] -> Gemini API Overloaded: {error_str}")
                # Remove recent failed prompt from memory 
                if user_sessions[session_id] and user_sessions[session_id][-1]["role"] == "user":
                    user_sessions[session_id].pop()
            else:
                print(f"[DEBUG] -> Gemini Exception: {error_str}")
                raise e
        
        user_sessions[session_id][-1]["parts"][0]["text"] = user_text 
        user_sessions[session_id].append({"role": "model", "parts": [{"text": doctor_reply}]})

        print("[DEBUG] 4. Generating Azure TTS Audio...")
        audio_b64 = _azure_tts(doctor_reply)
        print("[DEBUG] -> Audio Generation Finished Successfully.")
        
        return {
            "user_text": user_text,
            "doctor_reply": doctor_reply,
            "audio_base64": audio_b64
        }
    finally:
        # Cleanup input file
        if os.path.exists(in_path):
            os.remove(in_path)

def generate_medical_advice_for_prediction(prediction_label: str, confidence: float, session_id: str = "default", language: str = "Darija") -> dict:
    client = get_gemini_client()
    
    # --- RAG Integration ---
    retriever = get_retriever()
    background_context = retriever.get_context(prediction_label)
    
    # Initialize session if not exists
    if session_id not in user_sessions:
        if language == "Tamazight":
            sys_prompt = "أنت طبيب مغربي رقمي ومتعاطف. تتواصل مع المريض حصرياً بـ 'الأمازيغية المغربية (Tamazight)'. يجب عليك كتابة الأمازيغية باستخدام الحروف العربية فقط (Phonetic Arabic script) وليس اللاتينية. مثال: 'أزول مانيك أنتگيت'.\n\nRULES FOR BEHAVIOR:\n1. NEVER repeat greetings. 2. If vague, ask follow-up questions. 3. Answer strictly based on CLINICAL BACKGROUND if provided."
            first_reply = "أزول! نكي د أمجاي نك أراك."
        elif language == "English":
            sys_prompt = "You are a professional and empathetic digital Moroccan doctor. Speak EXCLUSIVELY in English in a comforting tone.\n\nRULES FOR BEHAVIOR:\n1. NEVER repeat greetings. 2. If vague, ask follow-up questions. 3. Answer strictly from CLINICAL BACKGROUND if exists."
            first_reply = "Hello! I am your digital doctor."
        else:
            sys_prompt = (
                "أنت طبيب مغربي رقمي ومحترف ومتعاطف. تتحدث بالدارجة المغربية فقط بشكل مريح وودود.\n\n"
                "RULES FOR BEHAVIOR:\n"
                "1. NEVER repeat the same greetings or closings. Vary your conversational flow naturally.\n"
                "2. If symptoms or results are vague, do not diagnose extreme diseases automatically. Ask logical follow-up questions.\n"
                "3. Answer strictly based on the provided CLINICAL BACKGROUND if it exists."
            )
            first_reply = "مرحبا! أنا طبيبك الرقمي."
            
        user_sessions[session_id] = [
            {"role": "user", "parts": [{"text": sys_prompt}]},
            {"role": "model", "parts": [{"text": first_reply}]}
        ]
        
    prompt_text = (
        f"المريض يقول: لقد قمت بعمل فحص صورة للجلد، والذكاء الاصطناعي شخص النتيجة: '{prediction_label}' بنسبة {confidence}%.\n"
        f"\n[CLINICAL BACKGROUND]:\n{background_context if background_context else 'No extra background available.'}\n"
        "\nأخبر المريض بالنتيجة باختصار وأسلوب متعاطف بالدارجة، وقدم نصيحة طبية سريعة بناءً على المعطيات الطبية أعلاه. "
        "اختم حديثك بسؤال منطقي ومفتوح عن حالته لتشجيعه على الكلام، وإياك أن تكرر نفس الأسئلة المعتادة حرفياً."
    )
    
    # Append user prompt
    user_sessions[session_id].append({"role": "user", "parts": [{"text": prompt_text}]})
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_sessions[session_id],
        )
        doctor_reply = response.text
    except Exception as e:
        error_str = str(e)
        if any(code in error_str for code in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
            try:
                print("[DEBUG] -> Gemini 2.5 Failed (Quota). Falling back to gemini-1.5-flash...")
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=user_sessions[session_id],
                )
                doctor_reply = response.text
            except Exception as e2:
                doctor_reply = "عفواً، الطبيب الرقمي مشغول حالياً بكثرة الطلبات (Quota). المرجو محاولة الفحص مرة أخرى بعد قليل."
                if user_sessions[session_id] and user_sessions[session_id][-1]["role"] == "user":
                    user_sessions[session_id].pop()
        else:
            raise e
    
    # Append model reply
    user_sessions[session_id].append({"role": "model", "parts": [{"text": doctor_reply}]})
    
    # Text to Speech (Azure)
    voice_name = "ar-MA-JamalNeural"
    if language == "English":
        voice_name = "en-US-AriaNeural"
        
    audio_b64 = _azure_tts(doctor_reply, voice_name=voice_name)
    
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
        
        # 3. Gemini Query with Session Tracking
        client = get_gemini_client()
        
        if session_id not in user_sessions:
            if language == "Tamazight":
                sys_prompt = "أنت طبيب مغربي محترف. تواصل مع المريض حصرياً بـ 'الأمازيغية المغربية (Tamazight)'. يجب عليك كتابة الأمازيغية باستخدام الحروف العربية فقط (Phonetic Arabic script). مثال: 'أزول مانيك أنتگيت'.\n\nRULES FOR BEHAVIOR:\n1. NEVER repeat greetings. 2. If vague (headache), ask follow-ups. 3. Answer ONLY from Medical Context. 4. If answer not in context, say you do not know."
                first_reply = "أزول! نكي د أمجاي نك أراك."
            elif language == "English":
                sys_prompt = "You are a professional Moroccan doctor. Speak EXCLUSIVELY in English.\n\nRULES FOR BEHAVIOR:\n1. NEVER repeat greetings. 2. Ask follow-up questions if symptoms are vague. 3. Answer strictly from the provided Medical Context. 4. Say you don't know if out of context."
                first_reply = "Hello! I am your digital doctor."
            else:
                sys_prompt = (
                    "أنت طبيب مغربي محترف ومتعاطف جداً. تواصل مع المريض حصرياً بـ 'الدارجة المغربية' بشكل طبيعي ومريح.\n\n"
                    "RULES FOR BEHAVIOR:\n"
                    "1. NEVER repeat the same greetings or closings. Vary your conversational flow naturally.\n"
                    "2. If the user gives generic symptoms (like fever or headache), DO NOT immediately output a random severe or extreme illness from the RAG context. Instead, ask short, logical follow-up questions to gather more specific symptoms (e.g., 'من إمتى باديك هاد الحريق؟', 'واش كاين شي أعراض خرى؟').\n"
                    "3. ONLY suggest a diagnosis when you have collected enough specific symptoms that match the provided medical context.\n"
                    "4. Answer ONLY using the provided medical context. If the answer is not in the context, politely say that you don't know in naturally sounding Moroccan Darija: 'سمح ليا، ماعنديش معلومات دقيقة على هاد الحالة دابا.'."
                )
                first_reply = "مرحبا! أنا طبيبك الرقمي."
                
            user_sessions[session_id] = [
                {"role": "user", "parts": [{"text": sys_prompt}]},
                {"role": "model", "parts": [{"text": first_reply}]}
            ]
            
        contextual_prompt = f"[Medical Context]:\n{medical_context if medical_context else 'No context found.'}\n\n[User]: {user_text}"
        user_sessions[session_id].append({"role": "user", "parts": [{"text": contextual_prompt}]})
        
        if len(user_sessions[session_id]) > 42:
            user_sessions[session_id] = user_sessions[session_id][:2] + user_sessions[session_id][-40:]
        
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_sessions[session_id],
            )
            doctor_reply = response.text
        except Exception as e:
            error_str = str(e)
            if any(code in error_str for code in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                try:
                    print("[DEBUG] -> Gemini 2.5 Failed (Quota). Falling back to gemini-1.5-flash...")
                    response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=user_sessions[session_id],
                    )
                    doctor_reply = response.text
                except Exception as e2:
                    doctor_reply = "عفواً، النظام عليه ضغط كبير دابا من طرف جوجل. المرجو انتظار بضع ثواني وإعادة المحاولة."
                    if user_sessions[session_id] and user_sessions[session_id][-1]["role"] == "user":
                        user_sessions[session_id].pop()
            else:
                raise e
        
        user_sessions[session_id][-1]["parts"][0]["text"] = user_text
        user_sessions[session_id].append({"role": "model", "parts": [{"text": doctor_reply}]})

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

