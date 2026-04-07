"""
voice_model.py – Wrap Whisper transcribing, Gemini reasoning, and gTTS generation.
"""

import os
import tempfile
import base64
import whisper
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

def _azure_tts(text: str) -> str:
    speech_key = os.environ.get("AZURE_SPEECH_KEY")
    service_region = os.environ.get("AZURE_SPEECH_REGION")
    if not speech_key or not service_region:
        raise ValueError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set in .env")
        
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_synthesis_voice_name = "ar-MA-JamalNeural"
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
        
        # 1. Transcribe
        result = model.transcribe(in_path, language="ar", initial_prompt=darija_prompt)
        user_text = result["text"].strip()
        
        if not user_text:
            return {
                "user_text": "(No speech detected)",
                "doctor_reply": "ما سمعت والو، عاود هضر عافاك.",
                "audio_base64": ""
            }

        # 2. RAG Retrieval
        retriever = get_retriever()
        medical_context = retriever.get_context(user_text)
        
        # 3. Gemini Query with Session Tracking
        client = get_gemini_client()
        
        # Initialize session if not exists
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
            
        # Append user text with context
        contextual_prompt = f"[Medical Context]:\n{medical_context if medical_context else 'No context found.'}\n\n[User]: {user_text}"
        user_sessions[session_id].append({"role": "user", "parts": [{"text": contextual_prompt}]})
        
        # Enforce max depth of 20 exchanges (40 messages total) + initial 2
        if len(user_sessions[session_id]) > 42:
            user_sessions[session_id] = user_sessions[session_id][:2] + user_sessions[session_id][-40:]
        
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_sessions[session_id],
            )
            doctor_reply = response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                doctor_reply = "عفواً، النظام عليه ضغط كبير دابا (Quota). المرجو انتظار بضع ثواني وإعادة المحاولة."
                # Remove recent failed prompt from memory 
                if user_sessions[session_id] and user_sessions[session_id][-1]["role"] == "user":
                    user_sessions[session_id].pop()
            else:
                raise e
        
        # Append model reply, storing only the final reply without context for memory cleanliness
        user_sessions[session_id][-1]["parts"][0]["text"] = user_text # Clean history to pure user query
        user_sessions[session_id].append({"role": "model", "parts": [{"text": doctor_reply}]})

        # 4. Text to Speech (Azure)
        audio_b64 = _azure_tts(doctor_reply)
        
        return {
            "user_text": user_text,
            "doctor_reply": doctor_reply,
            "audio_base64": audio_b64
        }
    finally:
        # Cleanup input file
        if os.path.exists(in_path):
            os.remove(in_path)

def generate_medical_advice_for_prediction(prediction_label: str, confidence: float, session_id: str = "default") -> dict:
    client = get_gemini_client()
    
    # --- RAG Integration ---
    retriever = get_retriever()
    background_context = retriever.get_context(prediction_label)
    
    # Initialize session if not exists
    if session_id not in user_sessions:
        sys_prompt = (
            "أنت طبيب مغربي رقمي ومحترف ومتعاطف. تتحدث بالدارجة المغربية فقط بشكل مريح وودود.\n\n"
            "RULES FOR BEHAVIOR:\n"
            "1. NEVER repeat the same greetings or closings. Vary your conversational flow naturally.\n"
            "2. If symptoms or results are vague, do not diagnose extreme diseases automatically. Ask logical follow-up questions.\n"
            "3. Answer strictly based on the provided CLINICAL BACKGROUND if it exists."
        )
        user_sessions[session_id] = [
            {"role": "user", "parts": [{"text": sys_prompt}]},
            {"role": "model", "parts": [{"text": "مرحبا! أنا طبيبك الرقمي."}]}
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
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            doctor_reply = "عفواً، الطبيب الرقمي مشغول حالياً بكثرة الطلبات (Quota). المرجو محاولة الفحص مرة أخرى بعد قليل."
            if user_sessions[session_id] and user_sessions[session_id][-1]["role"] == "user":
                user_sessions[session_id].pop()
        else:
            raise e
    
    # Append model reply
    user_sessions[session_id].append({"role": "model", "parts": [{"text": doctor_reply}]})
    
    # Text to Speech (Azure)
    audio_b64 = _azure_tts(doctor_reply)
    
    return {
        "advice_text": doctor_reply,
        "advice_audio_base64": audio_b64
    }
