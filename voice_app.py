import os
import gradio as gr
import whisper
from google import genai
from gtts import gTTS
import tempfile
import time

# Initialize Whisper model
print("--- Loading Whisper model... ---")
whisper_model = whisper.load_model("medium")

# Initialize Gemini Client
client = genai.Client(api_key="AIzaSyC75E2J_OtbxaoqwKUHkygtYOiDhs2HzWc")

def process_audio(audio_path, history):
    if not audio_path:
        return history, None

    try:
        # 1. Transcribe with Whisper
        darija_prompt = (
            "هذه محادثة بالدارجة المغربية: واش، مزيان، بزاف، "
            "دابا، واخا، عافاك، ديالي، شنو، كيفاش."
        )
        result = whisper_model.transcribe(
            audio_path, language="ar", initial_prompt=darija_prompt
        )
        user_text = result["text"].strip()

        if not user_text:
            history.append({"role": "user", "content": " (No speech detected) "})
            history.append({"role": "assistant", "content": "ما سمعت والو، عاود هضر عافاك."})
            return history, None

        # 2. Call Gemini API
        sys_prompt = (
            "أنت طبيب مغربي رقمي. تحاور مع المريض، اسأله إذا احتجت "
            "لمعلومات، وأجب بالدارجة المغربية فقط وبشكل مختصر جدا ومهني. "
            "المريض يقول: "
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=sys_prompt + user_text,
        )
        doctor_reply = response.text

        # 3. Add to history (messages format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}])
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": doctor_reply})

        # 4. Generate TTS
        tts = gTTS(text=doctor_reply, lang="ar")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        
        return history, temp_audio.name

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append({"role": "user", "content": user_text if 'user_text' in locals() else "Error"})
        history.append({"role": "assistant", "content": f"وقع خطأ تقني: {error_msg}"})
        return history, None

# Build Gradio UI
with gr.Blocks(title="Silent-Doc Web") as demo:
    gr.Markdown("# 🩺 Silent-Doc: Digital Doctor (Darija)")
    gr.Markdown("Record your voice to start the consultation. The doctor will respond in Darija.")

    chatbot = gr.Chatbot(label="Consultation History")
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record your voice")
    
    with gr.Row():
        audio_output = gr.Audio(label="Doctor's Voice Response", autoplay=True)
        
    submit_btn = gr.Button("Send to Doctor", variant="primary")
    clear_btn = gr.Button("Clear Chat")

    # Connect components
    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, chatbot],
        outputs=[chatbot, audio_output]
    )
    
    clear_btn.click(lambda: ([], None), None, [chatbot, audio_output])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
