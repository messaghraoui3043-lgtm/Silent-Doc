import os
import gradio as gr
import whisper
import tempfile
import torch
import numpy as np
from PIL import Image
from google import genai
from gtts import gTTS
from pathlib import Path
import sys

# Ensure root directory is on the path for model imports
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from models.skin_model import predict_image
except ImportError:
    # Fallback if models directory is not found
    def predict_image(image_bytes, top_k=3):
        return [{"label": "Model Not Loaded", "confidence": 0.0}]

# Initialize Models
print("--- Loading models... ---")
whisper_model = whisper.load_model("medium")
client = genai.Client(api_key="AIzaSyC75E2J_OtbxaoqwKUHkygtYOiDhs2HzWc")

# Custom CSS for the Premium Dark Theme
CSS = """
.gradio-container {
    background-color: #0b0f19 !important;
    color: #ffffff !important;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
.sidebar-card {
    background: #161b2a;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #2d3343;
}
.prediction-card {
    background: #1c2333;
    border-radius: 10px;
    padding: 15px;
    margin-top: 10px;
    border-left: 4px solid #3b82f6;
}
.prediction-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.rank {
    background: #3b82f677;
    color: #60a5fa;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: bold;
}
.label {
    flex-grow: 1;
    margin-left: 10px;
    font-weight: 500;
}
.confidence {
    font-weight: bold;
    color: #94a3b8;
}
.progress-bar-bg {
    background: #2d3343;
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
}
.progress-fill {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    height: 100%;
}
.disclaimer-card {
    background: #271c11;
    border: 1px solid #fbbf24;
    border-radius: 8px;
    padding: 12px;
    margin-top: 15px;
    color: #fcd34d;
    font-size: 0.9em;
}
"""

def format_prediction_html(predictions):
    html = '<div style="margin-bottom: 20px; color: white; border-radius: 10px;">'
    html += '<h3 style="margin-bottom: 10px;">Here are the top-3 predictions for your image:</h3>'
    for pred in predictions:
        conf = f"{pred['confidence']}%"
        # Inline styling for maximum compatibility
        html += f"""
        <div style="background: #1c2333; border-radius: 10px; padding: 15px; margin-top: 10px; border-left: 4px solid #3b82f6; color: white;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="background: #3b82f677; color: #60a5fa; padding: 2px 8px; border-radius: 20px; font-size: 0.8em; font-weight: bold;">#{pred['rank']}</span>
                <span style="flex-grow: 1; margin-left: 10px; font-weight: 500;">{pred['label']}</span>
                <span style="font-weight: bold; color: #94a3b8;">{conf}</span>
            </div>
            <div style="background: #2d3343; height: 6px; border-radius: 3px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #3b82f6, #8b5cf6); height: 100%; width: {conf};"></div>
            </div>
        </div>
        """
    html += '<div style="background: #271c11; border: 1px solid #fbbf24; border-radius: 8px; padding: 12px; margin-top: 15px; color: #fcd34d; font-size: 0.9em;">⚠️ MEDICAL DISCLAIMER: This AI is a diagnostic aid only and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for any skin-related concerns.</div>'
    html += '</div>'
    # Remove any extra newlines/tabs that could trigger Markdown code blocks
    return "".join([line.strip() for line in html.split("\n")])

def process_diagnostic(image, history):
    if image is None:
        return history
    
    try:
        # Convert PIL Image to bytes
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
        
        # Run local model prediction
        predictions = predict_image(image_bytes)
        html_output = format_prediction_html(predictions)
        
        # Append to history with a clear user message
        history.append({"role": "user", "content": "Analyzing image..."})
        history.append({"role": "assistant", "content": html_output})
        
        return history
    except Exception as e:
        history.append({"role": "assistant", "content": f"Prediction failed: {str(e)}"})
        return history

def process_voice(audio_path, history):
    if not audio_path:
        return history, None
    
    try:
        # 1. Transcribe
        darija_prompt = "هذه محادثة بالدارجة المغربية: واش، مزيان، بزاف، دابا، واخا، عافاك، ديالي، شنو، كيفاش."
        result = whisper_model.transcribe(audio_path, language="ar", initial_prompt=darija_prompt)
        user_text = result["text"].strip()
        
        if not user_text:
            history.append({"role": "user", "content": "(No speech detected)"})
            history.append({"role": "assistant", "content": "ما سمعت والو، عاود هضر عافاك."})
            return history, None

        # 2. Call Gemini
        sys_prompt = "أنت طبيب مغربي رقمي. تحاور مع المريض، اسأله إذا احتجت لمعلومات، وأجب بالدارجة المغربية فقط وبشكل مختصر جدا ومهني. المريض يقول: "
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=sys_prompt + user_text,
        )
        doctor_reply = response.text
        
        # 3. Add to history
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": doctor_reply})
        
        # 4. Generate TTS
        tts = gTTS(text=doctor_reply, lang="ar")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        
        return history, temp_audio.name
    except Exception as e:
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history, None

with gr.Blocks() as demo:
    with gr.Row():
        gr.HTML("""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 2em;">🩺</span>
                <h1 style="margin: 0; color: white;">Silent Doctor</h1>
            </div>
            <div style="background: #059669; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 0.9em;">
                AI Powered
            </div>
        </div>
        """)

    with gr.Row():
        # LEFT COLUMN - Upload
        with gr.Column(scale=1):
            with gr.Group(elem_classes="sidebar-card"):
                gr.Markdown("### UPLOAD IMAGE")
                image_input = gr.Image(label="", type="pil", elem_id="image-upload")
                analyze_btn = gr.Button("🔍 Analyse Image", variant="primary")
            
            with gr.Group(elem_classes="sidebar-card", visible=True):
                gr.Markdown("### API Endpoint")
                api_url = gr.Textbox(value="http://localhost:8000/predict", label="", interactive=True)
        
        # RIGHT COLUMN - Chatbot
        with gr.Column(scale=2):
            # sanitize_html=False allows us to render our custom prediction cards
            chatbot = gr.Chatbot(label="Silent Doctor Assistant", height=600, sanitize_html=False)
            with gr.Row():
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Vocal Consultation")
                audio_output = gr.Audio(visible=False, autoplay=True)

    # Event Handlers
    analyze_btn.click(
        fn=process_diagnostic,
        inputs=[image_input, chatbot],
        outputs=[chatbot]
    )
    
    audio_input.stop_recording(
        fn=process_voice,
        inputs=[audio_input, chatbot],
        outputs=[chatbot, audio_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=CSS, theme=gr.themes.Default())
