# Silent-Doc 🩺
A premium, multimodal, local-first AI medical assistant. 

Silent-Doc allows patients to submit medical images (skin lesions or eye OCT scans) to local diagnostic AI models. It features a conversational module built with Whisper and Gemini, enabling voice-based consultations in Moroccan Darija.

## Features
- **Skin Lesion Detection**: Uses a MobileNetV2 architecture.
- **Eye Disease Detection**: Uses an OCTResnet PyTorch architecture.
- **Voice Interactivity**: Speaks Moroccan Darija via STT (Whisper) and LLM reasoning (Gemini Flash).
- **Web UI & Bot**: Responsive Glassmorphism dashboard and Telegram bot integration.

## Installation
1. Clone the repository and navigate to the directory.
2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Set your Google Gemini API key:
    Open the `.env` file and place your `GEMINI_API_KEY` inside.

## Usage
To start the FastAPI backend and web server:
```bash
python api/main.py
```
Then navigate to: `http://localhost:5000/web` in your browser.
