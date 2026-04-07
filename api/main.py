from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import io
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from pathlib import Path

# Ensure root directory is on the path
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Create audio/tmp directory if it doesn't exist
os.makedirs(os.path.join(root_dir, "audio", "tmp"), exist_ok=True)

from models.skin_model import get_skin_model
from models.eye_model import get_eye_model
from PIL import Image

app = FastAPI(
    title="Silent Doctor API",
    description="Unified API for Skin Lesion and Eye Disease Classifiers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static web interface
app.mount("/web", StaticFiles(directory=os.path.join(str(Path(__file__).resolve().parent.parent), "web"), html=True), name="web")

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "message": "Silent Doctor API is running. Available endpoints: /predict/skin, /predict/eye"
    }

@app.post("/predict/skin")
async def predict_skin(file: UploadFile = File(...), session_id: str = Form("default")):
    """Skin Lesion top-3 prediction using MobileNetv2"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    try:
        image_bytes = await file.read()
        import numpy as np
        from models.skin_model import preprocess_image, get_model, LESION_CLASSES, LESION_SHORT
        
        # We can either use get_skin_model().predict by saving temporarily, 
        # or just reuse the old method. Here we use the raw method to avoid disk I/O.
        processed = preprocess_image(image_bytes)
        model = get_model()
        probs = model.predict(processed, verbose=0)[0]
        
        top_k = 3
        top_indices = np.argsort(probs)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append({
                "rank": rank,
                "label": LESION_CLASSES[idx],
                "code": LESION_SHORT[idx],
                "confidence": float(round(probs[idx] * 100, 2)),
            })
        top_prediction = results[0]
        
        # --- NEW GUARDRAIL ---
        if top_prediction["confidence"] < 40.0:
            return JSONResponse({
                "predictions": results,
                "advice_text": "Imagen غير واضحة، عاود جرب بصورة خرى.",
                "advice_audio_base64": "",
                "disclaimer": "⚠️ MEDICAL DISCLAIMER: This AI is a diagnostic aid only and is NOT a substitute for professional medical advice."
            })
        # ---------------------
        
        from models.voice_model import generate_medical_advice_for_prediction
        advice_data = generate_medical_advice_for_prediction(top_prediction["label"], top_prediction["confidence"], session_id=session_id)
            
    except Exception as e:
        import traceback
        err_str = traceback.format_exc()
        with open("error.log", "w", encoding="utf-8") as f:
            f.write(err_str)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}\n\n{err_str}")

    return JSONResponse({
        "predictions": results,
        "advice_text": advice_data["advice_text"],
        "advice_audio_base64": advice_data["advice_audio_base64"],
        "disclaimer": "⚠️ MEDICAL DISCLAIMER: This AI is a diagnostic aid only and is NOT a substitute for professional medical advice."
    })


@app.post("/predict/eye")
async def predict_eye(file: UploadFile = File(...)):
    """Eye Disease prediction using OCTResnet"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")
    
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Use our singleton predictor
        predictor = get_eye_model()
        
        # Preprocess directly via PIL Image rather than writing to disk
        tensor = predictor.preprocess(img)
        
        import torch
        logits = predictor.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        
        top_idx = probabilities.argmax().item()
        top_class = predictor.classes[top_idx]
        top_confidence = probabilities[top_idx].item()
        
        result = {
            "prediction": top_class,
            "confidence": round(top_confidence, 4)
        }
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/voice")
async def predict_voice(file: UploadFile = File(...), session_id: str = Form("default")):
    """Voice Consultation endpoint"""
    if not file.content_type.startswith("audio/") and not file.content_type.startswith("video/"):
        # browsers sometimes send audio as video/webm
        raise HTTPException(status_code=400, detail="Only audio files are accepted.")
    
    try:
        audio_bytes = await file.read()
        from models.voice_model import process_voice_consultation
        result = process_voice_consultation(audio_bytes, session_id=session_id)
        return JSONResponse(result)
        
    except Exception as e:
        import traceback
        err_str = traceback.format_exc()
        with open("error.log", "w", encoding="utf-8") as f:
            f.write(err_str)
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}\n\n{err_str}")

if __name__ == '__main__':
    uvicorn.run("api.main:app", host='0.0.0.0', port=5000, reload=True)
