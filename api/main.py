from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, Depends, Security
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
import uvicorn
import io
import sys
import os
import logging
from logging.handlers import RotatingFileHandler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from pathlib import Path

# Ensure root directory is on the path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Create required directories
os.makedirs(os.path.join(root_dir, "audio", "tmp"), exist_ok=True)
os.makedirs(os.path.join(root_dir, "logs"), exist_ok=True)

# ----------------- LOGGING SETUP -----------------
logger = logging.getLogger("SilentDocAPI")
logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler(os.path.join(root_dir, "logs", "silent_doc.log"), maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
# -------------------------------------------------

from models.skin_model import get_skin_model
from models.eye_model import get_eye_model
from PIL import Image
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Silent Doctor API",
    description="Unified API for Skin Lesion and Eye Disease Classifiers.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ----------------- SECURITY SETUP -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://127.0.0.1:5000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("SILENT_DOC_API_KEY", "silent-doc-dev-key")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        logger.warning(f"Failed API Key attempt.")
        raise HTTPException(status_code=403, detail="Could not validate API credentials")
    return api_key

MAX_FILE_SIZE = 8 * 1024 * 1024 # 8MB
def validate_file_size(file_bytes: bytes):
    if len(file_bytes) > MAX_FILE_SIZE:
        logger.warning(f"File upload rejected: {len(file_bytes)} bytes > 8MB")
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 8MB.")
# --------------------------------------------------

# Mount the static web interface
app.mount("/web", StaticFiles(directory=os.path.join(root_dir, "web"), html=True), name="web")

from fastapi.responses import HTMLResponse

@app.get("/")
@limiter.limit("30/minute")
def serve_index(request: Request):
    html_path = os.path.join(root_dir, "web", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict/skin")
@limiter.limit("10/minute")
async def predict_skin(request: Request, file: UploadFile = File(...), session_id: str = Form("default"), language: str = Form("Darija"), api_key: str = Depends(verify_api_key)):
    """Skin Lesion prediction with limit & auth"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    try:
        image_bytes = await file.read()
        validate_file_size(image_bytes)
        
        from models.skin_model import predict_image
        result = predict_image(image_bytes)
        results = result["top_k"]
        heatmap_b64 = result.get("heatmap_base64", "")
        top_prediction = results[0]
        
        # --- GUARDRAIL ---
        if top_prediction["confidence"] < 40.0:
            logger.info("Skin prediction rejected by guardrail: < 40 confidence")
            return JSONResponse({
                "predictions": results,
                "heatmap_base64": heatmap_b64,
                "advice_text": "الصورة غير واضحة، عاود جرب بصورة خرى.",
                "advice_audio_base64": "",
                "disclaimer": "⚠️ MEDICAL DISCLAIMER: This AI is a diagnostic aid only and is NOT a substitute for professional medical advice."
            })
        
        from models.voice_model import generate_medical_advice_for_prediction
        advice_data = generate_medical_advice_for_prediction(top_prediction["label"], top_prediction["confidence"], session_id=session_id, language=language)
            
    except Exception as e:
        import traceback
        err_str = traceback.format_exc()
        logger.error(f"Prediction failed: {str(e)}\n\n{err_str}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    logger.info(f"Skin prediction successful for session {session_id}")
    return JSONResponse({
        "predictions": results,
        "heatmap_base64": heatmap_b64,
        "advice_text": advice_data["advice_text"],
        "advice_audio_base64": advice_data["advice_audio_base64"],
        "disclaimer": "⚠️ MEDICAL DISCLAIMER: This AI is a diagnostic aid only and is NOT a substitute for professional medical advice."
    })


@app.post("/predict/eye")
@limiter.limit("10/minute")
async def predict_eye(request: Request, file: UploadFile = File(...), session_id: str = Form("default"), language: str = Form("Darija"), api_key: str = Depends(verify_api_key)):
    """Eye Disease prediction with Voice Advice, limit & auth"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")
    
    try:
        image_bytes = await file.read()
        validate_file_size(image_bytes)
        
        predictor = get_eye_model()
        result = predictor.predict(image_bytes)
        
        results = result["top_k"]
        top_prediction = results[0]
        
        from models.voice_model import generate_medical_advice_for_prediction
        advice_data = generate_medical_advice_for_prediction(top_prediction["label"], top_prediction["confidence"], session_id=session_id, language=language)
        
        result["advice_text"] = advice_data["advice_text"]
        result["advice_audio_base64"] = advice_data["advice_audio_base64"]
        result["disclaimer"] = "⚠️ MEDICAL DISCLAIMER: This AI is a diagnostic aid only and is NOT a substitute for professional medical advice."
        
        logger.info(f"Eye prediction successful for session {session_id}")
        return JSONResponse(result)
        
    except Exception as e:
        import traceback
        err_str = traceback.format_exc()
        logger.error(f"Eye Prediction failed: {str(e)}\n\n{err_str}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/acne")
@limiter.limit("10/minute")
async def predict_acne(request: Request, file: UploadFile = File(...), session_id: str = Form("default"), language: str = Form("Darija"), api_key: str = Depends(verify_api_key)):
    """Acne prediction with limit & auth"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")
    
    try:
        image_bytes = await file.read()
        validate_file_size(image_bytes)
        
        # --- ROBOFLOW INFERENCE PIPELINE ---
        from inference_sdk import InferenceHTTPClient
        import io
        import base64
        import numpy as np
        from PIL import Image, ImageDraw
        from dotenv import load_dotenv
        
        load_dotenv()
        roboflow_key = os.getenv("ROBOFLOW_API_KEY")
        if not roboflow_key:
            logger.error("ROBOFLOW_API_KEY is not set in the environment.")
            raise HTTPException(status_code=500, detail="Acne detection service is not configured (missing API key).")
            
        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=roboflow_key
        )
        
        # Pass the PIL Image directly. The Roboflow Inference SDK handles color space (RGB/BGR) 
        # internally for PIL Images, preventing any manual numpy channel corruption.
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        try:
            inference_result = CLIENT.infer(img, model_id="acne-ygqhs/1")
            print("ROBOFLOW RAW API OUTPUT:", inference_result)
        except Exception as api_e:
            logger.error(f"Roboflow API error: {api_e}")
            raise HTTPException(status_code=502, detail="External acne detection service failed.")
            
        # Extract predictions correctly from the Roboflow InferenceResponse dictionary
        try:
            if hasattr(inference_result, "dict"):
                inference_result = inference_result.dict() # Convert InferenceResponse object to dict if needed
        except:
            pass
            
        predictions = inference_result.get("predictions", [])
        
        valid_spots = 0
        draw = ImageDraw.Draw(img)
        # The new Roboflow model contains 5 specific classes. We treat all of them as valid acne.
        valid_acne_classes = ["blackheads", "nodules", "papules", "pustules", "whiteheads"]
        
        for p in predictions:
            conf = p.get("confidence", 0)
            class_name = str(p.get("class", "")).lower()
            
            # Accept ANY spot that has confidence >= 0.15, regardless of the class name, 
            # but we explicitly log the specific subclass detected for debugging.
            if conf >= 0.15:
                print(f"Detected valid acne spot -> Class: {class_name}, Confidence: {conf}")
                valid_spots += 1
                x = p["x"]
                y = p["y"]
                w = p["width"]
                h = p["height"]
                
                # Convert center x,y,w,h to top-left and bottom-right
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2
                
                # Draw Bounding Box (Red, 3px width)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                # Draw Label
                draw.text((x1, max(y1 - 10, 0)), f"{class_name.capitalize()} {conf:.2f}", fill="red")
                
        # Re-encode the image with drawn boxes back to Base64
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Build standard output structure expected by the rest of the app
        # Added "rank": 1 to fix the frontend "#undefined" UI bug
        results = [{"label": f"{valid_spots} Acne Spots Detected" if valid_spots > 0 else "No acne", "confidence": 100.0, "rank": 1}]
        top_prediction = results[0]
        # -----------------------------------
        
        if "No acne" in top_prediction["label"]:
            logger.info("Acne prediction clear (no spots)")
            return JSONResponse({
                "predictions": results,
                "heatmap_base64": heatmap_b64,
                "advice_text": "وجهك صافي تبارك الله! ما كاين حتى حبة.",
                "advice_audio_base64": "",
                "disclaimer": "⚠️ MEDICAL DISCLAIMER: This AI is a diagnostic aid only and is NOT a substitute for professional medical advice."
            })
        
        from models.voice_model import generate_medical_advice_for_prediction
        advice_data = generate_medical_advice_for_prediction(top_prediction["label"], top_prediction["confidence"], session_id=session_id, language=language)
            
    except Exception as e:
        import traceback
        err_str = traceback.format_exc()
        logger.error(f"Acne Prediction failed: {str(e)}\n\n{err_str}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    logger.info(f"Acne prediction successful for session {session_id}")
    return JSONResponse({
        "predictions": results,
        "heatmap_base64": heatmap_b64,
        "advice_text": advice_data["advice_text"],
        "advice_audio_base64": advice_data["advice_audio_base64"],
        "disclaimer": "⚠️ MEDICAL DISCLAIMER: This AI is a diagnostic aid only and is NOT a substitute for professional medical advice."
    })


@app.post("/predict/voice")
@limiter.limit("20/minute")
async def predict_voice(request: Request, file: UploadFile = File(...), session_id: str = Form("default"), language: str = Form("Darija"), api_key: str = Depends(verify_api_key)):
    """Voice Consultation endpoint with limit & auth"""
    if not file.content_type.startswith("audio/") and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only audio files are accepted.")
    
    try:
        audio_bytes = await file.read()
        validate_file_size(audio_bytes)
        
        from fastapi.responses import StreamingResponse
        from models.voice_model import process_voice_consultation_stream
        
        logger.info(f"Starting voice stream for session {session_id}")
        return StreamingResponse(process_voice_consultation_stream(audio_bytes, session_id=session_id, language=language), media_type="text/event-stream")
        
    except Exception as e:
        import traceback
        err_str = traceback.format_exc()
        logger.error(f"Voice processing failed: {str(e)}\n\n{err_str}")
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")

if __name__ == '__main__':
    uvicorn.run("api.main:app", host='127.0.0.1', port=5000, reload=True)
