from flask import Flask, request, jsonify
from pathlib import Path
import os
import sys

# Ensure the root directory is on the path so we can import models.predict
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.predict import EyeDiseasePredictor

app = Flask(__name__)

# Configure the paths
BASE_DIR = Path(__file__).resolve().parent
# Typically weights can be placed inside models/weights or alongside predict.py
MODEL_PATH = BASE_DIR / "models" / "OCTResnet.pth"

# Initialize predictor globally
predictor = EyeDiseasePredictor(model_path=str(MODEL_PATH))

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online", 
        "model_loaded": MODEL_PATH.exists()
    })

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Endpoint that accepts an image file as multipart/form-data.
    Input form field: 'file'
    Returns JSON: {"prediction": "class_name", "confidence": score}
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file parameter in request. Use 'file'."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Pass the file stream directly to PIL via sequence Predictor
        result = predictor.predict(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
