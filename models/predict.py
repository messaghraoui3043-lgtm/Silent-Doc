"""
Silent Doctor - OCT Eye Disease Inference Script
=================================================
This script handles inference for the ResNet18 model trained on OCT images.
Classes: CNV, DME, DRUSEN, Normal

Usage:
    predictor = EyeDiseasePredictor("OCTResnet.pth")
    result = predictor.predict("image.jpg")
"""

import sys
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as tt
from PIL import Image
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Recreating the base architecture as defined in the notebook.
# This ensures that PyTorch's pickle loading doesn't fail if the full model object was saved.
class OCTresbase(nn.Module):
    pass

class OCTres(OCTresbase):
    def __init__(self):
        super().__init__()
        # Initialize an empty ResNet18 model
        self.network = models.resnet18(weights=None)
        
        # We freeze layers matching notebook structure
        for param in self.network.fc.parameters():
            param.requires_grad = False
            
        num_features = self.network.fc.in_features 
        self.network.fc = nn.Linear(num_features, 4) 
    
    def forward(self, xb):
        return self.network(xb)


class EyeDiseasePredictor:
    """
    Prediction wrapper that handles loading the PyTorch model,
    preprocessing the inputs, and executing forward inference.
    """
    def __init__(self, model_path: str = "OCTResnet.pth", device: str = "cpu"):
        self.device = torch.device(device)
        self.classes = ['CNV', 'DME', 'DRUSEN', 'Normal']
        self.model_path = Path(model_path)
        
        # Standard preprocessing pipeline derived from notebook
        self.transform = tt.Compose([
            tt.Resize(128),
            tt.CenterCrop(128),
            tt.ToTensor(),
        ])
        
        self.model = OCTres()
        self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        if self.model_path.exists():
            logger.info(f"Loading weights from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info("Successfully loaded model weights.")
        else:
            logger.warning(
                f"Model file '{self.model_path}' not found!\n"
                f"Please run the notebook to train the model and generate '{self.model_path}',\n"
                f"or place a pre-trained weights file at this path. Using untrained model for now."
            )

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Prepares the PIL Image for validation/testing by resizing and converting to Tensor."""
        # The model inherently uses 3 channels (RGB)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image_source) -> dict:
        """
        Runs prediction on a given image.

        Args:
            image_source: Path to the image file, or a file-like object (e.g., from Flask).

        Returns:
            dict containing prediction label & confidence score.
        """
        try:
            img = Image.open(image_source)
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            raise ValueError(f"Invalid image source provided: {e}")

        tensor = self.preprocess(img)
        
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        
        top_idx = probabilities.argmax().item()
        top_class = self.classes[top_idx]
        top_confidence = probabilities[top_idx].item()
        
        return {
            "prediction": top_class,
            "confidence": round(top_confidence, 4)
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <image_path>")
        sys.exit(1)
        
    image_to_predict = sys.argv[1]
    # Assuming OCTResnet.pth is placed in the same directory for testing
    local_weights = Path(__file__).parent / "OCTResnet.pth"
    predictor = EyeDiseasePredictor(model_path=str(local_weights))
    
    result = predictor.predict(image_to_predict)
    print(f"\nPrediction Results for {image_to_predict}:")
    print(f"Class: {result['prediction']} (Confidence: {result['confidence']:.2%})")
