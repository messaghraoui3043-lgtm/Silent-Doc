import sys
import os
from pathlib import Path

# Fix paths for imports
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
os.chdir(root_dir)

from models.voice_model import generate_medical_advice_for_prediction

def test_darija():
    # Force UTF-8 for console output on Windows
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("Testing 'Perfect Darija' Advice Generation...")
    try:
        result = generate_medical_advice_for_prediction(
            prediction_label="Melanocytic Nevus",
            confidence=92.5,
            session_id="test_verify",
            language="Darija"
        )
        print("\n--- DOCTOR REPLY ---")
        print(result["advice_text"])
        print("\n--- AUDIO GENERATED (B64 SIZE) ---")
        print(len(result["advice_audio_base64"]))
        
    except Exception as e:
        import traceback
        print(f"Test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_darija()
