import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def check_env():
    print("====================================")
    print("Silent-Doc Environment Verification")
    print("====================================")
    
    # 1. Directories
    root = Path(__file__).resolve().parent
    expected_dirs = ["api", "bot", "cli", "core", "models", "utils", "config"]
    missing_dirs = [d for d in expected_dirs if not (root / d).exists()]
    
    if missing_dirs:
        print(f"❌ Missing expected directories: {', '.join(missing_dirs)}")
    else:
        print("✅ All core directories present.")

    # 2. Weights
    weights_dir = root / "models" / "weights"
    if not weights_dir.exists():
        print("❌ Cannot find models/weights directory!")
    else:
        h5_path = weights_dir / "mobilenet_model.h5"
        pth_path = weights_dir / "OCTResnet.pth"
        
        if h5_path.exists():
            print("✅ Skin Model weights found.")
        else:
            print("⚠️ Skin Model weights missing. (Ensure you run training notebook or provide .h5)")

        if pth_path.exists():
            print("✅ Eye Model weights found.")
        else:
            print("⚠️ Eye Model weights missing. (Ensure you run training notebook or provide .pth)")
            
    print("====================================")
    print("Run endpoints with:")
    print("1. API: uvicorn api.main:app --reload --port 5000")
    print("2. Bot: python bot/telegram_bot.py")
    print("3. CLI: python cli/interactive.py")
    print("4. Voice: python cli/voice_consultation.py")

if __name__ == "__main__":
    check_env()
