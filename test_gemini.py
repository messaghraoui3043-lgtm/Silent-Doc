import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

try:
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=["Translate to french:", "Hello world"]
    )
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("FAIL:", e)
