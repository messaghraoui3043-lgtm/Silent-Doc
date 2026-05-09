from google import genai

# حط الساروت ديالك هنا
client = genai.Client(api_key="AIzaSyC3Z6cmEudIv0gMEjqHeHiX6w6JxFi7IGM")

print("--- جاري البحث عن الموديلات المتاحة للـ API Key ديالك ---")
try:
    for model in client.models.list():
        print(f"✅ متاح: {model.name}")
except Exception as e:
    print(f"❌ كاين مشكل: {e}")