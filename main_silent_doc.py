# import os
# import sys
# import whisper
# import speech_recognition as sr
# from google import genai
# from gtts import gTTS

# # --- مكتبات إصلاح اللغة العربية فـ الـ Terminal ---
# import arabic_reshaper
# from bidi.algorithm import get_display

# def print_arabic(text):
#     """هادي دالة كتقاد الحروف وتلاصقهم باش يتقراو مزيان فـ الويندوز"""
#     try:
#         reshaped_text = arabic_reshaper.reshape(text)    # تلاصق الحروف
#         bidi_text = get_display(reshaped_text)           # تقلب الاتجاه من اليمين لليسار
#         print(bidi_text)
#     except:
#         print(text) # إلى وقع شي مشكل، طبعو عادي

# # 1. إعداد مسار FFmpeg 
# os.environ["PATH"] += os.path.pathsep + os.getcwd()

# # 2. إعداد Gemini API
# client = genai.Client(api_key="AIzaSyBzaLPpwu-5OhgeZWpkLVmsfrOB5XCTTKw")

# # 3. تحميل محرك Whisper
# print_arabic("--- جاري تحميل محرك الذكاء الاصطناعي للصوت (Whisper)... ---")
# whisper_model = whisper.load_model("base")

# def voice_consultation():
#     print("\n" + "="*50)
#     print_arabic(" 🩺 مرحبا بك في Silent-Doc (النسخة الصوتية) ")
#     print("="*50)
    
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         print_arabic("\n[Silent-Doc]: راني كنسمعك، شنو هو المشكل اللي عندك؟ (هضر بالدارجة دابا)")
#         r.adjust_for_ambient_noise(source)
#         audio = r.listen(source)
        
#         with open("input.wav", "wb") as f:
#             f.write(audio.get_wav_data())

#     print_arabic("--- جاري تحليل الكلام... ---")
#     try:
#         # 4. تحويل الصوت ديالك لنص
#         result = whisper_model.transcribe("input.wav", language="ar")
#         user_speech = result["text"].strip()
        
#         if not user_speech:
#             print_arabic("❌ ماسمعت والو، تأكد باللي الميكروفون خدام.")
#             return

#         print_arabic(f"\n🗣️ أنت قلتي: {user_speech}")

#         # 5. استشارة Gemini
#         print_arabic("--- جاري استشارة الطبيب الرقمي... ---")
#         sys_prompt = "أنت طبيب مغربي رقمي. أجب بالدارجة المغربية فقط وبشكل مختصر جدا ومهني. المريض يقول: "
#         response = client.models.generate_content(
#             model='gemini-2.5-flash',
#             contents=sys_prompt + user_speech
#         )
        
#         doctor_reply = response.text
#         print_arabic(f"\n👨‍⚕️ [Silent-Doc]: {doctor_reply}\n")

#         # 6. تحويل الجواب لصوت
#         tts = gTTS(text=doctor_reply, lang='ar')
#         tts.save("output.mp3")
#         os.system("start output.mp3")
        
#     except Exception as e:
#         print_arabic(f"❌ وقع خطأ تقني: {e}")

# if __name__ == "__main__":
#     voice_consultation()




import os
import sys
import whisper
import speech_recognition as sr
from google import genai
from gtts import gTTS

# --- مكتبات إصلاح اللغة العربية فـ الـ Terminal ---
import arabic_reshaper
from bidi.algorithm import get_display

def print_arabic(text):
    """هادي دالة كتقاد الحروف وتلاصقهم باش يتقراو مزيان فـ الويندوز"""
    try:
        reshaped_text = arabic_reshaper.reshape(text)    
        bidi_text = get_display(reshaped_text)           
        print(bidi_text)
    except:
        print(text) 

# 1. إعداد مسار FFmpeg 
os.environ["PATH"] += os.path.pathsep + os.getcwd()

# 2. إعداد Gemini API (ما تنساش تحط الساروت الجديد ديالك)
client = genai.Client(api_key="AIzaSyDkoJXb0S1VKYze8WDSOH0eKFvGyRBNxEU")

# 3. تحميل محرك Whisper
print_arabic("--- جاري تحميل محرك الذكاء الاصطناعي للصوت (Whisper)... قد يستغرق هذا بعض الوقت لتحميل النموذج ---")
whisper_model = whisper.load_model("medium") # استخدمت "medium" لأنه أقوى بكثير في فهم الدارجة

def voice_consultation():
    print("\n" + "="*50)
    print_arabic(" 🩺 مرحبا بك في Silent-Doc (النسخة المتزامنة الصوت/النص) ")
    print("="*50)
    
    r = sr.Recognizer()
    
    while True:
        with sr.Microphone() as source:
            print_arabic("\n[Silent-Doc]: راني كنسمعك، شنو هو المشكل اللي عندك؟ (باش تحبس قول 'سالينا' أو 'صافي')")
            r.adjust_for_ambient_noise(source)
            try:
                audio = r.listen(source)
            except KeyboardInterrupt:
                print_arabic("👋 بسلامة! الله يجيب الشفاء.")
                break
            
            with open("input.wav", "wb") as f:
                f.write(audio.get_wav_data())

        print_arabic("--- جاري تحليل الكلام... ---")
        try:
            # --- المرحلة الأولى: صوت المريض ---
            # 1. تحويل الصوت لنص، مع إضافة prompt ليساعد في فهم الدارجة
            darija_prompt = "هذه محادثة بالدارجة المغربية: واش، مزيان، بزاف، دابا، واخا، عافاك، ديالي، شنو، كيفاش."
            result = whisper_model.transcribe("input.wav", language="ar", initial_prompt=darija_prompt)
            user_speech = result["text"].strip()
            
            if not user_speech:
                print_arabic("❌ ماسمعت والو، عاود هضر عافاك.")
                continue

            # 2. طباعة النص ديالك فـ الشاشة
            print_arabic(f"\n🗣️ أنت قلتي: {user_speech}")
            
            # Check for stop words
            stop_words = ["سالينا", "صافي", "حبس", "قف", "stop", "exit", "quit", "safy", "salina"]
            if any(word in user_speech.lower() for word in stop_words):
                print_arabic("👋 بسلامة! نتمنى لك الشفاء العاجل.")
                break

            # 3. تشغيل الصوت مباشرة من بعد الطباعة (باش تقرا وتسمع)
            os.system("start input.wav")

            # --- المرحلة الثانية: جواب الطبيب ---
            print_arabic("--- جاري استشارة الطبيب الرقمي... ---")
            sys_prompt = "أنت طبيب مغربي رقمي. تحاور مع المريض، اسأله إذا احتجت لمعلومات، وأجب بالدارجة المغربية فقط وبشكل مختصر جدا ومهني. المريض يقول: "
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=sys_prompt + user_speech
            )
            doctor_reply = response.text

            # 4. تحويل جواب الطبيب لصوت (gTTS)
            tts = gTTS(text=doctor_reply, lang='ar')
            tts.save("output.mp3")

            # 5. طباعة جواب الطبيب فـ الشاشة
            print_arabic(f"\n👨‍⚕️ [Silent-Doc]: {doctor_reply}\n")
            
            # 6. تشغيل صوت الطبيب مباشرة من بعد الطباعة
            os.system("start output.mp3")
            
        except Exception as e:
            print_arabic(f"❌ وقع خطأ تقني: {e}")
            print_arabic("غانحاول نعاود نسمعك...")

if __name__ == "__main__":
    try:
        voice_consultation()
    except KeyboardInterrupt:
        print("\nGoodbye!")
