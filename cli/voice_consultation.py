import os
import sys
import time
import subprocess
import importlib
import threading

# ─────────────────────────────────────────────
# 1. Auto-Install Missing Libraries
# ─────────────────────────────────────────────
def install_and_import(package, import_name=None):
    import_name = import_name or package
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_and_import("arabic-reshaper", "arabic_reshaper")
install_and_import("python-bidi", "bidi")
install_and_import("pygame")
install_and_import("keyboard")          # for Push-to-Talk
install_and_import("SpeechRecognition", "speech_recognition")

# ─────────────────────────────────────────────
# 2. Standard Imports
# ─────────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

import whisper
import speech_recognition as sr
from google import genai
from gtts import gTTS
import pygame
import arabic_reshaper
from bidi.algorithm import get_display

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except Exception:
    KEYBOARD_AVAILABLE = False

pygame.mixer.init()

# ─────────────────────────────────────────────
# 3. Configuration  ← tweak these as needed
# ─────────────────────────────────────────────

# VAD: minimum RMS energy level to consider audio "speech"
# Raise this if background noise keeps triggering false requests.
VAD_ENERGY_THRESHOLD = 400          # sr default is ~300

# Maximum seconds to wait for a single phrase before giving up
PHRASE_TIME_LIMIT_SEC = 15

# Seconds to wait *after* each successful API round-trip
# (prevents hammering the API in quick succession)
INTER_TURN_DELAY_SEC = 3

# Set to True to require Spacebar held down to record
PUSH_TO_TALK_MODE = False           # flip to True to enable

# ─────────────────────────────────────────────
# 4. Arabic Terminal Fix
# ─────────────────────────────────────────────
def print_arabic(text):
    """Reshape + reorder Arabic so Windows terminal renders it correctly."""
    try:
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        print(bidi_text)
    except Exception:
        print(text)

# ─────────────────────────────────────────────
# 5. Setup API + Whisper
# ─────────────────────────────────────────────
client = genai.Client(api_key="AIzaSyC75E2J_OtbxaoqwKUHkygtYOiDhs2HzWc")

print_arabic("--- جاري تحميل محرك الذكاء الاصطناعي للصوت (Whisper)... قد يستغرق هذا بعض الوقت ---")
whisper_model = whisper.load_model("medium")

# ─────────────────────────────────────────────
# 6. Voice-Activity-Detection helpers
# ─────────────────────────────────────────────

def _calibrate_recognizer(r: sr.Recognizer, source: sr.Microphone, duration: float = 1.0):
    """Adjust for ambient noise, then force our minimum energy threshold."""
    r.adjust_for_ambient_noise(source, duration=duration)
    # Never go *below* VAD_ENERGY_THRESHOLD – stops silence from triggering
    if r.energy_threshold < VAD_ENERGY_THRESHOLD:
        r.energy_threshold = VAD_ENERGY_THRESHOLD
    print(f"[VAD] Energy threshold set to: {r.energy_threshold:.0f}")


def vad_listen(r: sr.Recognizer, source: sr.Microphone) -> sr.AudioData | None:
    """
    Listen with VAD.  Returns an AudioData object only when the recognizer
    detects audio that crosses the energy threshold, or None on timeout.
    """
    try:
        print_arabic("🎙️  ... أنا كنسمع (VAD) ...")
        audio = r.listen(
            source,
            timeout=10,                          # wait up to 10 s for speech to start
            phrase_time_limit=PHRASE_TIME_LIMIT_SEC,
        )
        return audio
    except sr.WaitTimeoutError:
        return None   # silence / no speech detected → skip API call


def ptt_listen(r: sr.Recognizer, source: sr.Microphone) -> sr.AudioData | None:
    """
    Push-to-Talk mode: record only while SPACEBAR is held down.
    Returns AudioData or None if the user didn't press anything.
    """
    if not KEYBOARD_AVAILABLE:
        print("[PTT] 'keyboard' library not available – falling back to VAD mode.")
        return vad_listen(r, source)

    print_arabic("🔴 اضغط مع الاستمرار على [SPACE] للكلام، ثم ارفع إصبعك للإرسال ...")
    # Wait until the user presses space
    keyboard.wait("space", suppress=False)

    frames = []
    print_arabic("⏺️  ... جاري التسجيل ...")

    # Record in small chunks while space is held
    CHUNK = 1024
    import array, audioop                          # noqa: F401 (stdlib)
    while keyboard.is_pressed("space"):
        chunk_audio = source.stream.read(CHUNK)
        frames.append(chunk_audio)
        time.sleep(0.01)

    if not frames:
        return None

    # Assemble raw bytes into an AudioData object
    raw = b"".join(frames)
    try:
        rms = audioop.rms(raw, source.SAMPLE_WIDTH)
        if rms < VAD_ENERGY_THRESHOLD:
            print_arabic("⚠️  الصوت خافت جدا – تم تجاهله.")
            return None
    except Exception:
        pass

    return sr.AudioData(raw, source.SAMPLE_RATE, source.SAMPLE_WIDTH)

# ─────────────────────────────────────────────
# 7. Main Consultation Loop
# ─────────────────────────────────────────────
def voice_consultation():
    print("\n" + "=" * 50)
    print_arabic(" 🩺 مرحبا بك في Silent-Doc (مع حماية من تجاوز الحصص) ")
    print("=" * 50)

    if PUSH_TO_TALK_MODE:
        print_arabic("🔵 وضع: Push-to-Talk (SPACE للكلام)")
    else:
        print_arabic("🔵 وضع: كشف الصوت التلقائي (VAD)")

    print_arabic(f"ℹ️  حد الطاقة الصوتية: {VAD_ENERGY_THRESHOLD}  |  تأخير بين الطلبات: {INTER_TURN_DELAY_SEC}s")

    r = sr.Recognizer()
    r.dynamic_energy_threshold = True    # let sr auto-adjust upward over time
    r.energy_threshold = VAD_ENERGY_THRESHOLD

    stop_words = ["سالينا", "صافي", "حبس", "قف", "stop", "exit", "quit", "safy", "salina"]

    with sr.Microphone() as source:
        # One-time calibration at startup
        print_arabic("⏳ جاري معايرة الميكروفون...")
        _calibrate_recognizer(r, source)

        while True:
            # ── Step A: Capture audio (VAD or PTT) ──────────────────────
            print_arabic("\n[Silent-Doc]: راني كنسمعك، شنو هو المشكل اللي عندك؟ (باش تحبس قول 'صافي')")
            try:
                if PUSH_TO_TALK_MODE:
                    audio = ptt_listen(r, source)
                else:
                    audio = vad_listen(r, source)
            except KeyboardInterrupt:
                print_arabic("👋 بسلامة! الله يجيب الشفاء.")
                break

            # ── Step B: Guard – skip if nothing detected ─────────────────
            if audio is None:
                print_arabic("🔇 ما سمعت أي كلام، غانعاود نجرب...")
                # Small idle delay so we don't spin the CPU
                time.sleep(1)
                continue

            # ── Step C: Save captured audio ──────────────────────────────
            with open("input.wav", "wb") as f:
                f.write(audio.get_wav_data())

            # ── Step D: Transcribe with Whisper ──────────────────────────
            print_arabic("--- جاري تحليل الكلام... ---")
            try:
                darija_prompt = (
                    "هذه محادثة بالدارجة المغربية: واش، مزيان، بزاف، "
                    "دابا، واخا، عافاك، ديالي، شنو، كيفاش."
                )
                result = whisper_model.transcribe(
                    "input.wav", language="ar", initial_prompt=darija_prompt
                )
                user_speech = result["text"].strip()

                if not user_speech:
                    print_arabic("❌ ماسمعت والو، عاود هضر عافاك.")
                    time.sleep(INTER_TURN_DELAY_SEC)
                    continue

                print_arabic(f"\n🗣️ أنت قلتي: {user_speech}")

                # Check stop words
                if any(word in user_speech.lower() for word in stop_words):
                    print_arabic("👋 بسلامة! نتمنى لك الشفاء العاجل.")
                    break

                # ── Step E: Call Gemini API ───────────────────────────────
                print_arabic("--- جاري استشارة الطبيب الرقمي... ---")
                sys_prompt = (
                    "أنت طبيب مغربي رقمي. تحاور مع المريض، اسأله إذا احتجت "
                    "لمعلومات، وأجب بالدارجة المغربية فقط وبشكل مختصر جدا ومهني. "
                    "المريض يقول: "
                )
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=sys_prompt + user_speech,
                )
                doctor_reply = response.text

                # ── Step F: TTS + playback ────────────────────────────────
                tts = gTTS(text=doctor_reply, lang="ar")
                tts.save("output.mp3")
                print_arabic(f"\n👨‍⚕️ [Silent-Doc]: {doctor_reply}\n")

                pygame.mixer.music.load("output.mp3")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                pygame.mixer.music.unload()

            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    print_arabic("🚫 تجاوزت الحد المسموح به للاستخدام (API Quota).")
                    print_arabic("⏳ انتظر 60 ثانية قبل المحاولة مرة أخرى...")
                    time.sleep(60)
                else:
                    print_arabic(f"❌ وقع خطأ تقني: {e}")

            # ── Step G: Intentional pacing – rate-limit guard ─────────────
            # Wait before going back to the mic so we can't hammer the API.
            print(f"[Pacing] Waiting {INTER_TURN_DELAY_SEC}s before next turn...")
            time.sleep(INTER_TURN_DELAY_SEC)


if __name__ == "__main__":
    try:
        voice_consultation()
    except KeyboardInterrupt:
        print("\nGoodbye!")