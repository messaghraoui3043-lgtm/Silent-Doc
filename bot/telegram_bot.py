"""
bot.py – Telegram bot that wraps the Silent Doctor FastAPI backend.

Setup:
  1. Create a bot via @BotFather on Telegram and copy the token.
  2. Set the TELEGRAM_TOKEN environment variable:
       $env:TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"          # PowerShell
       set TELEGRAM_TOKEN=YOUR_BOT_TOKEN                # CMD
  3. Make sure api.py is running:
       uvicorn api:app --reload --port 8000
  4. Run the bot:
       python bot.py

Requirements:
    pip install python-telegram-bot requests
"""

import os
import io
import logging
import requests

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ── Config ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s – %(name)s – %(levelname)s – %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN", "PASTE_YOUR_TOKEN_HERE")
API_URL   = os.environ.get("API_URL", "http://localhost:5000/predict/skin")

DISCLAIMER = (
    "⚠️ *MEDICAL DISCLAIMER:* This AI is a diagnostic aid only and is "
    "*NOT* a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified dermatologist or healthcare provider."
)

# ── Command handlers ──────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🩺 *Welcome to Silent Doctor!*\n\n"
        "I use a trained AI model to analyse skin lesion images.\n\n"
        "📸 Simply *send me a photo* of a skin lesion and I will return "
        "the top-3 most likely diagnoses with confidence scores.\n\n"
        "Type /help for more information.",
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🔍 *How to use Silent Doctor:*\n\n"
        "1️⃣  Send any skin lesion photo to this chat.\n"
        "2️⃣  Wait a few seconds while the AI analyses it.\n"
        "3️⃣  Receive the top-3 most probable diagnoses.\n\n"
        "*Supported conditions:*\n"
        "• Actinic keratoses (akiec)\n"
        "• Basal cell carcinoma (bcc)\n"
        "• Benign keratosis-like lesions (bkl)\n"
        "• Dermatofibroma (df)\n"
        "• Melanocytic nevi (nv)\n"
        "• Melanoma (mel)\n"
        "• Vascular lesions (vasc)\n\n"
        + DISCLAIMER,
        parse_mode="Markdown",
    )


# ── Photo handler ─────────────────────────────────────────────────────────────
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download the sent photo and forward it to the FastAPI /predict endpoint."""

    await update.message.reply_text("🔬 Analysing your image, please wait…")

    # Get the highest-res version of the image
    photo = update.message.photo[-1]
    photo_file = await photo.get_file()

    # Download photo bytes
    photo_bytes = await photo_file.download_as_bytearray()

    # Send to FastAPI
    try:
        response = requests.post(
            API_URL,
            files={"file": ("image.jpg", io.BytesIO(photo_bytes), "image/jpeg")},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        await update.message.reply_text(
            "❌ Could not reach the prediction server.\n"
            "Please make sure `api.py` is running at: " + API_URL
        )
        return
    except Exception as e:
        await update.message.reply_text(f"❌ Prediction failed: {str(e)}")
        return

    # Format the results
    predictions = data.get("predictions", [])
    if not predictions:
        await update.message.reply_text("⚠️ No predictions returned. Try another image.")
        return

    rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}
    lines = ["🩺 *Diagnostic Results:*\n"]
    for p in predictions:
        bar_len = int(p['confidence'] / 10)           # 0–10 blocks
        bar     = "█" * bar_len + "░" * (10 - bar_len)
        lines.append(
            f"{rank_emoji.get(p['rank'], '•')} *{p['label']}*\n"
            f"   `{bar}` {p['confidence']}%\n"
        )
    lines.append("\n" + DISCLAIMER)
    reply_text = "\n".join(lines)

    await update.message.reply_text(reply_text, parse_mode="Markdown")


# ── Non-photo message handler ─────────────────────────────────────────────────
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📸 Please send me a *photo* of a skin lesion to analyse.\n"
        "Type /help for instructions.",
        parse_mode="Markdown",
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if BOT_TOKEN == "PASTE_YOUR_TOKEN_HERE":
        print("ERROR: Set the TELEGRAM_TOKEN environment variable first.")
        print("  PowerShell: $env:TELEGRAM_TOKEN = 'YOUR_TOKEN'")
        print("  CMD:         set TELEGRAM_TOKEN=YOUR_TOKEN")
        return

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Silent Doctor Telegram Bot is running…")
    app.run_polling()


if __name__ == "__main__":
    main()
