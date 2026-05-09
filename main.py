#!/usr/bin/env python3
"""
Silent Doctor — Main Entry Point
==================================
Interactive CLI for the local-first AI medical assistant.

Provides a menu-driven interface to:
    1. 🔬 Analyze skin image
    2. 👁️ Analyze eye image
    3. 🎤 Voice consultation
    4. ❓ Ask a medical question
    5. 👶 Child health support
    6. 📚 Build knowledge base
    0. 🚪 Exit

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Ensure the project root is in the Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.router import Orchestrator
from utils.helpers import setup_logger

logger = setup_logger("main")


# ── Banner ──────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🏥  S I L E N T   D O C T O R  🏥                    ║
║                                                              ║
║        Local-First AI Medical Assistant                       ║
║        For Rural Communities                                  ║
║                                                              ║
║        ⚠️  This is an AI tool, NOT a real doctor.             ║
║        Always seek professional medical care.                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

MENU = """
┌─────────────────────────────────────┐
│           MAIN MENU                 │
├─────────────────────────────────────┤
│  1. 🔬  Analyze Skin Image         │
│  2. 👁️  Analyze Eye Image          │
│  3. 🎤  Voice Consultation          │
│  4. ❓  Ask a Medical Question      │
│  5. 👶  Child Health Support        │
│  6. 📚  Build Knowledge Base        │
│  0. 🚪  Exit                        │
└─────────────────────────────────────┘
"""


def print_separator():
    """Print a visual separator."""
    print("\n" + "─" * 60 + "\n")


def format_result(result: dict) -> str:
    """Format a pipeline result for display."""
    result_type = result.get("type", "unknown")

    if result_type in ("skin_analysis", "eye_analysis"):
        pred = result["prediction"]
        output = [
            f"🔍 Detected Condition: {pred['prediction'].upper()}",
            f"📊 Confidence: {pred['confidence']:.1%}",
            "",
            "📋 All Scores:",
        ]
        for cls, score in pred["all_scores"].items():
            bar = "█" * int(score * 30)
            output.append(f"   {cls:20s} {score:.1%} {bar}")
        output.append("")
        output.append("💡 Medical Advice:")
        output.append(result.get("advice", "No advice available."))
        return "\n".join(output)

    elif result_type == "text_answer":
        output = [
            f"❓ Question: {result['question']}",
            "",
            "💬 Answer:",
            result["answer"],
        ]
        if result.get("sources"):
            output.append("")
            output.append(f"📚 Sources: {', '.join(result['sources'])}")
        return "\n".join(output)

    elif result_type == "voice_consultation":
        output = [
            f"🎤 You said ({result['detected_language']}): {result['original_text']}",
            f"🌐 In English: {result['english_query']}",
            "",
            "💬 Answer (English):",
            result["english_answer"],
            "",
            f"🌐 Translated ({result['target_language']}):",
            result["translated_answer"],
        ]
        return "\n".join(output)

    elif result_type == "child_assessment":
        output = [
            f"👶 Symptoms: {result['description']}",
            f"📌 Categories: {', '.join(result['categories']) or 'general'}",
            f"⚠️  Severity: {result['severity'].upper()}",
            "",
            "💬 Advice:",
            result["advice"],
        ]
        return "\n".join(output)

    else:
        return str(result)


# ── Menu Handlers ───────────────────────────────────────────────────────

def handle_skin_image(orchestrator: Orchestrator):
    """Handle skin image analysis."""
    image_path = input("📷 Enter path to skin image: ").strip()
    if not image_path:
        print("⚠️  No path provided.")
        return

    if not Path(image_path).exists():
        print(f"⚠️  File not found: {image_path}")
        return

    result = orchestrator.route({
        "input_type": "skin_image",
        "data": image_path,
    })
    print_separator()
    print(format_result(result))


def handle_eye_image(orchestrator: Orchestrator):
    """Handle eye image analysis."""
    image_path = input("📷 Enter path to eye image: ").strip()
    if not image_path:
        print("⚠️  No path provided.")
        return

    if not Path(image_path).exists():
        print(f"⚠️  File not found: {image_path}")
        return

    result = orchestrator.route({
        "input_type": "eye_image",
        "data": image_path,
    })
    print_separator()
    print(format_result(result))


def handle_voice(orchestrator: Orchestrator):
    """Handle voice consultation."""
    print("🎤 Voice Consultation")
    print("Options:")
    print("  1. Record from microphone")
    print("  2. Use existing audio file")

    choice = input("Choice (1/2): ").strip()

    language = input(
        "🌐 Your language (darija/amazigh/arabic/english) [darija]: "
    ).strip() or "darija"

    if choice == "2":
        audio_path = input("📁 Enter path to audio file: ").strip()
        if not Path(audio_path).exists():
            print(f"⚠️  File not found: {audio_path}")
            return
        result = orchestrator.route({
            "input_type": "voice",
            "data": audio_path,
            "language": language,
        })
    else:
        duration = input("⏱️  Recording duration in seconds [5]: ").strip()
        duration = float(duration) if duration else 5.0
        result = orchestrator.route({
            "input_type": "voice",
            "data": None,
            "language": language,
            "duration": duration,
        })

    print_separator()
    print(format_result(result))


def handle_question(orchestrator: Orchestrator):
    """Handle text-based medical question."""
    question = input("❓ Enter your medical question: ").strip()
    if not question:
        print("⚠️  No question provided.")
        return

    result = orchestrator.route({
        "input_type": "question",
        "data": question,
    })
    print_separator()
    print(format_result(result))


def handle_child(orchestrator: Orchestrator):
    """Handle child health support."""
    print("👶 Child Health Support")
    description = input("📝 Describe the child's symptoms: ").strip()
    if not description:
        print("⚠️  No description provided.")
        return

    age = input("📅 Child's age (optional, e.g., '2 years'): ").strip() or None

    result = orchestrator.route({
        "input_type": "child",
        "data": description,
        "child_age": age,
    })
    print_separator()
    print(format_result(result))


def handle_build_knowledge_base():
    """Build or rebuild the FAISS knowledge base from datasets/."""
    from rag.vector_store import get_vector_store

    print("📚 Building Knowledge Base")
    print(f"   Source directory: datasets/")

    directory = input(
        "📁 Directory to ingest (press Enter for 'datasets/'): "
    ).strip()

    if not directory:
        from config.settings import DATASETS_DIR
        directory = str(DATASETS_DIR)

    store = get_vector_store()
    count = store.build_from_directory(directory)

    if count > 0:
        store.save()
        print(f"✅ Knowledge base built: {count} chunks indexed.")
    else:
        print("⚠️  No documents found. Add PDFs/text files to datasets/.")


# ── Main Loop ───────────────────────────────────────────────────────────

def main():
    """Main application loop."""
    print(BANNER)

    orchestrator = Orchestrator()

    handlers = {
        "1": handle_skin_image,
        "2": handle_eye_image,
        "3": handle_voice,
        "4": handle_question,
        "5": handle_child,
    }

    while True:
        print(MENU)
        choice = input("👉 Select an option: ").strip()

        if choice == "0":
            print("\n👋 Thank you for using Silent Doctor. Stay healthy!")
            break
        elif choice == "6":
            handle_build_knowledge_base()
        elif choice in handlers:
            try:
                handlers[choice](orchestrator)
            except KeyboardInterrupt:
                print("\n⚠️  Operation cancelled.")
            except Exception as exc:
                logger.error(f"Error: {exc}", exc_info=True)
                print(f"\n❌ Error: {exc}")
                print("   Please check that required models are installed.")
        else:
            print("⚠️  Invalid option. Please try again.")

        print_separator()


if __name__ == "__main__":
    main()
