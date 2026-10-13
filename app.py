"""
app.py — Flask Backend for Voice-Enabled Chatbot
=================================================
Endpoints:
  GET  /           → Serve the main HTML page
  POST /chat       → Accept JSON {message} → return {intent, confidence, response}
  POST /transcribe → Accept audio blob (any browser format) → return {transcript}
                     Audio is converted to 16 kHz mono WAV via pydub before STT.
"""

import io
import os
import tempfile
import logging

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import speech_recognition as sr
from pydub import AudioSegment

from chatbot_engine import get_response

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Accept text message and return chatbot response."""
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    logger.info(f"[CHAT] User: {user_message}")
    result = get_response(user_message)
    logger.info(f"[CHAT] Bot [{result['intent']}]: {result['response']}")

    return jsonify(result)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Accept any audio blob from the browser (WebM, OGG, MP4, WAV…),
    convert to 16 kHz mono WAV via pydub, then transcribe with
    Google Web Speech API through the SpeechRecognition library.

    Root cause of 'PCM WAV' error:
      MediaRecorder records in audio/webm;codecs=opus by default.
      SpeechRecognition's AudioFile only reads PCM WAV/AIFF/FLAC.
      Fix: pydub converts any format → PCM WAV in memory.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    content_type = audio_file.content_type or "audio/webm"

    logger.info(f"[STT] Received audio: {len(audio_bytes)} bytes, type={content_type}")

    try:
        # ── Step 1: Detect format from content-type ──────────────────
        fmt = "webm"
        if "ogg"  in content_type: fmt = "ogg"
        elif "mp4" in content_type or "m4a" in content_type: fmt = "mp4"
        elif "wav" in content_type: fmt = "wav"
        elif "webm" in content_type: fmt = "webm"
        elif "flac" in content_type: fmt = "flac"

        # ── Step 2: Convert to 16 kHz mono WAV using pydub ───────────
        raw_io = io.BytesIO(audio_bytes)
        try:
            segment = AudioSegment.from_file(raw_io, format=fmt)
        except Exception:
            # Retry with format auto-detection if explicit fmt fails
            raw_io.seek(0)
            segment = AudioSegment.from_file(raw_io)

        # Normalise: 16 kHz, mono, 16-bit — optimal for Google STT
        segment = segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)

        wav_io = io.BytesIO()
        segment.export(wav_io, format="wav")
        wav_io.seek(0)

        logger.info(f"[STT] Converted to WAV: {len(wav_io.getvalue())} bytes, "
                    f"{segment.duration_seconds:.1f}s")

        # ── Step 3: Transcribe with SpeechRecognition ─────────────────
        with sr.AudioFile(wav_io) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = recognizer.record(source)

        transcript = recognizer.recognize_google(audio_data)
        logger.info(f"[STT] Transcript: {transcript}")
        return jsonify({"transcript": transcript})

    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio. Please speak clearly."}), 422
    except sr.RequestError as e:
        logger.error(f"[STT] Google API error: {e}")
        return jsonify({"error": f"Speech service error: {e}"}), 503
    except Exception as e:
        logger.error(f"[STT] Unexpected error: {e}", exc_info=True)
        return jsonify({"error": f"Audio processing error: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "VoiceBot API"}), 200


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
