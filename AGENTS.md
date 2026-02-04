* This project uses uv. DO NOT RUN PYTHON OR PIP DIRECTLY. ALWAYS USE UV.

# Holo-Chan Agent Guide

## Overview
- `main.py` is the orchestrator. It listens to STT, sends transcripts to the LLM, handles tool calls, and forwards spoken responses to TTS.
- `stt.py` streams microphone audio to a remote Whisper server over TCP and yields transcriptions.
- `tts.py` sends text to a remote TTS server over ZeroMQ and plays back audio locally.

## Runtime Flow
1. `main.py` loads environment variables (via `python-dotenv`), configures `STTConfig`, and starts `STTListener`.
2. `STTListener` streams audio frames to the Whisper server and queues transcripts.
3. Each transcription triggers `run_agent()`, which:
   - Calls the LLM (LiteLLM + Groq) to get a response or tool call.
   - Executes local tools when requested.
   - Sends assistant responses to `speak()` for TTS playback.

## Environment Variables
- `GROQ_API_KEY`: required for LiteLLM/Groq requests.
- `STT_HOST`, `STT_PORT`: remote STT server address (defaults: `localhost:43007`).
- `TTS_HOST`, `TTS_PORT`: remote TTS server address (defaults: `localhost:5511`).

## How To Run
- Ensure `.env` exists with the required vars.
- Use uv to run:
  - `uv run python main.py`

## Notes
- `tts.py` guards Windows-only event loop policy calls, so it works on macOS.
- Tool calls are JSON-only and must match the schema in `SYSTEM_PROMPT`.
