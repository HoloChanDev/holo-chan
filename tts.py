import asyncio
import os
import sys

from dotenv import load_dotenv
import simpleaudio as sa
import zmq
import zmq.asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ----------------------------------------------------------------------
# 🎛️  CONFIGURATION

load_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


# Set up ZeroMQ client
context = zmq.asyncio.Context()
socket = context.socket(zmq.REQ)
tts_host = os.getenv("TTS_HOST", "localhost")
tts_port = _env_int("TTS_PORT", 5511)
socket.connect(f"tcp://{tts_host}:{tts_port}")

is_speaking = False
_logged_tts_connection = False


async def speak(text: str) -> None:
    global _logged_tts_connection
    if not _logged_tts_connection:
        print(f"🔊 TTS -> {tts_host}:{tts_port}")
        _logged_tts_connection = True
    # Wait for the model to stop speaking
    global is_speaking
    while is_speaking is True:
        await asyncio.sleep(1)
    is_speaking = True

    # Send text to TTS server and receive audio data
    socket.send_unicode(text)

    # Receive sampling rate and audio data together
    sampling_rate_bytes, audio_bytes = await socket.recv_multipart()
    sample_rate = int.from_bytes(sampling_rate_bytes)

    # Play the audio
    play_obj = sa.play_buffer(
        audio_bytes,
        num_channels=1,
        bytes_per_sample=2,  # int16 = 2 bytes
        sample_rate=sample_rate,
    )

    # Set is_speaking to False when we're done speaking
    async def _watch():
        global is_speaking
        # wait in a worker thread
        await asyncio.to_thread(play_obj.wait_done)
        is_speaking = False

    # schedule watcher and return immediately
    asyncio.create_task(_watch())
