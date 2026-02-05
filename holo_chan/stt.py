"""
Async Speech-to-Text (STT) module for continuous audio listening.
Streams microphone audio to a Whisper server and yields transcriptions.
"""

import asyncio
import queue
import socket
import sys
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import numpy as np
import sounddevice as sd

# Audio configuration matching whisper_online_server.py
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
DTYPE = np.int16


@dataclass
class STTConfig:
    """Configuration for STT listener."""

    host: str = "localhost"
    port: int = 43007
    chunk_ms: int = 1000


class STTListener:
    """Async listener that streams audio to Whisper server and yields transcriptions."""

    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or STTConfig()
        self._socket: Optional[socket.socket] = None
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._audio_queue: Optional[queue.Queue[bytes]] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def start(self) -> None:
        """Start the audio streaming and transcription."""
        if self._running:
            return

        # Connect to the Whisper server
        print(f"🔌 Connecting to STT server at {self.config.host}:{self.config.port}...")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._socket.connect((self.config.host, self.config.port))
        except Exception as exc:
            print(
                f"❌ Failed to connect to STT server at {self.config.host}:{self.config.port}: {exc}",
                file=sys.stderr,
            )
            raise
        print(f"✅ Connected to STT server at {self.config.host}:{self.config.port}")
        # Use a timeout to prevent blocking indefinitely on recv.
        self._socket.settimeout(0.5)

        # Set up audio stream with callback
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._audio_callback,
            blocksize=CHUNK,
        )
        self._stream.start()

        # Queue audio chunks from the callback to a sender task.
        self._audio_queue = queue.Queue(maxsize=20)
        self._send_task = asyncio.create_task(self._send_audio())

        # Start receiving transcriptions in background
        self._receive_task = asyncio.create_task(self._receive_transcriptions())
        self._running = True

    async def stop(self) -> None:
        """Stop the audio streaming and cleanup."""
        if not self._running:
            return

        self._running = False
        print("🛑 Stopping STT listener")

        # Cancel receive task
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass

        # Stop audio stream
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Close socket
        if self._socket:
            self._socket.close()
            self._socket = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        """Callback for audio data - sends to socket."""
        if status:
            print(status, file=sys.stderr)
        if self._audio_queue:
            try:
                self._audio_queue.put_nowait(indata.tobytes())
            except queue.Full:
                # Drop audio if the sender is temporarily backed up.
                pass

    async def _send_audio(self) -> None:
        """Background task to send queued audio to server."""
        while self._running:
            try:
                if self._socket is None or self._audio_queue is None:
                    await asyncio.sleep(0.01)
                    continue
                chunk = await asyncio.to_thread(self._audio_queue.get)
                await asyncio.to_thread(self._socket.sendall, chunk)
            except (BlockingIOError, TimeoutError):
                await asyncio.sleep(0.01)
            except Exception as e:
                if self._running:
                    print(f"Error sending audio: {e}", file=sys.stderr)
                break

    async def _receive_transcriptions(self) -> None:
        """Background task to receive transcriptions from server."""
        while self._running:
            try:
                if self._socket is None:
                    await asyncio.sleep(0.01)
                    continue
                data = await asyncio.to_thread(self._socket.recv, 4096)
                if not data:
                    # Server closed the connection.
                    print("🛑 STT server closed the connection.")
                    self._running = False
                    break
                text = " ".join(data.decode().split(" ")[2:]).strip()
                if text:
                    await self._queue.put(text)
            except (BlockingIOError, TimeoutError):
                await asyncio.sleep(0.01)
            except Exception as e:
                if self._running:
                    print(f"Error receiving transcription: {e}", file=sys.stderr)
                break

    async def transcriptions(self) -> AsyncGenerator[str, None]:
        """Yield transcriptions as they arrive."""
        while self._running or not self._queue.empty():
            try:
                text = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                if text:
                    yield text
            except asyncio.TimeoutError:
                continue


async def listen(config: Optional[STTConfig] = None) -> AsyncGenerator[str, None]:
    """
    Convenience function to listen for transcriptions.

    Usage:
        async for transcription in listen():
            print(f"Transcription: {transcription}")
    """
    async with STTListener(config) as listener:
        async for transcription in listener.transcriptions():
            yield transcription
