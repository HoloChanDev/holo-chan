from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from typing import Optional

import numpy as np
from discord.ext import voice_recv

from holo_chan.io.interfaces import InputSource
from holo_chan.integrations.discord.session import DiscordSession
from holo_chan.stt import STTConfig


class _DiscordSTTBridge:
    def __init__(self, config: STTConfig) -> None:
        self._config = config
        self._socket: Optional[socket.socket] = None
        self._running = False
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)
        self._text_queue: asyncio.Queue[str] = asyncio.Queue()
        self._send_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        print(f"🔌 Connecting to STT server at {self._config.host}:{self._config.port}...")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._socket.connect((self._config.host, self._config.port))
        except Exception as exc:
            print(
                f"❌ Failed to connect to STT server at {self._config.host}:{self._config.port}: {exc}"
            )
            raise
        self._socket.settimeout(0.5)
        print(f"✅ Connected to STT server at {self._config.host}:{self._config.port}")
        self._running = True
        self._send_task = asyncio.create_task(self._send_audio())
        self._recv_task = asyncio.create_task(self._receive_transcriptions())

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._socket:
            self._socket.close()
            self._socket = None

    def submit_audio(self, pcm: bytes) -> None:
        if not self._running or self._loop is None:
            return

        def _put() -> None:
            if self._audio_queue.full():
                return
            self._audio_queue.put_nowait(pcm)

        self._loop.call_soon_threadsafe(_put)

    async def _send_audio(self) -> None:
        while self._running:
            try:
                if self._socket is None:
                    await asyncio.sleep(0.01)
                    continue
                chunk = await self._audio_queue.get()
                await asyncio.to_thread(self._socket.sendall, chunk)
            except (BlockingIOError, TimeoutError):
                await asyncio.sleep(0.01)
            except Exception as exc:
                if self._running:
                    print(f"Error sending audio: {exc}")
                break

    async def _receive_transcriptions(self) -> None:
        while self._running:
            try:
                if self._socket is None:
                    await asyncio.sleep(0.01)
                    continue
                data = await asyncio.to_thread(self._socket.recv, 4096)
                if not data:
                    print("🛑 STT server closed the connection.")
                    self._running = False
                    break
                text = " ".join(data.decode().split(" ")[2:]).strip()
                if text:
                    await self._text_queue.put(text)
            except (BlockingIOError, TimeoutError):
                await asyncio.sleep(0.01)
            except Exception as exc:
                if self._running:
                    print(f"Error receiving transcription: {exc}")
                break

    async def transcriptions(self) -> AsyncIterator[str]:
        while self._running or not self._text_queue.empty():
            try:
                text = await asyncio.wait_for(self._text_queue.get(), timeout=0.1)
                if text:
                    yield text
            except asyncio.TimeoutError:
                continue


def _pcm_to_16k_mono(pcm: bytes, sample_rate: int = 48000) -> bytes:
    if not pcm:
        return b""
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return b""
    if samples.size % 2 != 0:
        samples = samples[:-1]
    stereo = samples.reshape(-1, 2)
    mono = stereo.mean(axis=1).astype(np.int16)
    if sample_rate == 16000:
        return mono.tobytes()
    if sample_rate == 48000:
        return mono[::3].tobytes()
    ratio = 16000 / sample_rate
    new_len = max(1, int(len(mono) * ratio))
    x_old = np.linspace(0, len(mono) - 1, num=len(mono))
    x_new = np.linspace(0, len(mono) - 1, num=new_len)
    resampled = np.interp(x_new, x_old, mono).astype(np.int16)
    return resampled.tobytes()


class _DiscordSTTSink(voice_recv.AudioSink):
    def __init__(self, bridge: _DiscordSTTBridge) -> None:
        super().__init__()
        self._bridge = bridge
        self._logged_first_audio = False
        self._logged_opus_only = False

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data) -> None:
        if user is not None and getattr(user, "bot", False):
            return
        pcm = getattr(data, "pcm", None)
        if not pcm:
            if not self._logged_opus_only and getattr(data, "opus", None) is not None:
                print("⚠️  Received Opus frames but no PCM. Check wants_opus handling.")
                self._logged_opus_only = True
            return
        if not self._logged_first_audio:
            name = getattr(user, "display_name", None) or getattr(user, "name", "unknown")
            print(f"🎧 Receiving voice from {name}")
            self._logged_first_audio = True
        converted = _pcm_to_16k_mono(pcm)
        if converted:
            self._bridge.submit_audio(converted)

    def cleanup(self) -> None:
        return None


class DiscordInputSource(InputSource):
    def __init__(self, session: DiscordSession, stt_config: STTConfig) -> None:
        self._session = session
        self._stt = _DiscordSTTBridge(stt_config)
        self._sink: Optional[_DiscordSTTSink] = None
        self._voice_client: Optional[voice_recv.VoiceRecvClient] = None

    async def __aenter__(self) -> "DiscordInputSource":
        await self._stt.start()
        self._voice_client = await self._session.get_voice_client()
        self._sink = _DiscordSTTSink(self._stt)
        self._voice_client.listen(self._sink)
        print(
            "🎧 Listening to Discord voice channel audio "
            f"(listening={self._voice_client.is_listening()})"
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._voice_client and self._voice_client.is_listening():
            self._voice_client.stop_listening()
        await self._stt.stop()

    def __aiter__(self) -> AsyncIterator[str]:
        if self._sink is None:
            raise RuntimeError("DiscordInputSource must be used as an async context manager.")
        return self._stt.transcriptions()
