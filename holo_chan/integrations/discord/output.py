from __future__ import annotations

import asyncio
import io
import os
from typing import Optional

import numpy as np
import zmq
import zmq.asyncio

import discord

from holo_chan.io.interfaces import OutputSink
from holo_chan.integrations.discord.session import DiscordSession


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def _resample_mono(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples
    ratio = dst_rate / src_rate
    new_len = max(1, int(len(samples) * ratio))
    x_old = np.linspace(0, len(samples) - 1, num=len(samples))
    x_new = np.linspace(0, len(samples) - 1, num=new_len)
    return np.interp(x_new, x_old, samples).astype(np.int16)


def _to_48k_stereo(pcm: bytes, sample_rate: int) -> bytes:
    if not pcm:
        return b""
    mono = np.frombuffer(pcm, dtype=np.int16)
    if mono.size == 0:
        return b""
    mono = _resample_mono(mono, sample_rate, 48000)
    stereo = np.repeat(mono[:, None], 2, axis=1).reshape(-1)
    return stereo.astype(np.int16).tobytes()


class _DiscordTTSClient:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.REQ)
        print(f"🔌 Connecting to TTS server at {host}:{port}...")
        try:
            self._socket.connect(f"tcp://{host}:{port}")
        except Exception as exc:
            print(f"❌ Failed to connect to TTS server at {host}:{port}: {exc}")
            raise
        print(f"✅ Connected to TTS server at {host}:{port}")

    async def synthesize(self, text: str) -> tuple[int, bytes]:
        self._socket.send_unicode(text)
        sampling_rate_bytes, audio_bytes = await self._socket.recv_multipart()
        sample_rate = int.from_bytes(sampling_rate_bytes)
        return sample_rate, audio_bytes

    async def close(self) -> None:
        self._socket.close(0)
        self._context.term()


class DiscordSpeaker(OutputSink):
    def __init__(
        self,
        session: DiscordSession,
        tts_host: Optional[str] = None,
        tts_port: Optional[int] = None,
    ) -> None:
        self._session = session
        host = tts_host or os.getenv("TTS_HOST", "localhost")
        port = tts_port or _env_int("TTS_PORT", 5511)
        self._tts = _DiscordTTSClient(host, port)

    async def __aenter__(self) -> "DiscordSpeaker":
        return self

    async def speak(self, text: str) -> None:
        voice_client = await self._session.get_voice_client()
        sample_rate, audio_bytes = await self._tts.synthesize(text)
        pcm = _to_48k_stereo(audio_bytes, sample_rate)
        if not pcm:
            return
        while voice_client.is_playing():
            await asyncio.sleep(0.05)
        source = discord.PCMAudio(io.BytesIO(pcm))
        voice_client.play(source)

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._tts.close()
