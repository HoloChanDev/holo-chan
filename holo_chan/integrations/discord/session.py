from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

try:
    import discord
    import discord.opus
    from discord.ext import voice_recv
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "Discord voice dependencies are not installed. Install discord.py[voice] and discord-ext-voice-recv."
    ) from exc


def _env_int(name: str) -> int:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is required.")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


@dataclass(frozen=True)
class DiscordConfig:
    token: str
    voice_channel_id: int

    @classmethod
    def from_env(cls) -> "DiscordConfig":
        token = os.getenv("DISCORD_TOKEN")
        if not token:
            raise ValueError("DISCORD_TOKEN is required to enable Discord I/O.")
        voice_channel_id = os.getenv("DISCORD_VOICE_CHANNEL_ID")
        if not voice_channel_id:
            voice_channel_id = os.getenv("DISCORD_CHANNEL_ID")
        if not voice_channel_id:
            raise ValueError(
                "DISCORD_VOICE_CHANNEL_ID is required to enable Discord voice I/O."
            )
        try:
            channel_id = int(voice_channel_id)
        except ValueError as exc:
            raise ValueError(
                f"DISCORD_VOICE_CHANNEL_ID must be an integer, got {voice_channel_id!r}."
            ) from exc
        return cls(token=token, voice_channel_id=channel_id)


class DiscordSession:
    def __init__(self, config: DiscordConfig) -> None:
        self.config = config
        self._ready = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._client = self._build_client()
        self._voice_client: Optional[voice_recv.VoiceRecvClient] = None

    def _build_client(self) -> discord.Client:
        intents = discord.Intents.default()
        intents.guilds = True
        intents.voice_states = True

        session = self

        class _Client(discord.Client):
            async def on_ready(self) -> None:
                session._ready.set()

        return _Client(intents=intents)

    async def __aenter__(self) -> "DiscordSession":
        self._task = asyncio.create_task(self._client.start(self.config.token))
        done, _ = await asyncio.wait(
            {self._task, asyncio.create_task(self._ready.wait())},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if self._task in done:
            exc = self._task.exception()
            if exc:
                raise exc
        await self._ready.wait()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._voice_client and self._voice_client.is_connected():
            await self._voice_client.disconnect(force=True)
        if self._client.is_closed():
            return
        await self._client.close()
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def get_voice_client(self) -> voice_recv.VoiceRecvClient:
        await self._ready.wait()
        if self._voice_client and self._voice_client.is_connected():
            return self._voice_client
        channel = self._client.get_channel(self.config.voice_channel_id)
        if channel is None:
            channel = await self._client.fetch_channel(self.config.voice_channel_id)
        if not isinstance(channel, discord.VoiceChannel):
            raise ValueError(
                f"Channel {self.config.voice_channel_id} is not a voice channel."
            )
        print(f"🔌 Connecting to Discord voice channel: {channel.name} ({channel.id})")
        self._voice_client = await channel.connect(
            cls=voice_recv.VoiceRecvClient,
            self_deaf=False,
            self_mute=False,
        )
        print(f"✅ Connected to Discord voice channel: {channel.name} ({channel.id})")
        print(f"🔧 Voice client type: {type(self._voice_client).__name__}")
        if not discord.opus.is_loaded():
            opus_path = os.getenv("DISCORD_OPUS_LIBRARY")
            if opus_path:
                try:
                    discord.opus.load_opus(opus_path)
                except Exception as exc:
                    print(f"⚠️  Failed to load Opus from {opus_path}: {exc}")
        if not discord.opus.is_loaded():
            print("⚠️  Discord Opus library is not loaded; voice receive may not work.")
        else:
            print("✅ Discord Opus library loaded.")
        return self._voice_client
