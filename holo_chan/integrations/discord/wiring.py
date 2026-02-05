from __future__ import annotations

import os

from holo_chan.integrations.discord.input import DiscordInputSource
from holo_chan.integrations.discord.output import DiscordSpeaker
from holo_chan.integrations.discord.session import DiscordConfig, DiscordSession
from holo_chan.io.interfaces import InputSource, OutputSink
from holo_chan.stt import STTConfig


def build_session() -> DiscordSession:
    config = DiscordConfig.from_env()
    return DiscordSession(config)


def build_input_source(session: DiscordSession) -> InputSource:
    stt_config = STTConfig(
        host=os.getenv("STT_HOST", "localhost"),
        port=_env_int("STT_PORT", 43007),
        chunk_ms=1000,
    )
    return DiscordInputSource(session, stt_config)


def build_output_sink(session: DiscordSession) -> OutputSink:
    return DiscordSpeaker(session)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
