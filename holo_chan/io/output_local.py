from __future__ import annotations

from holo_chan.io.interfaces import OutputSink
from holo_chan.tts import speak


class LocalSpeaker(OutputSink):
    async def speak(self, text: str) -> None:
        await speak(text)

    async def __aenter__(self) -> "LocalSpeaker":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None
