from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Optional

from holo_chan.io.interfaces import InputSource
from holo_chan.stt import STTConfig, STTListener


class LocalInputSource(InputSource):
    def __init__(self, config: Optional[STTConfig] = None) -> None:
        self._config = config or STTConfig()
        self._listener: Optional[STTListener] = None

    async def __aenter__(self) -> "LocalInputSource":
        self._listener = STTListener(self._config)
        await self._listener.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._listener is not None:
            await self._listener.stop()
            self._listener = None

    def __aiter__(self) -> AsyncIterator[str]:
        if self._listener is None:
            raise RuntimeError("LocalInputSource must be used as an async context manager.")
        return self._listener.transcriptions()
