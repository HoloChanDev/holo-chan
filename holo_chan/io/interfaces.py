from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol


class InputSource(Protocol):
    def __aiter__(self) -> AsyncIterator[str]: ...


class OutputSink(Protocol):
    async def speak(self, text: str) -> None: ...
