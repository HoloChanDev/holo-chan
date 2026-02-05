import asyncio
import os
import re
import sys
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from holo_chan.agent import FULL_SYSTEM_PROMPT, build_message, run_agent
from holo_chan.io.input_local import LocalInputSource
from holo_chan.io.interfaces import InputSource, OutputSink
from holo_chan.stt import STTConfig

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    sys.exit(f"❌ {name} must be a boolean-like value, got {raw!r}.")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        sys.exit(f"❌ {name} must be an integer, got {value!r}.")


def _build_local_input_source() -> InputSource:
    stt_config = STTConfig(
        host=os.getenv("STT_HOST", "localhost"),
        port=_env_int("STT_PORT", 43007),
        chunk_ms=1000,
    )
    print(f"STT -> {stt_config.host}:{stt_config.port}")
    return LocalInputSource(stt_config)


def _build_local_output_sink() -> OutputSink:
    from holo_chan.io.output_local import LocalSpeaker

    return LocalSpeaker()


if TYPE_CHECKING:
    from holo_chan.io.output_local import LocalSpeaker
    from holo_chan.integrations.discord import wiring as discord_wiring
    from holo_chan.integrations.discord.session import DiscordSession


def _build_discord_session():
    from holo_chan.integrations.discord import wiring as discord_wiring

    return discord_wiring.build_session()


def _build_discord_input_source(session: "DiscordSession") -> InputSource:
    from holo_chan.integrations.discord import wiring as discord_wiring

    return discord_wiring.build_input_source(session)


def _build_discord_output_sink(session: "DiscordSession") -> OutputSink:
    from holo_chan.integrations.discord import wiring as discord_wiring

    return discord_wiring.build_output_sink(session)


async def _maybe_enter(stack: AsyncExitStack, obj):
    aenter = getattr(obj, "__aenter__", None)
    if aenter is None:
        return obj
    return await stack.enter_async_context(obj)

_SENTENCE_BOUNDARY_RE = re.compile(r"([.!?,;:])")


def _split_complete_sentences(text: str) -> tuple[list[str], str]:
    parts = _SENTENCE_BOUNDARY_RE.split(text)
    if len(parts) == 1:
        return [], text.strip()

    sentences: list[str] = []
    for i in range(0, len(parts) - 1, 2):
        segment = parts[i].strip()
        delimiter = parts[i + 1]
        if segment:
            sentences.append(f"{segment}{delimiter}")
        elif delimiter.strip():
            sentences.append(delimiter)
    remainder = parts[-1].strip() if len(parts) % 2 == 1 else ""
    return sentences, remainder


async def _process_transcriptions(
    incoming: asyncio.Queue[str | None],
    messages: list[dict[str, str]],
    output_sink: OutputSink,
) -> list[dict[str, str]]:
    buffer = ""
    pending: list[str] = []
    in_flight: asyncio.Task[list[dict[str, str]]] | None = None
    finished = False
    no_item = object()

    while True:
        try:
            text = await asyncio.wait_for(incoming.get(), timeout=0.05)
        except asyncio.TimeoutError:
            text = no_item

        if text is None:
            finished = True
        elif text is not no_item:
            buffer = f"{buffer} {text}".strip()
            sentences, remainder = _split_complete_sentences(buffer)
            if sentences:
                pending.extend(sentences)
            buffer = remainder

        if in_flight is None and pending:
            batch = " ".join(pending).strip()
            pending.clear()
            print(f"\nUser: {batch}")
            in_flight = asyncio.create_task(
                run_agent(
                    batch,
                    messages,
                    output_sink=output_sink,
                )
            )

        if in_flight is not None and in_flight.done():
            messages = in_flight.result()
            in_flight = None

        if finished and in_flight is None and not pending and buffer:
            batch = buffer
            buffer = ""
            print(f"\nUser: {batch}")
            in_flight = asyncio.create_task(
                run_agent(
                    batch,
                    messages,
                    output_sink=output_sink,
                )
            )

        if finished and in_flight is None and not pending and not buffer:
            return messages


async def main() -> None:
    """Main loop that listens for audio and processes transcriptions."""
    use_discord = _env_bool("HOLO_DISCORD", False)

    # Initialize conversation state
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("❌ Environment variable GROQ_API_KEY not set.")

    messages: list[dict[str, str]] = [
        build_message("system", FULL_SYSTEM_PROMPT),
    ]

    session = None
    if use_discord:
        session = _build_discord_session()

    if use_discord:
        input_source = _build_discord_input_source(session)
    else:
        input_source = _build_local_input_source()

    if use_discord:
        output_sink = _build_discord_output_sink(session)
    else:
        output_sink = _build_local_output_sink()

    async with AsyncExitStack() as stack:
        if session is not None:
            session = await _maybe_enter(stack, session)
        input_source = await _maybe_enter(stack, input_source)
        output_sink = await _maybe_enter(stack, output_sink)

        incoming: asyncio.Queue[str | None] = asyncio.Queue()
        processor = asyncio.create_task(
            _process_transcriptions(incoming, messages, output_sink)
        )

        async for transcription in input_source:
            if not transcription or not transcription.strip():
                continue

            await incoming.put(transcription)

        await incoming.put(None)
        messages = await processor


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down.")
