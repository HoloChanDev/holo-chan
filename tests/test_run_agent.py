import pytest

from holo_chan import main


@pytest.mark.asyncio
async def test_run_agent_plain_response_calls_speak(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    spoken: list[str] = []

    async def fake_completion(_messages: list[dict[str, str]]):
        return "Hello there."

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    messages = await main.run_agent(
        "Hi",
        messages=[main.build_message("system", "sys")],
        completion_func=fake_completion,
        speak_func=fake_speak,
    )

    assert spoken == ["Hello there."]
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "Hello there."


@pytest.mark.asyncio
async def test_run_agent_done_tool_does_not_speak(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    spoken: list[str] = []

    async def fake_completion(_messages: list[dict[str, str]]):
        return '{"tool_calls":{"name":"done","arguments":{}}}'

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    messages = await main.run_agent(
        "Hi",
        messages=[main.build_message("system", "sys")],
        completion_func=fake_completion,
        speak_func=fake_speak,
    )

    assert spoken == []
    assert messages[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_run_agent_completion_error_does_not_exit(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    async def fake_completion(_messages: list[dict[str, str]]):
        raise RuntimeError("boom")

    async def fake_speak(_text: str) -> None:
        return None

    messages = await main.run_agent(
        "Hi",
        messages=[main.build_message("system", "sys")],
        completion_func=fake_completion,
        speak_func=fake_speak,
    )

    assert messages[0]["role"] == "system"
