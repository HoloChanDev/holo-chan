import json
import os
import pprint as pp
import sys
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable

from litellm import acompletion

from typing import TYPE_CHECKING

from holo_chan.io.interfaces import OutputSink

if TYPE_CHECKING:
    from holo_chan.io.output_local import LocalSpeaker

# ----------------------------------------------------------------------
# 🎛️  CONFIGURATION

DEFAULT_LLM_MODEL = (
    "groq/meta-llama/llama-4-scout-17b-16e-instruct"  # LiteLLM format for Groq models
)
TEMPERATURE = 0.7
MAX_TOKENS = 1024

MAX_TURNS = 10
TOOL_CALL_TAG = "tool_calls"

SYSTEM_PROMPT = """You are Hatsune Miku, an anime girl and a hologram. You have access to some external tools.
When you need to use a tool, respond **only** with a JSON object that follows this schema:

{
    "tool_calls": {
        "name": "<tool_name>",
        "arguments": { ... }               # a JSON-serializable dict of arguments
    }
}

Do NOT add any other text before or after the JSON.  If you do not need a tool, answer the user directly in plain English.

Available tools:
"""

SYSTEM_PROMPT_AFTER_TOOL_CALLS = """
You are a virtual hologram that speaks, you respond in short sentences when possible, \
you don't want to speak for minutes unless you're sure that is what the user wants, and they usually don't.

You are a dilligent assistant, you should usually respond to the user, unless you're completely sure they only want you to do something and not tell you anything about what you're doing.
"""

# ----------------------------------------------------------------------
# 🛠️  TOOL IMPLEMENTATIONS


def tool_done() -> str | None:
    return None


def tool_dance(dance: str | int) -> str | None:
    return f"Performing dance number {dance}"


def tool_backflip() -> str | None:
    return "Performing backflip"


@dataclass
class Tool:
    func: Callable[..., str | None]
    description: str


TOOLS: dict[str, Tool] = {
    "wait_for_more": Tool(
        tool_done,
        "Wait for the user to continue speaking, ALWAYS call this if you think the user isn't done speaking. Takes no arguments.",
    ),
    "done": Tool(
        tool_done,
        "You're done, let the user continue speaking, You can continue making more calls until you call this. YOU ALWYAS NEED TO CALL THIS TO STOP.",
    ),
    "dance": Tool(
        tool_dance,
        "Dance as the hologram. Takes `dance` as the dance you want to perform, can be 1 2 or 3",
    ),
    "backflip": Tool(tool_backflip, "Perform a backflip. Takes no arguments."),
}


FULL_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + "\n"
    + pp.pformat({k: v for k, v in TOOLS.items()})
    + "\n"
    + SYSTEM_PROMPT_AFTER_TOOL_CALLS
)

# ----------------------------------------------------------------------
# 🧩  Helper utilities


def build_message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def parse_tool_call(message: str) -> tuple[str, dict[str, Any]] | None:
    try:
        payload = json.loads(message)
        if isinstance(payload, dict) and TOOL_CALL_TAG in payload:
            call = payload[TOOL_CALL_TAG]
            name = call["name"]
            args = call.get("arguments", {})
            if not isinstance(args, dict):
                raise ValueError("`arguments` must be a JSON object")
            return name, args
    except json.JSONDecodeError:
        pass
    except Exception:
        pass
    return None


def call_tool(name: str, args: dict[str, Any]) -> str | None:
    tool = TOOLS.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'."
    try:
        return tool.func(**args)
    except TypeError as exc:
        return f"Error: wrong arguments for tool '{name}': {exc}"
    except Exception as exc:
        return f"Error while executing tool '{name}': {exc}"


# ----------------------------------------------------------------------
# 🤖  Core agent loop (async)


async def _ai_completion(messages: list[dict[str, str]]) -> str:
    model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    response = await acompletion(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )

    return response.choices[0].message.content  # type: ignore


async def run_agent(
    user_query: str,
    messages: list[dict[str, str]] | None = None,
    completion_func: Callable[[list[dict[str, str]]], Awaitable[str]] | None = None,
    output_sink: OutputSink | None = None,
) -> list[dict[str, str]]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("❌ Environment variable GROQ_API_KEY not set.")

    # litellm.api_key = api_key

    full_system_prompt = (
        SYSTEM_PROMPT
        + "\n"
        + pp.pformat({k: v for k, v in TOOLS.items()})
        + "\n"
        + SYSTEM_PROMPT_AFTER_TOOL_CALLS
    )

    # Initialize messages if not provided
    if messages is None:
        messages = [
            build_message("system", full_system_prompt),
        ]

    # Add user query to conversation
    messages.append(build_message("user", user_query))

    completion = completion_func or _ai_completion
    if output_sink is None:
        from holo_chan.io.output_local import LocalSpeaker

        speak_out = LocalSpeaker()
    else:
        speak_out = output_sink

    for turn in range(1, MAX_TURNS + 1):
        try:
            print("🧠 Calling LLM...")
            assistant_msg = await completion(messages)
        except Exception as exc:
            print(f"❌ LiteLLM API request failed: {exc}", file=sys.stderr)
            break

        print(f"🧾 LLM raw output: {assistant_msg!r}")

        try:
            if assistant_msg is None:
                assistant_msg = ""
            else:
                assistant_msg = assistant_msg.strip()
        except (IndexError, AttributeError, ValueError):
            assistant_msg = ""  # Fallback for unexpected response format

        parsed = parse_tool_call(assistant_msg)

        if parsed is None:
            # No tool call - just a text response
            if not assistant_msg:
                break
            messages.append(build_message("assistant", assistant_msg))
            print("🗣️  Speaking response")
            await speak_out.speak(assistant_msg)
            break

        tool_name, tool_args = parsed
        print(f"🛠️  Tool call: {tool_name} {tool_args}")

        # Check if this is a stop tool
        if tool_name in ("wait_for_more", "done"):
            messages.append(build_message("assistant", assistant_msg))
            break

        tool_result = call_tool(tool_name, tool_args)

        messages.append(build_message("assistant", assistant_msg))
        messages.append(
            build_message("user", f"Result of `{tool_name}`: {tool_result}")
        )

    return messages
