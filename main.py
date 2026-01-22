import asyncio
import json
import os
import pprint as pp
import sys
from dataclasses import dataclass
from typing import Any, Callable

import zmq
from litellm import acompletion

from tts import speak

# ----------------------------------------------------------------------
# 🎛️  CONFIGURATION

GROQ_MODEL = (
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
    print(f"Dancing dance number {dance!r}")
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
    except Exception as exc:
        print(f"⚠️  Failed to parse tool call: {exc}", file=sys.stderr)
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


async def run_agent(user_query: str) -> None:
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

    messages: list[dict[str, str]] = [
        build_message("system", full_system_prompt),
        build_message("user", user_query),
    ]

    for turn in range(1, MAX_TURNS + 1):
        try:
            response = await acompletion(
                model=GROQ_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
        except Exception as exc:
            sys.exit(f"❌ LiteLLM API request failed: {exc}")

        try:
            assistant_msg = response.choices[0].message.content
            if assistant_msg is None:
                assistant_msg = ""
            else:
                assistant_msg = assistant_msg.strip()
        except (IndexError, AttributeError, ValueError) as e:
            print("ERROR GETTING MESSAGE FROM AI", e)
            assistant_msg = ""  # Fallback for unexpected response format
        print(f"\n🗣️  Assistant (turn {turn}):\n{assistant_msg}\n")

        parsed = parse_tool_call(assistant_msg)

        if parsed is None:
            # No tool call - just a text response
            if not assistant_msg:
                # Empty response - treat as stop signal
                print("🛑 STOPPING: Empty response received")
                break
            print("ℹ️  No tool call detected, treating as text response")
            messages.append(build_message("assistant", assistant_msg))
            await speak(assistant_msg)
            continue

        tool_name, tool_args = parsed
        print(f"🔧  Detected tool call → {tool_name}{tool_args}")

        # Check if this is a stop tool
        if tool_name in ("wait_for_more", "done"):
            print(f"🛑 STOPPING: Tool '{tool_name}' was called")
            messages.append(build_message("assistant", assistant_msg))
            break

        print(f"⚙️  Executing tool: {tool_name}")
        tool_result = call_tool(tool_name, tool_args)
        print(f"📦  Tool result:\n{tool_result}\n")

        messages.append(build_message("assistant", assistant_msg))
        messages.append(
            build_message("user", f"Result of `{tool_name}`: {tool_result}")
        )

    else:
        print("⚠️  Reached MAX_TURNS without a final answer.")

    print(f"\n{'=' * 60}")
    print(f"🏁 Agent loop finished")
    print(f"{'=' * 60}\n")


# ----------------------------------------------------------------------
# 🏁  Example driver


async def main() -> None:
    print("Enter prompt")
    query = input(">>> ").strip()
    await run_agent(query)


if __name__ == "__main__":
    asyncio.run(main())
