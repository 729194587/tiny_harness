"""Command-line entry point for TinyHarness."""

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tiny_harness.agent.loop import agent_loop
from tiny_harness.models.chat_completions import ChatCompletionsProvider
from tiny_harness.runtime.events import NULL_EVENT_LOGGER, JsonlEventLogger

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyharness",
        description="Run the minimal TinyHarness coding agent.",
    )
    parser.add_argument("task", help="Coding task for the agent")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace available to the agent (default: current directory)",
    )
    parser.add_argument(
        "--max-turns",
        type=_positive_int,
        default=20,
        help="Maximum model calls (default: 20)",
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        help="Append lifecycle events to a JSONL file",
    )
    parser.add_argument(
        "--max-context-chars",
        type=_positive_int,
        help="Maximum compact-JSON characters sent as model context",
    )
    return parser


def _ask_permission(tool_name: str, arguments: Mapping[str, Any]) -> bool:
    """Ask the CLI user to approve one tool call; default to denial."""

    print("\nPermission required:")
    print(f"Tool: {tool_name}")
    print("Arguments:")
    print(json.dumps(arguments, ensure_ascii=False, indent=2))
    try:
        choice = input("Allow this tool call? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return choice in {"y", "yes"}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, run one task, and print the final answer."""

    parser = _parser()
    args = parser.parse_args(argv)

    api_key = os.getenv("TINYHARNESS_API_KEY")
    if not api_key:
        parser.error("TINYHARNESS_API_KEY is required")

    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        parser.error(f"workspace is not a directory: {workspace}")

    provider = ChatCompletionsProvider(
        api_key=api_key,
        model=os.getenv("TINYHARNESS_MODEL", DEFAULT_MODEL),
        base_url=os.getenv("TINYHARNESS_BASE_URL", DEFAULT_BASE_URL),
    )
    event_logger = (
        JsonlEventLogger(args.event_log)
        if args.event_log is not None
        else NULL_EVENT_LOGGER
    )
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a coding agent working in {workspace}. "
                "Use the available tools to complete the user's task. "
                "Before starting a multi-step task, use todo_write to plan "
                "the steps and update their status as you work."
            ),
        },
        {"role": "user", "content": args.task},
    ]
    answer = agent_loop(
        provider,
        workspace,
        messages,
        max_turns=args.max_turns,
        permission_prompt=_ask_permission,
        event_logger=event_logger,
        max_context_chars=args.max_context_chars,
    )
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
