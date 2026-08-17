"""Run TinyHarness with one blocking Pre hook and one observing Post hook."""

import argparse
import os
from pathlib import Path

from tiny_harness.agent.loop import agent_loop
from tiny_harness.models.chat_completions import ChatCompletionsProvider
from tiny_harness.runtime.events import NULL_EVENT_LOGGER, JsonlEventLogger
from tiny_harness.runtime.hooks import HookBlock, ToolHooks


def main() -> int:
    parser = argparse.ArgumentParser(description="TinyHarness Phase 5 hook demo")
    parser.add_argument("task")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--event-log", type=Path)
    args = parser.parse_args()

    api_key = os.getenv("TINYHARNESS_API_KEY")
    if not api_key:
        parser.error("TINYHARNESS_API_KEY is required")

    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        parser.error(f"workspace is not a directory: {workspace}")

    hooks = ToolHooks()

    def block_write_file(context):
        if context.tool_name == "write_file":
            return HookBlock("write_file is blocked by the Phase 5 demo")
        return None

    def report_result(context, result):
        print(
            f"[PostToolUse] {context.tool_name} returned "
            f"{len(result.content)} characters"
        )

    hooks.register_pre(block_write_file)
    hooks.register_post(report_result)

    provider = ChatCompletionsProvider(
        api_key=api_key,
        model=os.getenv("TINYHARNESS_MODEL", "deepseek-v4-flash"),
        base_url=os.getenv("TINYHARNESS_BASE_URL", "https://api.deepseek.com"),
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
                "Use the available tools to complete the user's task."
            ),
        },
        {"role": "user", "content": args.task},
    ]
    answer = agent_loop(
        provider,
        workspace,
        messages,
        event_logger=event_logger,
        tool_hooks=hooks,
    )
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
