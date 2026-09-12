"""Explicit replacement of the current run's bounded working note."""

from typing import TYPE_CHECKING

from tiny_harness.runtime.events import EventType, hash_text
from tiny_harness.runtime.working_memory import MAX_NOTE_CHARS
from tiny_harness.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


def build_tools(context: "AgentRunContext") -> tuple[ToolDefinition, ...]:
    memory = getattr(context, "working_memory", None)
    if memory is None:
        return ()

    def update(call, arguments):
        if set(arguments) != {"note"}:
            raise ValueError("Expected only the note argument")
        note = arguments["note"]
        memory.update(note)
        context.event_logger.emit(EventType.WORKING_MEMORY_UPDATED, {
            "turn": context.current_turn,
            "tool_call_id": call.id,
            "note_length": len(memory.note),
            "note_hash": hash_text(memory.note),
            "truncated": len(note) > MAX_NOTE_CHARS,
        })
        return f"工作记忆已替换（{len(memory.note)} 字符，上限 {MAX_NOTE_CHARS}）。"

    return (ToolDefinition(
        name="update_working_memory",
        description=(
            "Replace the current task's working note with concise state needed to continue. "
            "Update only when important task understanding or progress changes materially "
            "and the state will still be needed later. Do not merely repeat recent tool "
            "output or Todo items. "
            "The note survives context compaction; it is not persistent memory. "
            f"Only the first {MAX_NOTE_CHARS} characters are retained. Empty text clears it."
        ),
        parameters={
            "type": "object",
            "properties": {"note": {"type": "string", "maxLength": MAX_NOTE_CHARS}},
            "required": ["note"],
            "additionalProperties": False,
        },
        execute=update,
    ),)
