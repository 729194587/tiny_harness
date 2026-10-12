import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tiny_harness.agent.loop import run_agent
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.runtime.hooks import HookBlock, ToolHooks
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.tools.outline import code_outline, MAX_OUTLINE_OUTPUT_CHARS, MAX_OUTLINE_FILE_BYTES
from tiny_harness.tools.discovery import discover_tools
from tiny_harness.tools.registry import dispatch


class CodeOutlineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name) / "workspace"
        self.workspace.mkdir()
        self.file = self.workspace / "sample.py"
        self.file.write_text(
            "@decorator\nclass Example:\n    def method(self):\n"
            "        async def nested():\n            pass\n"
            "        return nested\n    @classmethod\n"
            "    async def create(cls):\n        pass\n"
            "def outer():\n    if True:\n        class Inner:\n"
            "            def run(self):\n                pass\n"
            "async def top():\n    pass\n", encoding="utf-8")

    def test_nested_definitions_and_inclusive_lines(self):
        self.assertEqual(code_outline(self.workspace, "sample.py"),
                         "class Example [2-9]\n  method method [3-6]\n"
                         "    async function nested [4-5]\n  async method create [8-9]\n"
                         "function outer [10-14]\n  class Inner [12-14]\n"
                         "    method run [13-14]\nasync function top [15-16]")

    def test_limits_empty_file_and_source_encoding(self):
        self.assertNotIn("truncated", code_outline(self.workspace, "sample.py", 8))
        self.assertTrue(code_outline(self.workspace, "sample.py", 1).endswith("truncated after 1 symbols"))
        self.file.write_text("value = 1\n", encoding="utf-8")
        self.assertEqual(code_outline(self.workspace, "sample.py"), "(no definitions)")
        self.file.write_bytes(b"# coding: latin-1\ndef caf\xe9(): pass\n")
        self.assertEqual(code_outline(self.workspace, "sample.py"), "function café [2-2]")
        self.file.write_text("\n".join(f"def f{i}_" + "x" * 1000 + "(): pass" for i in range(100)), encoding="utf-8")
        result = code_outline(self.workspace, "sample.py")
        self.assertLessEqual(len(result), MAX_OUTLINE_OUTPUT_CHARS)
        self.assertIn("output character limit", result)
        self.file.write_bytes(b"#" * (MAX_OUTLINE_FILE_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "2 MiB"):
            code_outline(self.workspace, "sample.py")

    def test_invalid_input_and_syntax(self):
        for path in ("missing.py", ".", "sample.txt", "", None):
            with self.subTest(path=path), self.assertRaises(ValueError):
                code_outline(self.workspace, path)
        for limit in (0, 501, True, 1.5, None):
            with self.subTest(limit=limit), self.assertRaises((ValueError, TypeError)):
                code_outline(self.workspace, "sample.py", limit)
        self.file.write_text("def broken(:\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "syntax error at line 1"):
            code_outline(self.workspace, "sample.py")

    def test_path_boundary_and_symlinks(self):
        outside = self.workspace.parent / "outside.py"
        outside.write_text("def secret(): pass", encoding="utf-8")
        for path in ("../outside.py", str(outside)):
            with self.assertRaisesRegex(ValueError, "escapes workspace"):
                code_outline(self.workspace, path)
        try:
            (self.workspace / "link.py").symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Symlinks unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "escapes workspace"):
            code_outline(self.workspace, "link.py")

    def test_shared_dispatch_errors_hooks_permissions_and_safe_trace(self):
        registry = discover_tools(SimpleNamespace(workspace=self.workspace, todo_manager=None,
                                  subagent_runner=None, skill_catalog=None, test_runner=None))
        call = ToolCall("outline", "code_outline", '{"path":"sample.py"}')
        self.assertIn("class Example", dispatch(registry, call).content)
        self.assertEqual(registry.lookup("code_outline").trace_metadata({"path": "secret"}), {})
        bad = ToolCall("bad", "code_outline", '{"path":"missing.py"}')
        self.assertTrue(dispatch(registry, bad).content.startswith("Error:"))
        hooks = ToolHooks()
        hooks.register_pre(lambda context: HookBlock("blocked"))
        deny = SimpleNamespace(decide=lambda *args: PermissionDecision.DENY)
        with patch("tiny_harness.tools.outline.code_outline", side_effect=AssertionError("executed")):
            self.assertIn("blocked", dispatch(registry, call, tool_hooks=hooks).content)
            self.assertIn("Permission denied", dispatch(registry, call, permission_policy=deny).content)

    def test_scripted_agent_tool_call_and_matching_result(self):
        calls = []
        responses = iter([
            ModelResponse(None, None, [ToolCall("outline", "code_outline", json.dumps({"path": "sample.py"}))], "tool_calls"),
            ModelResponse("done", None, [], "stop"),
        ])

        def complete(messages, tools):
            calls.append((list(messages), tools))
            return next(responses)

        messages = [{"role": "user", "content": "Outline sample.py"}]
        self.assertEqual(run_agent(SimpleNamespace(complete=complete), self.workspace, messages), "done")
        self.assertIn("code_outline", [s["function"]["name"] for s in calls[0][1]])
        result = next(m for m in calls[1][0] if m.get("tool_call_id") == "outline")
        self.assertIn("class Example [2-9]", result["content"])
