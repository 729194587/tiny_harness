import unittest

from tiny_harness.runtime.permissions import (
    DEFAULT_PERMISSION_POLICY,
    PermissionDecision,
    is_read_only_shell_command,
    resolve_permission,
)


class FixedPolicy:
    def __init__(self, decision) -> None:
        self.decision = decision

    def decide(self, tool_name, arguments):
        return self.decision


class RaisingPolicy:
    def decide(self, tool_name, arguments):
        raise RuntimeError("policy failed")


class PermissionTest(unittest.TestCase):
    def test_default_policy_allows_non_shell_tools(self) -> None:
        for tool_name in (
            "read_file",
            "write_file",
            "edit_file",
            "list_files",
            "todo_write",
            "task",
            "load_skill",
            "compact",
        ):
            with self.subTest(tool_name=tool_name):
                self.assertIs(
                    DEFAULT_PERMISSION_POLICY.decide(tool_name, {}),
                    PermissionDecision.ALLOW,
                )

    def test_default_policy_allows_clear_read_only_bash(self) -> None:
        commands = (
            'find . -name "*.py" -type f | sort',
            'rg --files -g "*.py"',
            "dir /s /b *.py",
            "git status --short",
            "git diff -- README.md",
            "cd",
            r"type local.txt",
            r"type .\local.txt",
            r"type foo\bar.txt",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIs(
                    DEFAULT_PERMISSION_POLICY.decide(
                        "bash",
                        {"command": command},
                    ),
                    PermissionDecision.ALLOW,
                )

    def test_default_policy_asks_for_mutating_or_ambiguous_bash(self) -> None:
        commands = (
            "rm output.txt",
            "del output.txt",
            "echo bad > output.txt",
            'python -c "open(\'output.txt\', \'w\').write(\'bad\')"',
            "git reset --hard",
            "git diff --output=changes.patch",
            "find . -delete",
            'rg --pre "python mutate.py" pattern .',
            "sort -o sorted.txt input.txt",
            "dir ..",
            "cat /etc/passwd",
            "echo %USERPROFILE%",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIs(
                    DEFAULT_PERMISSION_POLICY.decide(
                        "bash",
                        {"command": command},
                    ),
                    PermissionDecision.ASK,
                )

    def test_read_only_classifier_fails_closed_on_invalid_input(self) -> None:
        for command in (None, "", "   ", 123, "rg --files &&"):
            with self.subTest(command=command):
                self.assertFalse(is_read_only_shell_command(command))

    def test_windows_external_path_forms_require_approval(self) -> None:
        commands = (
            r"type C:secret.txt",
            r"type \Windows\win.ini",
            r"type .\..\secret.txt",
            r"type foo\..\..\secret.txt",
            r"type %USERPROFILE:~0,99%\secret.txt",
            r"type %USERPROFILE:str1=str2%\secret.txt",
            r"type !USERPROFILE!\secret.txt",
            r"type C^:\Windows\win.ini",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertFalse(is_read_only_shell_command(command))
                self.assertIs(
                    DEFAULT_PERMISSION_POLICY.decide(
                        "bash",
                        {"command": command},
                    ),
                    PermissionDecision.ASK,
                )

    def test_default_policy_denies_unrecognized_tool(self) -> None:
        self.assertIs(
            DEFAULT_PERMISSION_POLICY.decide("unknown", {}),
            PermissionDecision.DENY,
        )

    def test_allow_and_deny_do_not_prompt(self) -> None:
        def unexpected_prompt(tool_name, arguments):
            raise AssertionError("prompt should not be called")

        self.assertIs(
            resolve_permission(
                FixedPolicy(PermissionDecision.ALLOW),
                "tool",
                {},
                unexpected_prompt,
            ),
            PermissionDecision.ALLOW,
        )
        self.assertIs(
            resolve_permission(
                FixedPolicy(PermissionDecision.DENY),
                "tool",
                {},
                unexpected_prompt,
            ),
            PermissionDecision.DENY,
        )

    def test_ask_resolves_from_prompt(self) -> None:
        policy = FixedPolicy(PermissionDecision.ASK)

        self.assertIs(
            resolve_permission(policy, "bash", {}, lambda *_: True),
            PermissionDecision.ALLOW,
        )
        self.assertIs(
            resolve_permission(policy, "bash", {}, lambda *_: False),
            PermissionDecision.DENY,
        )

    def test_ask_without_prompt_is_denied(self) -> None:
        self.assertIs(
            resolve_permission(
                FixedPolicy(PermissionDecision.ASK),
                "bash",
                {},
            ),
            PermissionDecision.DENY,
        )

    def test_policy_or_prompt_failure_is_denied(self) -> None:
        def failing_prompt(tool_name, arguments):
            raise EOFError

        self.assertIs(
            resolve_permission(RaisingPolicy(), "bash", {}),
            PermissionDecision.DENY,
        )
        self.assertIs(
            resolve_permission(
                FixedPolicy(PermissionDecision.ASK),
                "bash",
                {},
                failing_prompt,
            ),
            PermissionDecision.DENY,
        )

    def test_invalid_policy_result_is_denied(self) -> None:
        self.assertIs(
            resolve_permission(FixedPolicy("allow"), "tool", {}),
            PermissionDecision.DENY,
        )


if __name__ == "__main__":
    unittest.main()
