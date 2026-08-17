import unittest

from tiny_harness.runtime.permissions import (
    DEFAULT_PERMISSION_POLICY,
    PermissionDecision,
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
    def test_default_policy_allows_file_tools(self) -> None:
        for tool_name in ("read_file", "write_file", "edit_file", "list_files"):
            with self.subTest(tool_name=tool_name):
                self.assertIs(
                    DEFAULT_PERMISSION_POLICY.decide(tool_name, {}),
                    PermissionDecision.ALLOW,
                )

    def test_default_policy_asks_for_bash(self) -> None:
        self.assertIs(
            DEFAULT_PERMISSION_POLICY.decide("bash", {"command": "echo ok"}),
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
