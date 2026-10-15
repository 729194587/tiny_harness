"""Offline Docker command construction and real Bash semantics; no Docker needed."""

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.swe_bench_lite.data import SweTask
from evals.swe_bench_lite.docker_workspace import DockerShellRunner, DockerTaskEnvironment


class DockerShellPipefailTest(unittest.TestCase):
    def test_opt_in_changes_only_model_command(self):
        workspace = Path("workspace").resolve()
        command = "false | tail -20"
        runner = DockerShellRunner("container", workspace, pipefail=True)
        argv = runner.command_argv(workspace, command)
        self.assertEqual(argv[-2:], ["-lc", "umask 000\n" + command])
        with patch(
            "evals.swe_bench_lite.docker_workspace._run",
            return_value=subprocess.CompletedProcess([], 1, "", ""),
        ) as execute:
            self.assertEqual(runner.run(workspace, command), "Exit code: 1")
        self.assertEqual(
            execute.call_args.args[0],
            [*argv[:-1], "umask 000\nset -o pipefail\n" + command],
        )
        self.assertFalse(execute.call_args.kwargs["check"])

    def test_environment_configuration_preserves_initialization_and_exec(self):
        task = SweTask("owner__repo-1", "owner/repo", "abcdef1", "issue", "image")
        for strict in (False, True):
            with self.subTest(strict=strict):
                environment = DockerTaskEnvironment(task, model_shell_pipefail=strict)
                calls = []

                def fake_run(argv, **kwargs):
                    calls.append(argv)
                    if argv[1] == "cp":
                        (environment.workspace / ".git" / "info").mkdir(parents=True)
                    return subprocess.CompletedProcess(argv, 0, "", "")

                with patch(
                    "evals.swe_bench_lite.docker_workspace._run", side_effect=fake_run,
                ):
                    with environment:
                        self.assertEqual(environment.shell_runner.pipefail, strict)
                        self.assertTrue(all("set -o pipefail" not in a[-1] for a in calls))
                        environment.exec("false | tail -20")
                        self.assertEqual(calls[-1][-1], "umask 000\nfalse | tail -20")
                        environment.shell_runner.run(environment.workspace, "false | tail -20")
                        self.assertEqual("set -o pipefail" in calls[-1][-1], strict)


@unittest.skipUnless(os.name == "posix" and Path("/bin/bash").is_file(), "Requires local Bash")
class DockerShellBashSemanticsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        # Model login startup explicitly disabling pipefail. The runner's prefix
        # must take effect afterwards; default mode must leave startup untouched.
        startup = self.workspace / "startup.sh"
        startup.write_text("set +o pipefail\nexport TINYHARNESS_PIPEFAIL_STARTUP=loaded\n")
        environment = patch.dict(os.environ, {"BASH_ENV": str(startup)})
        environment.start()
        self.addCleanup(environment.stop)

    def run_model(self, command, *, strict=False):
        def local_bash(argv, *, timeout, check):
            self.assertEqual(argv[:5], ["docker", "exec", "--workdir", "/testbed", "container"])
            return subprocess.run(
                argv[5:], cwd=self.workspace, timeout=timeout, check=check,
                capture_output=True, text=True,
            )

        runner = DockerShellRunner("container", self.workspace, pipefail=strict)
        with patch("evals.swe_bench_lite.docker_workspace._run", side_effect=local_bash):
            return runner.run(self.workspace, command)

    def assert_status(self, command, expected, *, strict):
        result = self.run_model(command, strict=strict)
        self.assertEqual(result.splitlines()[0], f"Exit code: {expected}", result)
        return result

    def test_normal_pipeline_and_upstream_failure_in_both_modes(self):
        for strict in (False, True):
            with self.subTest(strict=strict):
                result = self.assert_status("printf 'ok\\n' | tail -20", 0, strict=strict)
                self.assertIn("\nok", result)
                self.assert_status("(exit 7) | tail -20", 7 if strict else 0, strict=strict)

    def test_multistage_pipeline_uses_rightmost_failure(self):
        for strict in (False, True):
            with self.subTest(strict=strict):
                self.assert_status(
                    "(exit 7) | (exit 3) | tail -20", 3 if strict else 0, strict=strict,
                )

    def test_python_failure_survives_tail(self):
        command = f"{shlex.quote(sys.executable)} -c 'raise SystemExit(7)' 2>&1 | tail -20"
        self.assert_status(command, 0, strict=False)
        self.assert_status(command, 7, strict=True)

    def test_grep_no_match_is_nonzero_only_in_strict_mode(self):
        for strict in (False, True):
            with self.subTest(strict=strict):
                self.assert_status(
                    "printf 'ok\\n' | grep missing | tail -20", 1 if strict else 0, strict=strict,
                )

    def test_head_and_explicit_sigpipe_are_not_silently_ignored(self):
        # Explicitly deliver SIGPIPE to the producer: no buffer-size or scheduling
        # race is needed to check reporting when the consumer is head.
        command = "(trap - PIPE; kill -s PIPE \"$BASHPID\") | head -n 1"
        self.assert_status(command, 0, strict=False)
        self.assert_status(command, 141, strict=True)

    def test_and_or_control_flow_and_expected_failure(self):
        command = "(exit 7) | tail -20 && printf success || printf fallback"
        self.assertIn("\nsuccess", self.assert_status(command, 0, strict=False))
        self.assertIn("\nfallback", self.assert_status(command, 0, strict=True))
        self.assert_status("(exit 7) | tail -20 && printf unreachable", 7, strict=True)
        self.assert_status("(exit 7) | tail -20 || true", 0, strict=True)

    def test_no_errexit_or_nounset_and_later_commands_can_mask_failure(self):
        result = self.assert_status(
            "unset TINYHARNESS_PIPEFAIL_UNSET; false | tail -20; "
            "printf 'continued:%s' \"$TINYHARNESS_PIPEFAIL_UNSET\"",
            0, strict=True,
        )
        self.assertIn("\ncontinued:", result)

    def test_strict_mode_applies_after_startup(self):
        result = self.assert_status(
            'printf "%s\\n" "$TINYHARNESS_PIPEFAIL_STARTUP"; (exit 7) | tail -20',
            7, strict=True,
        )
        self.assertIn("\nloaded", result)


if __name__ == "__main__":
    unittest.main()
