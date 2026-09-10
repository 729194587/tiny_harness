import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evals.swe_bench_lite.data import SweTask
from evals.swe_bench_lite.docker_workspace import DockerTaskEnvironment


MODULE = "evals.swe_bench_lite.docker_workspace"


class DockerCleanupTest(unittest.TestCase):
    def setUp(self):
        self.environment = DockerTaskEnvironment(
            SweTask("owner__repo-1", "owner/repo", "abcdef1", "issue", "image"),
            command_timeout_seconds=7,
        )
        self.temporary = Mock()
        self.environment._temporary = self.temporary
        self.environment.workspace = Path.cwd()
        self.posix = SimpleNamespace(name="posix", getuid=lambda: 1001, getgid=lambda: 1002)

    def test_posix_restores_before_container_removal_and_host_cleanup(self):
        operations = Mock()
        with patch(f"{MODULE}.os", self.posix), patch(f"{MODULE}._run") as run:
            operations.attach_mock(run, "run")
            operations.attach_mock(self.temporary.cleanup, "cleanup")
            self.environment.close()
        calls = operations.mock_calls
        self.assertEqual(calls[0].args[0], [
            "docker", "exec", "--user", "0:0", self.environment.container_name,
            "chown", "-R", "-h", "1001:1002", "/testbed",
        ])
        self.assertEqual(calls[0].kwargs, {"timeout": 7, "check": False})
        self.assertEqual([call.args[0][1] for call in calls[1:3]], ["rm", "rm"])
        self.assertEqual(calls[3][0], "cleanup")
        self.assertIsNone(self.environment._temporary)

    def test_windows_without_uid_apis_skips_chown(self):
        with patch(f"{MODULE}.os", SimpleNamespace(name="nt")), patch(f"{MODULE}._run") as run:
            self.environment.close()
        self.assertEqual([call.args[0][1] for call in run.call_args_list], ["rm", "rm"])
        self.temporary.cleanup.assert_called_once()

    def test_restoration_failures_still_remove_containers_and_workspace(self):
        for failure in (OSError("docker unavailable"), subprocess.TimeoutExpired("docker", 7),
                        subprocess.CompletedProcess([], 1, "", "denied")):
            with self.subTest(failure=failure):
                self.environment._temporary = self.temporary
                self.temporary.reset_mock()
                with patch(f"{MODULE}.os", self.posix), patch(
                    f"{MODULE}._run", side_effect=[failure, None, None]
                ) as run:
                    self.environment.close()
                self.assertEqual(run.call_count, 3)
                self.temporary.cleanup.assert_called_once()

    def test_host_cleanup_failure_is_reported_without_primary_exception(self):
        self.temporary.cleanup.side_effect = PermissionError("cleanup failed")
        with patch(f"{MODULE}.os", self.posix), patch(f"{MODULE}._run"):
            with self.assertRaisesRegex(PermissionError, "cleanup failed"):
                self.environment.__exit__(None, None, None)

    def test_body_exception_survives_restoration_and_host_cleanup_failure(self):
        primary = RuntimeError("evaluation failed")
        self.temporary.cleanup.side_effect = PermissionError("cleanup failed")
        with patch(f"{MODULE}.os", self.posix), patch(
            f"{MODULE}._run", side_effect=OSError("docker unavailable")
        ) as run:
            with self.assertRaises(RuntimeError) as caught:
                try:
                    raise primary
                except BaseException as exc:
                    self.environment.__exit__(type(exc), exc, exc.__traceback__)
                    raise
        self.assertIs(caught.exception, primary)
        self.assertEqual(run.call_count, 3)
        self.temporary.cleanup.assert_called_once()

    def test_setup_failure_uses_same_cleanup_and_preserves_primary(self):
        primary = subprocess.TimeoutExpired("docker run", 7)
        self.temporary.name = str(Path.cwd())
        self.temporary.cleanup.side_effect = PermissionError("cleanup failed")
        with patch(f"{MODULE}.tempfile.TemporaryDirectory", return_value=self.temporary), \
             patch.object(Path, "mkdir"), patch.object(Path, "open", unittest.mock.mock_open()), \
             patch(f"{MODULE}.os", self.posix), \
             patch(f"{MODULE}._run", side_effect=[None, None, None, primary, None, None, None]) as run:
            with self.assertRaises(subprocess.TimeoutExpired) as caught:
                self.environment.__enter__()
        self.assertIs(caught.exception, primary)
        self.assertEqual(run.call_args_list[4].args[0][5], "chown")
        self.assertEqual([call.args[0][1] for call in run.call_args_list[5:]], ["rm", "rm"])
        self.temporary.cleanup.assert_called_once()

    def test_setup_makes_workspace_writable_after_reset_before_host_write(self):
        self.temporary.name = str(Path.cwd())
        operations = Mock()
        with patch(f"{MODULE}.tempfile.TemporaryDirectory", return_value=self.temporary), \
             patch.object(Path, "mkdir"), \
             patch.object(Path, "open", unittest.mock.mock_open()) as host_open, \
             patch(f"{MODULE}._run") as run:
            operations.attach_mock(run, "run")
            operations.attach_mock(host_open, "host_open")
            self.assertIs(self.environment.__enter__(), self.environment)
        calls = operations.mock_calls
        reset_index = next(
            i for i, call in enumerate(calls)
            if call[0] == "run" and "git reset --hard" in call.args[0][-1]
        )
        permission_call = calls[reset_index + 1]
        self.assertEqual(permission_call.args[0], [
            "docker", "exec", "--user", "0:0", self.environment.container_name,
            "chmod", "-R", "a+rwX", "/testbed",
        ])
        self.assertEqual(permission_call.kwargs, {"timeout": 7, "check": True})
        self.assertEqual(calls[reset_index + 2][0], "host_open")

    def test_permission_setup_failure_closes_environment_and_propagates(self):
        self.temporary.name = str(Path.cwd())
        primary = subprocess.CalledProcessError(1, "chmod")

        def fake_run(argv, **kwargs):
            if "chmod" in argv:
                raise primary

        with patch(f"{MODULE}.tempfile.TemporaryDirectory", return_value=self.temporary), \
             patch(f"{MODULE}.os", self.posix), \
             patch(f"{MODULE}._run", side_effect=fake_run) as run, \
             patch.object(Path, "open") as host_open:
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                self.environment.__enter__()
        self.assertIs(caught.exception, primary)
        host_open.assert_not_called()
        self.assertEqual(run.call_args_list[-3].args[0][5], "chown")
        self.assertEqual([call.args[0][1] for call in run.call_args_list[-2:]], ["rm", "rm"])
        self.temporary.cleanup.assert_called_once()
        self.assertIsNone(self.environment.workspace)


if __name__ == "__main__":
    unittest.main()
