"""Disposable Docker workspace used by SWE-bench rollouts and calibration."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from tiny_harness.runtime.shell_runner import DEFAULT_SHELL_TIMEOUT_SECONDS

from .data import SweTask

CONTAINER_WORKSPACE = "/testbed"
_COMMIT = re.compile(r"[0-9a-fA-F]{7,64}\Z")


def _run(argv: list[str], *, timeout: float, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=check,
    )


class DockerShellRunner:
    """Execute shell commands in one container's bind-mounted /testbed."""

    def __init__(
        self,
        container_name: str,
        host_workspace: Path,
        *,
        timeout_seconds: float = DEFAULT_SHELL_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Shell runner timeout must be positive")
        self.container_name = container_name
        self.host_workspace = host_workspace.resolve()
        self.timeout_seconds = timeout_seconds

    def command_argv(self, workspace: Path, command: str) -> list[str]:
        if workspace.resolve() != self.host_workspace:
            raise ValueError("Shell workspace does not match the Docker bind mount")
        return [
            "docker",
            "exec",
            "--workdir",
            CONTAINER_WORKSPACE,
            self.container_name,
            "/bin/bash",
            "-lc",
            command,
        ]

    def run(self, workspace: Path, command: str) -> str:
        completed = _run(
            self.command_argv(workspace, command),
            timeout=self.timeout_seconds,
            check=False,
        )
        output = (completed.stdout + completed.stderr).strip()
        detail = f"\n{output}" if output else ""
        return f"Exit code: {completed.returncode}{detail}"


class DockerTaskEnvironment:
    """Own a single disposable task container and its sole workspace copy."""

    def __init__(
        self,
        task: SweTask,
        *,
        command_timeout_seconds: float = 1200,
        network_mode: str | None = None,
    ) -> None:
        if not _COMMIT.fullmatch(task.base_commit):
            raise ValueError(f"Invalid base commit for {task.instance_id}")
        self.task = task
        self.command_timeout_seconds = command_timeout_seconds
        self.network_mode = network_mode
        suffix = uuid4().hex[:12]
        self.bootstrap_name = f"tinyharness-copy-{suffix}"
        self.container_name = f"tinyharness-agent-{suffix}"
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self.workspace: Path | None = None
        self.shell_runner: DockerShellRunner | None = None

    def __enter__(self) -> DockerTaskEnvironment:
        self._temporary = tempfile.TemporaryDirectory(prefix="tinyharness-swe-")
        self.workspace = Path(self._temporary.name).resolve()
        try:
            _run(
                ["docker", "create", "--name", self.bootstrap_name, self.task.image],
                timeout=self.command_timeout_seconds,
            )
            _run(
                [
                    "docker",
                    "cp",
                    f"{self.bootstrap_name}:{CONTAINER_WORKSPACE}/.",
                    str(self.workspace),
                ],
                timeout=self.command_timeout_seconds,
            )
            self._remove_container(self.bootstrap_name)
            mount = f"type=bind,source={self.workspace},target={CONTAINER_WORKSPACE}"
            run_argv = [
                "docker",
                "run",
                "--detach",
                "--name",
                self.container_name,
                "--mount",
                mount,
                "--workdir",
                CONTAINER_WORKSPACE,
            ]
            if self.network_mode is not None:
                run_argv.extend(["--network", self.network_mode])
            run_argv.extend(
                [
                    "--entrypoint",
                    "/bin/bash",
                    self.task.image,
                    "-lc",
                    "while true; do sleep 3600; done",
                ]
            )
            _run(
                run_argv,
                timeout=self.command_timeout_seconds,
            )
            self.shell_runner = DockerShellRunner(
                self.container_name,
                self.workspace,
                timeout_seconds=self.command_timeout_seconds,
            )
            self.exec(
                "git config --global --add safe.directory /testbed && "
                "git config --local core.fileMode false && "
                f"git reset --hard {self.task.base_commit} && git clean -fdx",
                check=True,
            )
            # Copy/reset can leave files owned by a different UID from the host
            # agent. Both host tools and container commands need write access to
            # this disposable bind mount, including directories and Git metadata.
            _run(
                [
                    "docker", "exec", "--user", "0:0", self.container_name,
                    "chmod", "-R", "a+rwX", CONTAINER_WORKSPACE,
                ],
                timeout=self.command_timeout_seconds,
                check=True,
            )
            exclude = self.workspace / ".git" / "info" / "exclude"
            exclude.parent.mkdir(parents=True, exist_ok=True)
            with exclude.open("a", encoding="utf-8") as stream:
                stream.write("\n.tinyharness/\n")
            return self
        except BaseException:
            self.close(suppress_errors=True)
            raise

    def exec(self, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        if self.workspace is None or self.shell_runner is None:
            raise RuntimeError("Docker task environment is not running")
        return _run(
            self.shell_runner.command_argv(self.workspace, command),
            timeout=self.command_timeout_seconds,
            check=check,
        )

    def collect_patch(self) -> str:
        """Collect tracked, deleted, untracked, and binary changes after rollout."""

        self.exec("git add -N -A", check=True)
        return self.exec(
            f"git diff --binary --no-ext-diff {self.task.base_commit} --",
            check=True,
        ).stdout

    def _remove_container(self, name: str) -> None:
        try:
            _run(
                ["docker", "rm", "--force", name],
                timeout=self.command_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    def close(self, *, suppress_errors: bool = False) -> None:
        # Attempt even after a failed docker run: it may have started the container.
        if self._temporary is not None and os.name == "posix":
            getuid = getattr(os, "getuid", None)
            getgid = getattr(os, "getgid", None)
            if getuid is not None and getgid is not None:
                try:
                    _run(
                        [
                            "docker", "exec", "--user", "0:0", self.container_name,
                            "chown", "-R", "-h", f"{getuid()}:{getgid()}",
                            CONTAINER_WORKSPACE,
                        ],
                        timeout=self.command_timeout_seconds,
                        check=False,
                    )
                except (OSError, subprocess.SubprocessError):
                    pass
        self._remove_container(self.container_name)
        self._remove_container(self.bootstrap_name)
        if self._temporary is not None:
            try:
                self._temporary.cleanup()
            except Exception:
                if not suppress_errors:
                    raise
                # Keep the temporary directory reference when deletion failed.
                return
            self._temporary = None
        self.workspace = None
        self.shell_runner = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(suppress_errors=exc_type is not None)
