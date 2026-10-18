"""Injectable execution capability used by the shell tool."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

DEFAULT_SHELL_TIMEOUT_SECONDS = 120
SHELL_CLEANUP_TIMEOUT_SECONDS = 5


class ShellRunner(Protocol):
    """Execute one model-requested shell command for a workspace."""

    def run(self, workspace: Path, command: str) -> str:
        """Return a mechanical exit-code and combined output report."""
        ...


def _terminate_process_tree(process: subprocess.Popen) -> bool:
    """Attempt tree termination with bounded waits, even if cleanup fails.

    Use taskkill /T on Windows, or signal the process group elsewhere.
    True means tree/group termination reported success and the shell wait
    completed; detached descendants may be outside this cleanup scope.
    """
    tree_stopped = False
    try:
        if os.name == "nt":
            taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
            result = subprocess.run(
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=SHELL_CLEANUP_TIMEOUT_SECONDS,
            )
            tree_stopped = result.returncode == 0
        else:
            os.killpg(process.pid, signal.SIGKILL)
            tree_stopped = True
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return tree_stopped


class SubprocessShellRunner:
    """Run non-interactive host-shell commands with bounded timeout cleanup.

    Use cmd.exe on Windows and subprocess's default shell elsewhere. The
    resolved workspace sets cwd only; it is not a filesystem sandbox.
    """

    def __init__(self, timeout_seconds: float = DEFAULT_SHELL_TIMEOUT_SECONDS) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Shell runner timeout must be positive")
        self.timeout_seconds = timeout_seconds

    def run(self, workspace: Path, command: str) -> str:
        """Return exit status and stdout followed by stderr, including nonzero exits.

        Decode output as UTF-8 with replacement and normalize newlines. The
        streams are concatenated, not interleaved in emission order.
        Timeouts raise TimeoutError after bounded cleanup attempts;
        interrupts also trigger cleanup before propagating. Cleanup does
        not guarantee that every descendant has stopped.
        """

        # Files avoid pipe-reader threads and EOF waits on inherited child handles.
        # Do not use Popen's context manager: its exit can wait without a timeout.
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            options = {}
            if os.name == "nt":
                options["executable"] = str(
                    Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "cmd.exe"
                )
            else:
                options["start_new_session"] = True
            process = subprocess.Popen(
                command, shell=True, cwd=workspace.resolve(), stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, **options,
            )
            try:
                process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                stopped = _terminate_process_tree(process)
                detail = "Process tree terminated." if stopped else (
                    "Process-tree cleanup could not be confirmed; descendants may still be running."
                )
                raise TimeoutError(
                    f"Shell command timed out after {self.timeout_seconds:g} seconds. {detail}"
                ) from None
            except BaseException:
                _terminate_process_tree(process)
                raise
            output = []
            for stream in (stdout, stderr):
                # Read a finite snapshot if a detached child continues writing.
                size = os.fstat(stream.fileno()).st_size
                stream.seek(0)
                decoded = stream.read(size).decode("utf-8", errors="replace")
                output.append(decoded.replace("\r\n", "\n").replace("\r", "\n"))
        text = "".join(output).strip()
        detail = f"\n{text}" if text else ""
        return f"Exit code: {process.returncode}{detail}"


DEFAULT_SHELL_RUNNER = SubprocessShellRunner()
