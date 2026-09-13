"""Opt-in, real Linux bind-mount regression; no model or benchmark execution.

Run as a non-root Linux user with Docker access and an already local image:
TINYHARNESS_DOCKER_TEST_IMAGE=<image> python -m unittest discover -s tests \
    -p test_docker_workspace_permissions.py -v
The image must contain bash and git. No images are pulled.
"""

import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from uuid import uuid4

from evals.swe_bench_lite.docker_workspace import DockerShellRunner
from tiny_harness.tools.filesystem import edit_file


IMAGE = os.environ.get("TINYHARNESS_DOCKER_TEST_IMAGE")


@unittest.skipUnless(
    IMAGE and os.name == "posix" and os.geteuid() != 0,
    "Requires opt-in local Docker image and non-root Linux host",
)
class DockerWorkspacePermissionsTest(unittest.TestCase):
    def docker(self, *args):
        result = subprocess.run(
            ["docker", *args], check=False, capture_output=True,
            text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.strip()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tinyharness-permissions-")
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.container = "tinyharness-permissions-" + uuid4().hex[:12]
        self.addCleanup(self.cleanup_container)
        self.docker(
            "run", "--pull=never", "--detach", "--network", "none",
            "--user", "0:0", "--name", self.container,
            "--mount", f"type=bind,source={self.workspace},target=/testbed",
            "--workdir", "/testbed", "--entrypoint", "/bin/bash", IMAGE,
            "-lc", "while true; do sleep 3600; done",
        )
        self.runner = DockerShellRunner(self.container, self.workspace)
        (self.workspace / "tracked.txt").write_text("base\n")
        self.raw(
            "git config --global --add safe.directory /testbed && "
            "git init -q && git config user.name Test && "
            "git config user.email test@example.invalid && "
            "git config core.fileMode false && git add tracked.txt && "
            "git commit -qm base && chmod -R a+rwX /testbed"
        )
        print(f"host={os.geteuid()}:{os.getegid()} container={self.raw('id -u; id -g')}"
              f" initial={self.metadata('tracked.txt')}")

    def cleanup_container(self):
        try:
            self.raw(
                f"chmod -R a+rwX /testbed && "
                f"chown -R -h {os.geteuid()}:{os.getegid()} /testbed"
            )
        finally:
            self.docker("rm", "--force", self.container)

    def raw(self, command):
        return self.docker(
            "exec", "--workdir", "/testbed", self.container,
            "/bin/bash", "-lc", command,
        )

    def fixed(self, command):
        result = self.runner.run(self.workspace, command)
        self.assertTrue(result.startswith("Exit code: 0"), result)

    def metadata(self, path):
        info = (self.workspace / path).stat()
        return info.st_uid, info.st_gid, oct(stat.S_IMODE(info.st_mode))

    def assert_editable(self, path, text):
        self.assertEqual(self.metadata(path), (0, 0, "0o666"))
        self.assertEqual(edit_file(self.workspace, path, text, "host edit"), f"Edited {path}")
        self.assertIn("host edit", (self.workspace / path).read_text())

    def test_old_umask_reproduces_checkout_stash_reset_permission_error(self):
        for command, expected in (
            ("git checkout -- tracked.txt", "base"),
            ("git stash push -q && git stash pop -q", "changed"),
            ("git reset --hard HEAD", "base"),
        ):
            with self.subTest(command=command):
                self.raw("chmod -R a+rwX /testbed")
                (self.workspace / "tracked.txt").write_text("changed\n")
                self.raw("umask 022\n" + command)
                print(f"before fix: {command}: {self.metadata('tracked.txt')}")
                self.assertEqual(self.metadata("tracked.txt"), (0, 0, "0o644"))
                with self.assertRaises(PermissionError):
                    edit_file(self.workspace, "tracked.txt", expected, "host edit")

    def test_checkout_then_host_edit(self):
        (self.workspace / "tracked.txt").write_text("changed\n")
        self.fixed("git checkout -- tracked.txt")
        self.assert_editable("tracked.txt", "base")

    def test_stash_and_pop_then_host_edit(self):
        (self.workspace / "tracked.txt").write_text("changed\n")
        self.fixed("git stash push -q")
        self.assert_editable("tracked.txt", "base")
        self.fixed("git checkout -- tracked.txt")
        self.fixed("git stash pop -q")
        self.assert_editable("tracked.txt", "changed")

    def test_reset_then_host_edit(self):
        (self.workspace / "tracked.txt").write_text("changed\n")
        self.fixed("git reset --hard HEAD")
        self.assert_editable("tracked.txt", "base")

    def test_container_new_file_and_directory_then_host_edit(self):
        self.fixed("mkdir created && printf 'container text\\n' > created/new.txt")
        self.assertEqual(self.metadata("created"), (0, 0, "0o777"))
        self.assert_editable("created/new.txt", "container text")
        (self.workspace / "created" / "host.txt").write_text("host created")


if __name__ == "__main__":
    unittest.main()
