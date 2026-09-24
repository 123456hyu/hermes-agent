"""Interrupted Windows updates leave node_modules in ENOTEMPTY (#75584).

``npm ci`` tries to ``rmdir`` a nested ``.bin`` that a killed install left
non-empty, then the desktop updater falls through to ``npm install`` against
that same tree and dead-ends. Both commands must delete the tree and retry
once. A non-ENOTEMPTY failure must not wipe the tree.
"""

import io
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from hermes_cli.main_web_build import _run_npm_install_deterministic

# Verbatim shape from the report: npm's own rmdir of a nested .bin.
ENOTEMPTY = (
    "npm error code ENOTEMPTY\n"
    "npm error syscall rmdir\n"
    "npm error path C:\\Users\\user\\AppData\\Local\\hermes\\hermes-agent"
    "\\node_modules\\@babel\\core\\node_modules\\.bin\n"
    "npm error errno -4051\n"
    "npm error ENOTEMPTY: directory not empty, rmdir\n"
)


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "_hermes_home"))


def _corrupt_tree(cwd: Path) -> Path:
    nested = cwd / "node_modules" / "@babel" / "core" / "node_modules" / ".bin"
    nested.mkdir(parents=True)
    (nested / "stale.cmd").write_text("@echo stale\n", encoding="utf-8")
    return cwd / "node_modules"


class _StreamingNpm:
    """Stand-in for the capture_output=False path the desktop updater uses."""

    def __init__(self, stderr: str, returncode: int):
        self.stderr = io.StringIO(stderr)
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False


def _stream(responses):
    """Return a Popen stand-in that replays *(stderr, returncode)* in order."""
    calls: list[list[str]] = []
    present: list[bool] = []

    def popen(cmd, **kwargs):
        calls.append(list(cmd))
        cwd = Path(kwargs["cwd"])
        present.append((cwd / "node_modules").exists())
        stderr, code = responses[len(calls) - 1]
        return _StreamingNpm(stderr, code)

    return popen, calls, present


def test_desktop_npm_ci_enotempty_deletes_tree_and_retries_ci_once(tmp_path):
    """The updater streams npm. ENOTEMPTY must wipe node_modules and retry ci once.

    Falling through to ``npm install`` against the same tree is the dead-end.
    """
    (tmp_path / "package-lock.json").write_text("{}\n", encoding="utf-8")
    tree = _corrupt_tree(tmp_path)
    popen, calls, present = _stream([
        (ENOTEMPTY, 1),
        ("", 0),
    ])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("hermes_cli.main_web_build.subprocess.Popen", popen)
        result = _run_npm_install_deterministic(
            "/usr/bin/npm", tmp_path, capture_output=False,
            extra_args=("--workspace", "web"),
        )

    assert result.returncode == 0
    assert [cmd[1] for cmd in calls] == ["ci", "ci"]
    assert calls[0] == calls[1]
    assert "--workspace" in calls[0] and "web" in calls[0]
    assert present == [True, False]
    assert not tree.exists()


def test_install_fallback_enotempty_deletes_tree_and_retries_once(tmp_path):
    """A non-ENOTEMPTY ``npm ci`` still falls through; the install fallback retries itself.

    The report's ``npm install`` retry failed against the same corrupted tree.
    That fallback must delete ``node_modules`` and run once more, not stop.
    """
    (tmp_path / "package-lock.json").write_text("{}\n", encoding="utf-8")
    tree = _corrupt_tree(tmp_path)
    lock_mismatch = (
        "npm error code EUSAGE\n"
        "npm error `npm ci` can only install packages when your package.json "
        "and package-lock.json are in sync\n"
    )
    popen, calls, present = _stream([
        (lock_mismatch, 1),
        (ENOTEMPTY, 1),
        ("", 0),
    ])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("hermes_cli.main_web_build.subprocess.Popen", popen)
        result = _run_npm_install_deterministic(
            "/usr/bin/npm", tmp_path, capture_output=False,
        )

    assert result.returncode == 0
    assert [cmd[1] for cmd in calls] == ["ci", "install", "install"]
    assert calls[1] == calls[2]
    assert "--no-save" in calls[1]
    assert present == [True, True, False]
    assert not tree.exists()


def test_enotempty_is_retried_once_then_the_failure_is_returned(tmp_path):
    """A second ENOTEMPTY on the same command is the result, not another wipe."""
    (tmp_path / "package-lock.json").write_text("{}\n", encoding="utf-8")
    _corrupt_tree(tmp_path)
    popen, calls, _present = _stream([
        (ENOTEMPTY, 1),
        (ENOTEMPTY, 1),
        (ENOTEMPTY, 1),
        (ENOTEMPTY, 1),
    ])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("hermes_cli.main_web_build.subprocess.Popen", popen)
        result = _run_npm_install_deterministic("/usr/bin/npm", tmp_path, capture_output=False)

    assert result.returncode == 1
    assert "ENOTEMPTY" in (result.stderr or "")
    assert [cmd[1] for cmd in calls] == ["ci", "ci", "install", "install"]


def test_other_ci_failures_leave_the_tree_in_place(tmp_path):
    """Lockfile mismatch must still fall through without deleting node_modules."""
    (tmp_path / "package-lock.json").write_text("{}\n", encoding="utf-8")
    tree = _corrupt_tree(tmp_path)
    marker = tree / "@babel" / "core" / "node_modules" / ".bin" / "stale.cmd"
    popen, calls, present = _stream([
        ("npm error code EUSAGE\n", 1),
        ("", 0),
    ])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("hermes_cli.main_web_build.subprocess.Popen", popen)
        result = _run_npm_install_deterministic("/usr/bin/npm", tmp_path, capture_output=False)

    assert result.returncode == 0
    assert [cmd[1] for cmd in calls] == ["ci", "install"]
    assert present == [True, True]
    assert marker.is_file()
