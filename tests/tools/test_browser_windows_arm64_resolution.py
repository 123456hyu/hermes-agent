"""Windows ARM64 agent-browser resolution.

The npm JS wrapper spawns either a 0-byte ``agent-browser-win32-arm64.exe`` or the
published x64 image from an arm64 Node process and fails with ``spawn EFTYPE``.
Resolution must prefer a real native ARM64 PE so that spawn never uses that x64
stub, heal an empty arm64 stub when only the published x64 image exists, and
doctor must not call the npx wrapper healthy on this host.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.browser_tool as bt
from tools.browser_tool_install import _find_agent_browser


_ARM64_MACHINE = 0xAA64
_X64_MACHINE = 0x8664


@pytest.fixture(autouse=True)
def _clear_browser_caches():
    bt._cached_agent_browser = None
    bt._agent_browser_resolved = False
    yield
    bt._cached_agent_browser = None
    bt._agent_browser_resolved = False


def _windows_arm64(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("tools.browser_tool_install.sys.platform", "win32")
    monkeypatch.setattr("hermes_platform.host.facts.native_arch", lambda: "arm64")
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("npm_config_cache", raising=False)
    monkeypatch.delenv("NPM_CONFIG_CACHE", raising=False)


def _write_pe(path: Path, *, machine: int, size: int = 4096) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = bytearray(size)
    blob[0:2] = b"MZ"
    e_lfanew = 0x80
    blob[0x3C:0x40] = e_lfanew.to_bytes(4, "little")
    blob[e_lfanew:e_lfanew + 4] = b"PE\0\0"
    blob[e_lfanew + 4:e_lfanew + 6] = machine.to_bytes(2, "little")
    path.write_bytes(blob)
    path.chmod(0o755)
    return path


def test_windows_arm64_prefers_native_arm64_over_runnable_shim_and_x64(tmp_path, monkeypatch):
    """A runnable .cmd shim must not win when a real ARM64 PE sits beside the x64 image."""
    _windows_arm64(monkeypatch)
    pkg_bin = tmp_path / "node_modules" / "agent-browser" / "bin"
    arm64 = _write_pe(pkg_bin / "agent-browser-win32-arm64.exe", machine=_ARM64_MACHINE)
    x64 = _write_pe(pkg_bin / "agent-browser-win32-x64.exe", machine=_X64_MACHINE)
    shim = tmp_path / "node_modules" / ".bin" / "agent-browser.cmd"
    shim.parent.mkdir(parents=True)
    shim.write_text("@echo off\r\n")
    shim.chmod(0o755)
    monkeypatch.setattr("tools.browser_tool_install.get_hermes_home", lambda: tmp_path / "empty-home")
    (tmp_path / "empty-home").mkdir()

    def which(cmd, path=None):
        if cmd == "agent-browser":
            return str(shim)
        if cmd == "npx":
            return "/usr/bin/npx"
        return None

    with patch("tools.browser_tool_install.shutil.which", side_effect=which), \
         patch("tools.browser_tool_install.agent_browser_runnable", return_value=True), \
         patch("tools.browser_tool_install._merge_browser_path", return_value=""), \
         patch("tools.browser_tool_install._discover_homebrew_node_dirs", return_value=()):
        resolved = _find_agent_browser()

    assert resolved == str(arm64)
    assert resolved != str(x64)
    assert "npx" not in resolved
    assert not resolved.lower().endswith(".cmd")


def test_windows_arm64_empty_stub_is_healed_and_spawn_skips_npx_wrapper(tmp_path, monkeypatch):
    """No native ARM64 PE: do not return the npx wrapper, heal the 0-byte stub, spawn the published exe directly."""
    _windows_arm64(monkeypatch)
    home = tmp_path / "hermes"
    pkg_bin = home / "node_modules" / "agent-browser" / "bin"
    x64 = _write_pe(pkg_bin / "agent-browser-win32-x64.exe", machine=_X64_MACHINE)
    arm64 = pkg_bin / "agent-browser-win32-arm64.exe"
    arm64.write_bytes(b"")
    monkeypatch.setattr("tools.browser_tool_install.get_hermes_home", lambda: home)

    def which(cmd, path=None):
        if cmd == "npx":
            return "/usr/bin/npx"
        return None

    with patch("tools.browser_tool_install.shutil.which", side_effect=which), \
         patch("tools.browser_tool_install.node_tool_runnable", return_value=True), \
         patch("tools.browser_tool_install.agent_browser_runnable", return_value=True), \
         patch("tools.browser_tool_install._merge_browser_path", return_value=""), \
         patch("tools.browser_tool_install._discover_homebrew_node_dirs", return_value=()):
        resolved = _find_agent_browser()

    assert resolved == str(x64)
    assert resolved != str(arm64)
    assert resolved != "npx agent-browser"
    assert arm64.stat().st_size >= 1024
    assert arm64.read_bytes() == x64.read_bytes()


def test_doctor_does_not_call_npx_wrapper_healthy_on_windows_arm64(monkeypatch, capsys):
    """Doctor's runnable probe must not treat the npx wrapper as installed on Windows ARM64."""
    _windows_arm64(monkeypatch)
    monkeypatch.setattr(
        "tools.browser_tool_install._find_agent_browser",
        lambda **_kw: "npx agent-browser",
    )
    from hermes_cli.doctor_tools import _check_agent_browser

    assert _check_agent_browser(False) is False
    out = capsys.readouterr().out
    assert "resolves via npx on first use" not in out
    assert "Windows ARM64" in out
