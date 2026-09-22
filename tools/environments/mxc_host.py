"""Host-side support for the Windows MXC terminal backend.

Everything here runs on the Hermes host, outside any sandbox: locating ``wxc-exec.exe``
(Microsoft's MXC launcher), probing what the OS can enforce, provisioning the POSIX shell
the sandbox runs (busybox-w32; Git-Bash's MSYS runtime cannot initialize inside an
AppContainer), reading the sandbox policy from config, and summarizing all of it as one
status record that the CLI, ``hermes doctor`` and the desktop share.

``tools.environments.mxc`` (the environment class) consumes these helpers; nothing in
this module depends on ``BaseEnvironment``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

# Well-known install locations for the MXC kit, in precedence order.
WXC_EXEC_CANDIDATES = (
    r"C:\mxc-kit\bin\wxc-exec.exe",
    r"C:\mxc\wxc-exec.exe",
)

# busybox-w32 (GPLv2, https://frippery.org/busybox/) is the in-sandbox POSIX shell. Pinned to a
# specific release and checksum so provisioning cannot silently pick up a different binary.
BUSYBOX_RELEASE = "FRP-6075-g169694ebd"
BUSYBOX_BASE_URL = "https://frippery.org/files/busybox/"
BUSYBOX_BUILDS = {
    # machine -> (file name on the release server, sha256)
    "arm64": (f"busybox-w64a-{BUSYBOX_RELEASE}.exe",
              "e67f873d19d58c535cc9f0c4965ffd622e19b7bab87e3da89cb2185fb54464d7"),
    "amd64": (f"busybox-w64u-{BUSYBOX_RELEASE}.exe",
              "6e263d154d8548d1eb936f65d1d8312c80df31c45974e48d6335e4dcc0f4f34c"),
}
BUSYBOX_LOCAL_NAME = "busybox-sh.exe"

# Probing is cheap, but every status read would otherwise spawn a process; verdicts change only
# when the host does (kit installed, host prep run), so a short cache is safe.
_PROBE_TTL_SECONDS = 60.0
_probe_lock = threading.Lock()
_probe_cache: dict[str, tuple[float, dict]] = {}

# One-shot containers launched by this process; the desktop shows it as a running tally.
_container_counter_lock = threading.Lock()
_containers_started = 0


def note_container_started() -> int:
    global _containers_started
    with _container_counter_lock:
        _containers_started += 1
        return _containers_started


def containers_started() -> int:
    with _container_counter_lock:
        return _containers_started


# ── configuration ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MxcPolicy:
    """What the sandbox may touch beyond the task's own working directory."""
    readwrite_paths: tuple[str, ...] = ()
    readonly_paths: tuple[str, ...] = ()
    network: bool = False

    def as_dict(self) -> dict:
        return {"readwrite_paths": list(self.readwrite_paths),
                "readonly_paths": list(self.readonly_paths),
                "network": self.network}


@dataclass(frozen=True)
class MxcSettings:
    wxc_exec_path: Optional[str]
    shell_path: Optional[str]
    policy: MxcPolicy
    debug: bool = False
    raw: dict = field(default_factory=dict)


def _terminal_section() -> dict:
    """The same live, strict authority used by terminal execution."""
    from tools.terminal_scope import get_live_terminal_config
    return get_live_terminal_config()


def resolve_settings(terminal_cfg: Optional[dict] = None) -> MxcSettings:
    """Live profile settings, or an explicitly supplied authoritative configuration."""
    cfg = terminal_cfg if terminal_cfg is not None else _terminal_section()
    if not isinstance(cfg, dict):
        raise ValueError("Terminal policy must be a mapping")
    for key in ("mxc_network", "mxc_debug"):
        if key in cfg and not isinstance(cfg[key], bool):
            raise ValueError(f"terminal.{key} must be a boolean")
    for key in ("mxc_wxc_exec_path", "mxc_shell_path"):
        if cfg.get(key) is not None and not isinstance(cfg[key], str):
            raise ValueError(f"terminal.{key} must be a path string")
    wxc = (cfg.get("mxc_wxc_exec_path") or "").strip() or None
    shell = (cfg.get("mxc_shell_path") or "").strip() or None
    policy = MxcPolicy(
        readwrite_paths=tuple(validate_grant_paths(cfg.get("mxc_readwrite_paths", []), writable=True)),
        readonly_paths=tuple(validate_grant_paths(cfg.get("mxc_readonly_paths", []), writable=False)),
        network=cfg.get("mxc_network", False))
    return MxcSettings(wxc_exec_path=wxc, shell_path=shell, policy=policy,
                       debug=cfg.get("mxc_debug", False), raw=dict(cfg))


# ── wxc-exec discovery and probe ─────────────────────────────────────────────

def find_wxc_exec(configured: Optional[str] = None) -> Optional[str]:
    """Absolute path of ``wxc-exec.exe`` or None. Order: explicit config, PATH, well-known dirs."""
    if configured:
        candidate = os.path.expandvars(os.path.expanduser(configured))
        return candidate if os.path.isfile(candidate) else None
    found = shutil.which("wxc-exec.exe") or shutil.which("wxc-exec")
    if found:
        return found
    for candidate in WXC_EXEC_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


def run_probe(wxc_exec: str, *, timeout: float = 15.0) -> dict:
    """``wxc-exec --probe`` decoded, cached for a short TTL per binary path.

    Returns ``{"ok": bool, "tier": str|None, "warnings": [...], "probes": {...}, "error": str|None}``.
    """
    now = time.monotonic()
    with _probe_lock:
        cached = _probe_cache.get(wxc_exec)
        if cached and now - cached[0] < _PROBE_TTL_SECONDS:
            return dict(cached[1])
    result = _run_probe_uncached(wxc_exec, timeout=timeout)
    with _probe_lock:
        _probe_cache[wxc_exec] = (now, result)
    return dict(result)


def _run_probe_uncached(wxc_exec: str, *, timeout: float) -> dict:
    try:
        completed = subprocess.run(
            [wxc_exec, "--probe"], capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace", **_hidden_window_kwargs())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "tier": None, "warnings": [], "probes": {}, "error": f"could not run wxc-exec --probe: {exc}"}
    text = (completed.stdout or "").strip()
    start = text.find("{")
    if completed.returncode != 0 or start == -1:
        detail = (completed.stderr or text or f"exit {completed.returncode}").strip()
        return {"ok": False, "tier": None, "warnings": [], "probes": {}, "error": detail[:500]}
    try:
        data = json.loads(text[start:])
    except ValueError as exc:
        return {"ok": False, "tier": None, "warnings": [], "probes": {}, "error": f"unreadable probe output: {exc}"}
    tier = data.get("tier")
    probes = data.get("probes") or {}
    result = {"ok": True, "tier": tier, "warnings": list(data.get("warnings") or []), "probes": probes, "error": None}
    result["error"] = strict_probe_reason(result)
    result["ok"] = result["error"] is None
    return result


_REQUIRED_UI_CAPABILITIES = (
    "canBlockClipboardRead", "canBlockClipboardWrite", "canBlockInputInjection",
    "canBlockInputMethodChanges", "canBlockExternalUiObjects", "canBlockGlobalUiNamespace",
    "canBlockDesktopSwitching", "canBlockLogoffOrShutdown",
    "canBlockSystemParameterChanges", "canBlockDisplaySettingsChanges",
)


def strict_probe_reason(probe: dict) -> Optional[str]:
    """A fallback tier is not the strict filesystem/UI boundary Hermes advertises."""
    if not probe.get("ok"):
        return probe.get("error") or "MXC could not verify this host's isolation capabilities."
    facts = probe.get("probes") or {}
    if probe.get("tier") != "base-container" or facts.get("baseContainerApiPresent") is not True:
        return "MXC requires the base-container tier; filesystem/DACL fallback is disabled."
    ui = facts.get("uiCapabilities") or {}
    missing = [name for name in _REQUIRED_UI_CAPABILITIES if ui.get(name) is not True]
    if missing:
        return "MXC cannot enforce the required UI isolation: " + ", ".join(missing)
    return None


def clear_probe_cache() -> None:
    with _probe_lock:
        _probe_cache.clear()


# ── shell provisioning ───────────────────────────────────────────────────────

def _machine_key() -> Optional[str]:
    machine = (platform.machine() or "").lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    if machine in ("amd64", "x86_64"):
        return "amd64"
    return None


def _host_machine_key() -> Optional[str]:
    """The OS architecture, not the interpreter's: Hermes may run as an emulated x64 process on
    an ARM64 host, and the sandbox shell must match the OS."""
    if _IS_WINDOWS:
        try:
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            process_machine = ctypes.c_ushort()
            native_machine = ctypes.c_ushort()
            handle = ctypes.c_void_p(kernel32.GetCurrentProcess())
            if kernel32.IsWow64Process2(handle, ctypes.byref(process_machine), ctypes.byref(native_machine)):
                return {0xAA64: "arm64", 0x8664: "amd64"}.get(native_machine.value)
        except Exception:
            logger.debug("mxc: IsWow64Process2 unavailable", exc_info=True)
    return _machine_key()


def default_shell_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "bin" / BUSYBOX_LOCAL_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_shell(configured: Optional[str] = None, *, download: bool = True) -> tuple[Optional[str], Optional[str]]:
    """``(shell_path, error)``: the POSIX shell binary for the sandbox.

    A configured path is used as-is. Otherwise the pinned busybox-w32 build for the host
    architecture is expected under ``$HERMES_HOME/bin`` and downloaded there (checksum
    verified, written atomically) when missing and *download* is allowed.
    """
    if configured:
        candidate = os.path.expandvars(os.path.expanduser(configured))
        if os.path.isfile(candidate):
            return candidate, None
        return None, f"terminal.mxc_shell_path does not exist: {configured}"
    target = default_shell_path()
    if target.is_file():
        return str(target), None
    key = _host_machine_key()
    if key not in BUSYBOX_BUILDS:
        return None, f"no pinned busybox-w32 build for this architecture ({platform.machine() or 'unknown'}); set terminal.mxc_shell_path"
    if not download:
        return None, "sandbox shell (busybox-w32) is not installed yet"
    name, expected = BUSYBOX_BUILDS[key]
    url = BUSYBOX_BASE_URL + name
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".download")
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:  # noqa: S310 - pinned https URL
            shutil.copyfileobj(response, out)
        actual = _sha256(tmp)
        if actual != expected:
            tmp.unlink(missing_ok=True)
            return None, f"downloaded {name} failed checksum verification"
        os.replace(tmp, target)
    except Exception as exc:  # network down, disk full, ...
        return None, f"could not download the sandbox shell ({name}): {exc}"
    logger.info("mxc: installed sandbox shell %s -> %s", name, target)
    return str(target), None


# ── workspace ancestors ──────────────────────────────────────────────────────
#
# Ancestor ACL preparation is retired. Keep a refusal for callers of the old API;
# Hermes must not alter global AppContainer access as a command-side workaround.


def workspace_ancestors(path: str) -> list[str]:
    """Ancestor directories of *path* from the drive root down, excluding *path* itself."""
    current = os.path.normpath(os.path.abspath(path))
    ancestors: list[str] = []
    while True:
        parent = os.path.dirname(current)
        if not parent or parent == current:
            break
        ancestors.append(parent)
        current = parent
    return list(reversed(ancestors))


def ancestor_ready(directory: str) -> Optional[bool]:
    """ACL traversal readiness is no longer inferred from locale-dependent listings."""
    return None


def ancestor_readiness(path: str) -> dict:
    """Compatibility status: unknown, never a claim that host ACLs are prepared."""
    return {"ready": False, "missing": [], "needs_admin": [], "admin_command": "",
            "error": "Ancestor ACL preparation is disabled; traversal readiness is unknown."}


def prepare_ancestors(path: str) -> dict:
    """Retired endpoint: never mutate host ACLs."""
    raise RuntimeError("Ancestor ACL preparation is disabled. See the Windows sandbox documentation for traversal limitations.")


def admin_prepare_command(directories: Iterable[str]) -> str:
    """No administrator command is generated for the retired ACL workflow."""
    return ""


def _canonical_path(path: str, *, strict: bool = True) -> str:
    """Resolve filesystem aliases before comparing or publishing authority."""
    if not isinstance(path, str) or not path.strip() or "\x00" in path:
        raise ValueError("A grant must name an existing absolute path")
    expanded = os.path.expandvars(os.path.expanduser(path.strip()))
    if not os.path.isabs(expanded):
        raise ValueError(f"A grant must be absolute: {path}")
    try:
        resolved = str(Path(expanded).resolve(strict=strict))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Cannot resolve grant path {path}: {exc}") from exc
    # realpath resolves junctions and 8.3 aliases; strip the Win32 extended prefix
    # only afterwards so the identity comparison uses one spelling.
    if resolved.startswith("\\\\?\\UNC\\"):
        resolved = "\\\\" + resolved[8:]
    elif resolved.startswith("\\\\?\\"):
        resolved = resolved[4:]
    return os.path.normpath(resolved)


def _overlaps(left: str, right: str) -> bool:
    left, right = os.path.normcase(left), os.path.normcase(right)
    try:
        return os.path.commonpath([left, right]) in (left, right)
    except ValueError:
        return False


def _trusted_runtime_path(path: str) -> str:
    """Internal exceptions cannot be redirected into another protected subtree."""
    canonical = _canonical_path(path)
    if (os.path.normcase(canonical) != os.path.normcase(os.path.abspath(path))
            and _protected_path_reason(canonical)):
        raise ValueError(f"Internal runtime path is an alias, not an authorized protected exception: {path}")
    return canonical


def _protected_path_reason(path: str) -> Optional[str]:
    home = _canonical_path(os.path.expanduser("~"), strict=False)
    if os.path.dirname(path) == path:
        return f"The sandbox grant would be the drive root ({path})."
    if os.path.normcase(path) == os.path.normcase(home):
        return f"The sandbox grant would be your home folder ({path})."
    install = _canonical_path(str(Path(__file__).resolve().parents[2]))
    if _overlaps(path, install):
        return f"The sandbox grant overlaps Hermes's own program files ({install}). Work in a separate clone."
    for canonical in _protected_data_roots():
        if _overlaps(path, canonical):
            return f"The sandbox grant overlaps Hermes's own data directory ({canonical}), including credentials."
    user_data = os.environ.get("HERMES_DESKTOP_USER_DATA")
    if user_data and _overlaps(path, _canonical_path(user_data, strict=False)):
        return "The sandbox grant overlaps protected desktop data and credentials."
    return None


def _protected_data_roots() -> list[str]:
    from hermes_constants import (get_hermes_home, get_process_hermes_home, get_default_hermes_root,
                                  _get_platform_default_hermes_home)
    return [_canonical_path(str(root), strict=False) for root in
            (get_hermes_home(), get_process_hermes_home(), get_default_hermes_root(),
             _get_platform_default_hermes_home())]


def _shell_runtime_path(shell: str) -> str:
    # A configured shell grants the executable only, never an arbitrary protected
    # descendant disguised as a runtime. The known installed shell is an explicit
    # exception to user-grant validation, checked for reparse redirection as well.
    shell = _trusted_runtime_path(shell)
    known = [os.path.join(root, "bin", BUSYBOX_LOCAL_NAME) for root in _protected_data_roots()]
    if os.path.normcase(shell) in {os.path.normcase(p) for p in known}:
        return shell
    return validate_grant_paths([shell], writable=False)[0]


def validate_grant_paths(paths, *, writable: bool) -> list[str]:
    """Validate user authority, never internal runtime/scratch exceptions.

    Both read and write grants exclude protected roots and their ancestors and
    descendants. Return canonical existing paths; malformed entries fail closed.
    """
    if not isinstance(paths, (list, tuple)):
        raise ValueError("Sandbox grant paths must be a list")
    result: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        path = _canonical_path(raw)
        reason = _protected_path_reason(path)
        if reason:
            raise ValueError(reason)
        key = os.path.normcase(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def unsafe_workspace_reason(workspace: str) -> Optional[str]:
    """Canonical protected overlap check shared by workspace and user grant selection."""
    try:
        return _protected_path_reason(_canonical_path(workspace, strict=False))
    except ValueError as exc:
        return str(exc)


# Sessions that have no project folder would otherwise be anchored at the user's home, which the
# sandbox refuses. They get a dedicated folder inside the profile instead: rooted where the user
# expects their work to live, without granting Documents, AppData or credentials.
DEFAULT_WORKSPACE_DIRNAME = "Hermes"


def default_workspace() -> str:
    """The folder a sandboxed session without a project works in (created on first use)."""
    target = Path(os.path.expanduser("~")) / DEFAULT_WORKSPACE_DIRNAME
    reason = unsafe_workspace_reason(str(target))
    if reason:
        raise ValueError(reason)
    target.mkdir(parents=True, exist_ok=True)
    return validate_grant_paths([str(target)], writable=True)[0]


def sandbox_workspace_for(cwd: str) -> str:
    """*cwd* when it may be a sandbox workspace, else the default workspace. The one rule every
    surface (session creation, the desktop's default folder, the Sandbox panel, the environment
    itself) applies, so they agree on where a sandboxed session works. An empty or relative *cwd*
    means the process's own directory, which is judged as the folder it resolves to."""
    resolved = os.path.abspath(os.path.expanduser(cwd)) if cwd else os.getcwd()
    return _canonical_path(resolved, strict=False) if unsafe_workspace_reason(resolved) is None else default_workspace()


# The desktop stages what the user pastes or attaches in the composer under its Electron user-data
# folder. That content was handed to the agent deliberately, so the sandbox may always read it;
# the desktop tells its spawned backend where that folder is. Only the staging subfolders are
# granted: the user-data folder itself holds connection tokens and browser storage.
DESKTOP_USER_DATA_ENV = "HERMES_DESKTOP_USER_DATA"
ATTACHMENT_STAGING_SUBDIRS = ("composer-images", "composer-pastes")


def attachment_staging_dirs() -> list[str]:
    """Existing composer staging folders of the desktop that spawned this backend (empty otherwise)."""
    user_data = (os.environ.get(DESKTOP_USER_DATA_ENV) or "").strip()
    if not user_data:
        return []
    return [_trusted_runtime_path(str(Path(user_data) / name))
            for name in ATTACHMENT_STAGING_SUBDIRS if (Path(user_data) / name).is_dir()]


# ── status ───────────────────────────────────────────────────────────────────

def _os_build() -> Optional[str]:
    if not _IS_WINDOWS:
        return None
    try:
        v = sys.getwindowsversion()  # type: ignore[attr-defined]
        return f"{v.major}.{v.minor}.{v.build}"
    except Exception:
        return None


def _hidden_window_kwargs() -> dict:
    if not _IS_WINDOWS:
        return {}
    from hermes_cli._subprocess_compat import windows_hide_flags
    return {"creationflags": windows_hide_flags()}


def backend_enabled(terminal_cfg: Optional[dict] = None) -> bool:
    """Whether the strict live terminal authority selects this backend."""
    cfg = terminal_cfg if terminal_cfg is not None else _terminal_section()
    backend = cfg.get("backend")
    return str(backend or "").strip().lower() == "mxc"


OFFLINE_REASON = ("Network is off in the sandbox policy, so Hermes will not fetch URLs on the agent's behalf "
                  "(Hermes desktop: Settings > Safety > Sandbox > Allow network access).")


def status(*, provision_shell: bool = False, settings: Optional[MxcSettings] = None) -> dict:
    """One record describing whether the MXC backend can run here and how it is configured.

    ``available`` means every strict prerequisite holds. Fallback tiers are unavailable,
    never silently degraded. ``reason`` is the first
    blocker in plain language, or None.
    """
    policy_error = None
    if settings is None:
        try:
            settings = resolve_settings()
        except Exception as exc:
            policy_error = f"Sandbox policy unavailable: {exc}"
            settings = MxcSettings(None, None, MxcPolicy())
    record: dict[str, Any] = {
        "platform_supported": _IS_WINDOWS,
        "os_build": _os_build(),
        "enabled": backend_enabled(settings.raw),
        "policy": settings.policy.as_dict(),
        "containers_started": containers_started(),
        "wxc_exec_path": None,
        "probe": None,
        "tier": None,
        "shell_path": None,
        "shell_missing": False,
        "available": False,
        "degraded": False,
        "warnings": [],
        "reason": None,
    }
    if policy_error:
        record["reason"] = policy_error
        return record
    if not _IS_WINDOWS:
        record["reason"] = "MXC sandboxing is a Windows feature; this host is not Windows."
        return record
    wxc = find_wxc_exec(settings.wxc_exec_path)
    record["wxc_exec_path"] = wxc
    if wxc is None:
        hint = settings.wxc_exec_path or ", ".join(WXC_EXEC_CANDIDATES)
        record["reason"] = f"wxc-exec.exe (the MXC kit) was not found at {hint}. Install MXC or set terminal.mxc_wxc_exec_path."
        return record
    probe = run_probe(wxc)
    record["probe"] = probe
    record["tier"] = probe.get("tier")
    reason = strict_probe_reason(probe)
    if reason:
        record["reason"] = reason
        return record
    shell, shell_error = ensure_shell(settings.shell_path, download=provision_shell)
    record["shell_path"] = shell
    if shell is None:
        record["reason"] = shell_error
        record["shell_missing"] = True
        return record
    record["warnings"] = list(probe.get("warnings") or [])

    record["available"] = True
    return record


def unavailable_reason() -> Optional[str]:
    """Plain-language reason the backend cannot run here, or None when it can (shell may still
    need provisioning, which the environment does on first use)."""
    record = status(provision_shell=False)
    if record["available"] or record.get("shell_missing"):
        return None
    return record["reason"]
