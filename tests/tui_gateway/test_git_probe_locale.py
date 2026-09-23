"""The Projects git probe decodes git's UTF-8 path output as UTF-8, never the host locale (#53367).

Issue #53367: a Windows desktop Projects list showed the repo ``001回旧`` as ``001鍥炴棫`` —
``git rev-parse --show-toplevel`` prints raw UTF-8 path bytes (``core.quotepath`` does not
apply to it), and ``git_probe.run_git`` decoded them with ``text=True`` alone, i.e. with the
backend's ANSI code page (cp936 on zh-CN Windows). The discovered-repo label is the basename
of that root, so the mojibake went straight to the sidebar.

Real git, real subprocess, real decode: a child interpreter with a non-UTF-8 locale codec
resolves a repo named ``001回旧`` and must report the true name. Under the locale-decoding
probe this yields ``001鍥炴棫`` (GB locale) or ``""`` (ASCII locale: strict decode fails).
"""

import codecs
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_NAME = "001回旧"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_CHILD = """
import json, locale, os, sys
from tui_gateway import git_probe
cwd = sys.argv[1]
print(json.dumps({
    "encoding": locale.getpreferredencoding(False),
    "root": os.path.basename(git_probe.repo_root(cwd)),
    "common": os.path.basename(git_probe.common_repo_root(cwd)),
}, ensure_ascii=True))
"""

# Locale overrides whose codec is not UTF-8, most faithful first: GB-family (what zh-CN
# Windows' cp936 produced), then ASCII. The inherited env covers Windows, where LC_ALL is
# ignored and the ANSI code page (cp1252/cp936/...) is already non-UTF-8.
_LOCALE_OVERRIDES = (
    {"LC_ALL": "zh_CN.GB18030"},
    {"LC_ALL": "zh_CN.GBK"},
    {"LC_ALL": "C", "PYTHONCOERCECLOCALE": "0"},
    {},
)


def _probe_in_child(repo: Path, override: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LC_") and k != "LANG"}
    env.update(override, PYTHONUTF8="0", PYTHONPATH=str(PROJECT_ROOT))
    out = subprocess.run(
        [sys.executable, "-c", _CHILD, str(repo)],
        capture_output=True, env=env, cwd=str(PROJECT_ROOT), timeout=60, check=True,
    ).stdout
    return json.loads(out.decode("ascii"))


@pytest.mark.skipif(shutil.which("git") is None, reason="needs a real git")
def test_repo_root_keeps_utf8_name_under_non_utf8_locale(tmp_path):
    repo = tmp_path / REPO_NAME
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)

    for override in _LOCALE_OVERRIDES:
        result = _probe_in_child(repo, override)
        if codecs.lookup(result["encoding"]).name == "utf-8":
            continue  # this host can't express that locale; try the next
        assert result["root"] == REPO_NAME, result
        assert result["common"] == REPO_NAME, result
        return
    pytest.skip("no non-UTF-8 locale codec available on this host")
