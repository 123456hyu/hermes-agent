"""Worktrees never hang off the running gateway's checkout.

``resolve_worktree_repo_root`` is the one seam every ``hermes -w`` / ``/worktree new`` / TUI
``--worktree`` path goes through. A worktree rooted in the install checkout shares its ``.git``
and lanes inherit it as cwd, so one ``git checkout``/``merge`` there swaps the live gateway's code.
"""

import subprocess

import pytest

from hermes_cli import worktree_ops


def _git_repo(path):
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    return path


@pytest.fixture
def install_and_dev(tmp_path, monkeypatch):
    install = _git_repo(tmp_path / "install")
    dev = _git_repo(tmp_path / "dev")
    monkeypatch.setattr(worktree_ops, "_install_checkout_root", lambda: install.resolve())
    monkeypatch.setattr(worktree_ops, "_cprint", lambda text: print(text))
    return install, dev


def test_install_checkout_refused_only_while_gateway_runs(install_and_dev, monkeypatch, capsys):
    install, _dev = install_and_dev
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})

    monkeypatch.setattr(worktree_ops, "_gateway_runs_from", lambda checkout: True)
    assert worktree_ops.resolve_worktree_repo_root(str(install)) is None
    out = capsys.readouterr().out
    assert "running gateway's checkout" in out and "worktree_repo_root" in out

    monkeypatch.setattr(worktree_ops, "_gateway_runs_from", lambda checkout: False)
    assert worktree_ops.resolve_worktree_repo_root(str(install)) == str(install)


def test_configured_dev_clone_wins_over_cwd(install_and_dev, monkeypatch):
    install, dev = install_and_dev
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"worktree_repo_root": str(dev)})
    monkeypatch.setattr(worktree_ops, "_gateway_runs_from", lambda checkout: True)
    # cwd (and the explicit arg) point at the live install; config redirects to the dev clone.
    assert worktree_ops.resolve_worktree_repo_root(str(install)) == str(dev.resolve())
    monkeypatch.chdir(install)
    assert worktree_ops.resolve_worktree_repo_root() == str(dev.resolve())
