"""Regression coverage for FTS storage upgrade discoverability."""

import sqlite3
from types import SimpleNamespace


def _notice_for(tmp_path, monkeypatch, capsys, *, trigram_has_tool_calls: bool) -> str:
    """Run the post-update notice against a v23 ``messages_fts`` plus a trigram vtable of the
    given shape (layout 1 still carries ``tool_calls``; layout 2+ does not)."""
    from hermes_cli import update_cmd
    import hermes_constants
    import hermes_state

    db_path = tmp_path / "state.db"
    db_path.touch()
    conn = sqlite3.connect(db_path)
    trigram_cols = "content TEXT, tool_name TEXT" + (", tool_calls TEXT" if trigram_has_tool_calls else "")
    conn.executescript(
        f"""
        CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE messages_fts (content TEXT, tool_name TEXT, tool_calls TEXT);
        CREATE TABLE messages_fts_trigram ({trigram_cols});
        """
    )

    class FakeSessionDB:
        def __init__(self, **_kwargs):
            self._conn = conn

        def close(self):
            pass

        _db_has_legacy_inline_fts = staticmethod(hermes_state.SessionDB._db_has_legacy_inline_fts)
        _db_has_trigram_tool_calls_projection = staticmethod(
            hermes_state.SessionDB._db_has_trigram_tool_calls_projection
        )

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(hermes_state, "SessionDB", FakeSessionDB)
    # Report a large state.db without patching Path.stat globally: a
    # 1-arg lambda on the class breaks pathlib.exists(follow_symlinks=...)
    # for every caller in the process (pytest's own teardown included).
    real_stat = update_cmd.Path.stat

    def _stat(path, *args, **kwargs):
        if path.name == "state.db":
            return SimpleNamespace(st_size=512 * 1024 ** 2)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(update_cmd.Path, "stat", _stat)

    update_cmd._print_fts_optimize_available_notice()
    conn.close()
    return capsys.readouterr().out


def test_update_notice_offers_v1_trigram_tool_calls_rebuild(tmp_path, monkeypatch, capsys):
    """A deployed v1 trigram projection still receives the opt-in notice, but as a layout
    change — never the legacy-inline "~60%" estimate, which describes a different migration
    and told users who had already optimized that their work was undone."""
    out = _notice_for(tmp_path, monkeypatch, capsys, trigram_has_tool_calls=True)
    assert "hermes sessions optimize-storage" in out
    assert "layout changed since it was last built" in out
    assert "60%" not in out


def test_update_notice_silent_on_current_layout(tmp_path, monkeypatch, capsys):
    assert _notice_for(tmp_path, monkeypatch, capsys, trigram_has_tool_calls=False) == ""
