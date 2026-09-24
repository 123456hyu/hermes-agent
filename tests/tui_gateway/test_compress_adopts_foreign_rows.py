"""Regression for #120155: the Desktop/TUI ``/compress`` must compact what is in ``state.db``, not the history
the surface holds in memory. Turns another surface (Telegram after ``/handoff``, cron) appended since the
surface last synced were archived by the in-place commit without the summarizer ever seeing them, and were
gone from every surface afterwards (``active=0, compacted=0`` is the superseded-duplicate flag)."""

import contextlib
import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from tui_gateway import server


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    yield db
    db.close()


def _exchanges(n):
    history = []
    for i in range(n):
        history.append({"role": "user", "content": f"question {i} about fruit{i} " + " ".join(["filler"] * 40)})
        history.append({"role": "assistant", "content": f"answer {i} " + " ".join(["lorem"] * 400)})
    return history


def _live_session(db, monkeypatch):
    """A real AIAgent (default in-place mode) and the TUI session dict a cold resume leaves: history stamped
    with its durable row ids, so the surface knows exactly which rows it holds."""
    db.create_session("sid", "desktop", model="test/model")
    for message in _exchanges(10):
        db.append_message("sid", message["role"], message["content"])
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, session_db=db, session_id="sid", skip_context_files=True, skip_memory=True)
    agent._compression_feasibility_checked = True
    history = db.get_messages_as_conversation("sid", include_row_ids=True)
    session = {"agent": agent, "history": history, "history_lock": threading.Lock(), "history_version": 0,
               "running": False, "session_key": "sid", "cwd": os.getcwd()}

    @contextlib.contextmanager
    def _owner_db(_session):
        yield db
    monkeypatch.setattr(server, "_session_db", _owner_db)
    for name in ("_status_update", "_emit"):
        monkeypatch.setattr(server, name, lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_session_info", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    return agent, session


def _summary_llm(seen_prompts):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "## Goal\nNumbered fruit questions.\n## Progress\nEarly ones answered."

    def _call(**kwargs):
        seen_prompts.append(kwargs.get("messages"))
        return response
    return _call


def _flags(db, content):
    rows = db._conn.execute("SELECT active, compacted FROM messages WHERE session_id = 'sid' AND content = ?",
                            (content,)).fetchall()
    return sorted(tuple(row) for row in rows)


@pytest.mark.parametrize("focus", ["", "here 2"])
def test_session_compress_keeps_turns_another_surface_appended(session_db, monkeypatch, focus):
    agent, session = _live_session(session_db, monkeypatch)
    # A Telegram turn lands on the same session while the desktop is idle (the desktop repaints it from the
    # DB, but its in-memory history never held it).
    foreign_user = "My codeword is MANGO."
    foreign_reply = "Noted: MANGO."
    session_db.append_message("sid", "user", foreign_user)
    session_db.append_message("sid", "assistant", foreign_reply)
    assert not any(m["content"] == foreign_user for m in session["history"])

    spy = MagicMock(wraps=agent._compress_context)
    with patch("agent.context_compressor.call_llm", _summary_llm([])), patch.object(agent, "_compress_context", spy):
        resp = server._compress_live("rid", "sid", session, focus)
    assert resp["result"]["status"] == "compressed" and agent.session_id == "sid"

    # The compactor was handed the foreign turn (head or kept tail): it is part of the compaction input…
    handed = [*spy.call_args.args[0], *(spy.call_args.kwargs.get("verbatim_tail") or ())]
    assert [m.get("content") for m in handed].count(foreign_user) == 1
    # …and it is still a live turn after the compacted set, once, on every read path.
    durable = session_db.get_messages_as_conversation("sid")
    assert [m["content"] for m in durable].count(foreign_user) == 1
    assert [m["content"] for m in durable].count(foreign_reply) == 1
    assert _flags(session_db, foreign_user) != [(0, 0)]
    assert _flags(session_db, foreign_reply) != [(0, 0)]
    model_history, display_history = session_db.get_resume_conversations("sid")
    assert [m["content"] for m in model_history].count(foreign_user) == 1
    assert [m["content"] for m in display_history].count(foreign_user) == 1
    # The surface's own history agrees with the DB.
    assert [m["content"] for m in session["history"]] == [m["content"] for m in durable]
