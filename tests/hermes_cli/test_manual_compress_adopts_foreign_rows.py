"""Regression for #120155: the CLI ``/compress`` must compact what is in ``state.db``, not the history it holds in
memory. Turns another surface (a Telegram continuation, cron) appended to the same session since the CLI
last synced were archived by the in-place commit without the summarizer seeing them, and were gone from every
surface afterwards (``active=0, compacted=0`` is the superseded-duplicate flag)."""

import os
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI
from hermes_state import SessionDB


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


def _resumed_cli(db):
    """A HermesCLI over a real in-place AIAgent, holding the history a ``--resume`` restores (row ids stamped)."""
    db.create_session("sid", "cli", model="test/model")
    for message in _exchanges(10):
        db.append_message("sid", message["role"], message["content"])
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, session_db=db, session_id="sid", skip_context_files=True, skip_memory=True)
    agent._compression_feasibility_checked = True
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = agent
    cli.session_id = "sid"
    cli._session_db = db
    cli._pending_title = None
    cli.conversation_history, _display = db.get_resume_conversations("sid")
    cli._busy_command = lambda _message, **_kwargs: nullcontext()
    cli._write_terminal_breadcrumb = lambda: None
    return cli, agent


def _summary_llm(**_kwargs):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "## Goal\nNumbered fruit questions.\n## Progress\nEarly ones answered."
    return response


def _flags(db, content):
    rows = db._conn.execute("SELECT active, compacted FROM messages WHERE session_id = 'sid' AND content = ?",
                            (content,)).fetchall()
    return sorted(tuple(row) for row in rows)


@pytest.mark.parametrize("command", ["/compress", "/compress here 2"])
def test_manual_compress_keeps_turns_another_surface_appended(session_db, command, capsys):
    cli, agent = _resumed_cli(session_db)
    # The same session continues on another surface while the CLI sits idle at its prompt.
    foreign_user = "My codeword is MANGO."
    foreign_reply = "Noted: MANGO."
    session_db.append_message("sid", "user", foreign_user)
    session_db.append_message("sid", "assistant", foreign_reply)
    assert not any(m["content"] == foreign_user for m in cli.conversation_history)

    spy = MagicMock(wraps=agent._compress_context)
    with patch("agent.context_compressor.call_llm", _summary_llm), patch.object(agent, "_compress_context", spy):
        cli._manual_compress(command)
    assert "Compression failed" not in capsys.readouterr().out
    assert agent.session_id == "sid" and cli.session_id == "sid"

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
    # The CLI's own history agrees with the DB.
    assert [m["content"] for m in cli.conversation_history] == [m["content"] for m in durable]
