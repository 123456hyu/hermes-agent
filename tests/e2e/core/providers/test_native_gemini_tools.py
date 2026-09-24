"""Gemini native wire conformance: the tool loop and thought-signature replay across ``--resume``.

Real ``hermes chat -q`` subprocesses talk to Google AI Studio's native ``streamGenerateContent``
dialect; only the vendor is faked (``tests/fakes/providers/gemini_native.py``, a TLS-terminating
loopback proxy that validates every request like Google and rejects it with a 400 when it would).

Turn 1 (process A): the model calls ``read_file`` on a seeded file, then answers.
Turn 2 (process B, ``--resume``): the model calls ``terminal``, then answers.
Gemini 3 needs the ``thoughtSignature`` of every functionCall step sent back verbatim; the fake
enforces it for the current turn and the tests assert it for the resumed (previous-turn) history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from tests.e2e.core.providers import _native_helpers as nh
from tests.fakes.providers.gemini_native import HERMES_ENV, Call, Calls, GeminiFake, Recorded, Text, hermes_model

SEED = "GEMINI-SEED-CANARY-5521"
ECHO = "GEMINI-RESUME-ECHO-8813"
ANSWER_1 = "Turn one answer GEMINI-ANS-ONE"
ANSWER_2 = "Turn two answer GEMINI-ANS-TWO"


@dataclass
class Run:
    home: nh.NativeHome
    fake: GeminiFake
    turns: list[nh.ChatResult]
    session_id: str

    def main(self) -> list[Recorded]:
        return self.fake.main_calls()


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    root = tmp_path_factory.mktemp("gemini_tools")
    home = nh.make_home(root, hermes_model(), env_file=HERMES_ENV)
    seed = home.project / "seed.txt"
    seed.write_text(f"{SEED}\n", encoding="utf-8")
    script = [
        Calls([Call("read_file", {"path": str(seed)})]),
        Text(ANSWER_1, thought="Summarising the file."),
        Calls([Call("terminal", {"command": f"echo {ECHO}"})]),
        Text(ANSWER_2),
    ]
    with GeminiFake(root / "fake", script) as fake:
        first = nh.run_chat(home, "Read seed.txt and tell me what it says.", env=fake.child_env())
        assert first.returncode == 0, first.describe()
        sid = nh.latest_session(home)
        second = nh.run_chat(home, "Now echo the marker in the terminal.", env=fake.child_env(), resume=sid)
        assert second.returncode == 0, second.describe()
    return Run(home, fake, [first, second], sid)


def _call_parts(rec: Recorded) -> dict[str, dict]:
    """functionCall id -> the whole Part (so the sibling ``thoughtSignature`` is visible)."""
    return {p["functionCall"].get("id"): p for p in rec.parts("functionCall")}


def _responses(rec: Recorded) -> dict[str, dict]:
    return {p["functionResponse"].get("id"): p["functionResponse"] for p in rec.parts("functionResponse")}


def test_function_call_round_trip_pairs_response_and_persists(run: Run) -> None:
    """a. functionCall -> real tool -> functionResponse (same name + id) -> final answer printed."""
    assert run.fake.rejections() == [], run.fake.rejections()
    main = run.main()
    assert len(main) == 4, [(r.reply, r.status) for r in main]
    assert all(r.stream and r.query.get("alt") == ["sse"] for r in main)
    (call_id,) = list(run.fake.call_signatures)[:1]
    follow_up = main[1]
    resp = _responses(follow_up).get(call_id)
    assert resp is not None, f"no functionResponse for {call_id}: {follow_up.parts('functionResponse')}"
    assert resp["name"] == "read_file"
    assert SEED in json.dumps(resp["response"]), resp
    part = _call_parts(follow_up)[call_id]
    assert part.get("thoughtSignature") == run.fake.call_signatures[call_id], part
    assert ANSWER_1 in run.turns[0].stdout, run.turns[0].describe()

    rows = nh.messages(run.home, run.session_id)
    ids = [tc.get("id") for r in rows if r["role"] == "assistant" for tc in nh.tool_calls_of(r)]
    assert call_id in ids, ids
    tool_rows = [r for r in rows if r["role"] == "tool" and r.get("tool_call_id") == call_id]
    assert len(tool_rows) == 1 and SEED in (tool_rows[0].get("content") or ""), tool_rows
    nh.assert_no_duplicate_assistant_text(rows, ANSWER_1)


def test_resume_replays_thought_signatures_verbatim(run: Run) -> None:
    """b. After ``--resume`` in a new process, turn 1's functionCall goes back with the exact
    signature Google issued (not dropped, not a skip-validator dummy), still paired to its result;
    the resumed turn's own call is signed too (the fake 400s a missing current-turn signature)."""
    assert run.fake.rejections() == [], run.fake.rejections()
    first_id, second_id = list(run.fake.call_signatures)
    resumed = run.main()[2]
    replayed = _call_parts(resumed).get(first_id)
    assert replayed is not None, f"turn-1 functionCall {first_id} missing after resume: {resumed.contents}"
    assert replayed.get("thoughtSignature") == run.fake.call_signatures[first_id], replayed
    assert SEED in json.dumps(_responses(resumed).get(first_id)), resumed.parts("functionResponse")
    assert ANSWER_1 in resumed.all_text()

    final = run.main()[3]
    assert _call_parts(final)[second_id].get("thoughtSignature") == run.fake.call_signatures[second_id]
    assert ECHO in json.dumps(_responses(final).get(second_id))
    assert ANSWER_2 in run.turns[1].stdout, run.turns[1].describe()
    nh.assert_no_duplicate_assistant_text(nh.messages(run.home), ANSWER_2)


def test_no_egress_beyond_the_google_host(run: Run) -> None:
    """Every generate call carried the API key; no other request reached a Google path."""
    for rec in run.fake.generate_calls():
        assert rec.headers.get("x-goog-api-key") or rec.query.get("key"), rec.headers
    other = [r.path for r in run.fake.requests if not r.rpc]
    assert other == [], other
