"""Missing required fields recover without enabling general CLI failure halts."""

import json
from unittest.mock import patch

import pytest

from tests.agent.test_tool_call_guardrail_runtime import _make_agent, _mock_response, _mock_tool_call
from tools.file_tools import PATCH_SCHEMA


def _agent(mode):
    agent = _make_agent("patch", "read_file", max_iterations=20)
    agent.tools = [{"type": "function", "function": PATCH_SCHEMA}]
    agent._execute_tool_calls = getattr(agent, f"_execute_tool_calls_{mode}")
    return agent


def _response(args, index):
    return _mock_response(content="", finish_reason="tool_calls", tool_calls=[
        _mock_tool_call("patch", json.dumps(args), f"call_{index}")])


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_incomplete_requests_are_bounded_without_dispatch(mode, tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("keep me\n", encoding="utf-8")
    agent = _agent(mode)
    agent.client.chat.completions.create.side_effect = [
        *[_response({"path": str(target)}, n) for n in range(10)],
        _mock_response(content="done"),
    ]
    with patch("model_tools.handle_function_call") as dispatch:
        result = agent.run_conversation("Edit the owned file.")
    assert dispatch.call_count == 0
    assert agent.client.chat.completions.create.call_count == 3
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert "rejected" in result["final_response"] and "did not execute" in result["final_response"]
    assert target.read_text(encoding="utf-8") == "keep me\n"
    results = [m for m in result["messages"] if m.get("role") == "tool"]
    assert len(results) == 3
    assert all("old_string" in m["content"] and "new_string" in m["content"] for m in results)
    assert {m["tool_call_id"] for m in results} == {f"call_{i}" for i in range(3)}


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_plugin_correction_and_empty_replacements_remain_executable(mode, tmp_path):
    target = str(tmp_path / "target.txt")
    agent = _agent(mode)
    # Two rejected requests, then a plugin-completed valid deletion; repeat to
    # prove landed calls reset the malformed streak without requiring success.
    requests = [_response({"path": target}, i) for i in range(6)]
    requests.append(_mock_response(content="done"))
    agent.client.chat.completions.create.side_effect = requests

    def complete(name, args, **kwargs):
        if kwargs.get("tool_call_id") in {"call_2", "call_5"}:
            return None, {**args, "old_string": "remove", "new_string": "", "replace_all": False}
        return None, None

    with (
        patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=complete),
        patch("model_tools.handle_function_call", return_value=json.dumps({"error": "no matching text"})) as dispatch,
    ):
        result = agent.run_conversation("Delete the requested text.")
    assert dispatch.call_count == 2
    assert all(c.args[1]["new_string"] == "" and c.args[1]["replace_all"] is False for c in dispatch.call_args_list)
    assert agent.client.chat.completions.create.call_count == 7
    assert result["turn_exit_reason"].startswith("text_response")


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
@pytest.mark.parametrize("family", ["base", "openai"])
@pytest.mark.parametrize("shape", ["replace", "v4a"])
def test_patch_replay_contract_survives_either_advertised_schema(mode, family, shape, tmp_path):
    from agent.auxiliary_client import scoped_runtime_main
    from tools.file_tools import _patch_schema_overrides

    target = tmp_path / "replay.txt"
    target.write_text("before\n", encoding="utf-8")
    agent = _agent(mode)
    with scoped_runtime_main({"provider": "openai" if family == "openai" else "custom", "model": "gpt-5" if family == "openai" else "test-model"}):
        schema = {**PATCH_SCHEMA, **_patch_schema_overrides()}
    agent.tools = [{"type": "function", "function": schema}]
    if shape == "v4a":
        args = {"mode": "patch", "patch": f"*** Begin Patch\n*** Update File: {target}\n@@\n-before\n+after\n*** End Patch"}
    else:
        args = {"path": str(target), "old_string": "before", "new_string": "after"}
    agent.client.chat.completions.create.side_effect = [_response(args, 0), _mock_response(content="done")]
    result = agent.run_conversation("Apply the requested edit.")
    assert target.read_text(encoding="utf-8") == "after\n"
    assert result["turn_exit_reason"].startswith("text_response")
    assert "guardrail" not in result
