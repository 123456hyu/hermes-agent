import asyncio

import hermes_cli.web_models as _web_models
import hermes_cli.web_routers.sessions as _rt_sessions
import hermes_cli.web_server_sessions as _web_server_sessions


def test_session_rename_routes_through_authority_without_second_writer(monkeypatch):
    """Main's rename fix offloaded the ad-hoc ``_with_db`` writer; on the unified runtime the
    PATCH route never opens a SessionDB at all — the owning session authority applies the
    ``sidebar`` mutation, so a second writer (on or off the loop) would be the regression."""
    seen = {}

    def _open_db(*_args, **_kwargs):
        raise AssertionError("rename must not open its own SessionDB writer")

    async def _mutate(request, profile, session_id, *, request_id, expected_revision, operation,
                      payload, expected_generation=None):
        seen.update(profile=profile, session_id=session_id, request_id=request_id,
                    expected_revision=expected_revision, operation=operation, payload=payload,
                    expected_generation=expected_generation)
        return {"ok": True, "title": payload.get("title")}

    monkeypatch.setattr(_web_server_sessions, "_open_session_db_for_profile", _open_db)
    monkeypatch.setattr(_web_server_sessions, "_mutate_session_request", _mutate)

    body = _web_models.SessionRename(title="renamed", pinned=True, request_id="req-1",
                                     expected_revision=3, expected_generation=1)
    result = asyncio.run(_rt_sessions.rename_session_endpoint("sess-1", body, request=object()))

    assert result == {"ok": True, "title": "renamed"}
    assert seen == {
        "profile": None, "session_id": "sess-1", "request_id": "req-1", "expected_revision": 3,
        "expected_generation": 1, "operation": "sidebar", "payload": {"title": "renamed", "pinned": True},
    }
