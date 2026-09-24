"""Resume contract for ranged local-model downloads.

A mid-transfer failure must keep the preallocated .part and a span ledger.
The next download_file call fetches only the holes. A short read inside one
attempt retries the remainder instead of deleting every worker's bytes.
"""

from __future__ import annotations

import io
import threading

import pytest

from hermes_cli.web_routers import local_models


_URL = "https://example.invalid/model.gguf"
_BODY = bytes(range(32))


class _Resp(io.BytesIO):
    def __init__(self, body: bytes, *, status: int, headers: dict):
        super().__init__(body)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _RangeServer:
    """urlopen stand-in. ``limit(start, end)`` returns a byte cap, or None for the whole span."""

    def __init__(self, body: bytes, limit):
        self.body = body
        self.limit = limit
        self.requests: list[tuple[int, int]] = []
        self._lock = threading.Lock()

    def urlopen(self, req, timeout=None):
        raw = "" if isinstance(req, str) else (req.headers.get("Range") or "")
        if not raw.startswith("bytes="):
            return _Resp(self.body, status=200, headers={"Content-Length": str(len(self.body))})
        start_s, _, end_s = raw[len("bytes="):].partition("-")
        start = int(start_s)
        end = int(end_s) if end_s else len(self.body) - 1
        with self._lock:
            self.requests.append((start, end))
        if start == 0 and end == 0:
            return _Resp(self.body[:1], status=206, headers={
                "Content-Range": f"bytes 0-0/{len(self.body)}",
                "Content-Length": "1",
            })
        chunk = self.body[start:end + 1]
        cap = self.limit(start, end)
        if cap is not None:
            chunk = chunk[:cap]
        last = start + len(chunk) - 1 if chunk else start
        return _Resp(chunk, status=206, headers={
            "Content-Range": f"bytes {start}-{last}/{len(self.body)}",
            "Content-Length": str(len(chunk)),
        })


def _job() -> dict:
    return {"done_bytes": 0, "total_bytes": 0, "detail": ""}


def _data_requests(requests: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [span for span in requests if span != (0, 0)]


def test_incomplete_ranged_download_keeps_part_and_resume_fetches_only_holes(tmp_path, monkeypatch):
    """A short read must not delete the .part. The next call fetches only the hole."""
    dest = tmp_path / "model.gguf"
    part = dest.with_suffix(".part")
    ledger = part.with_name(part.name + ".spans.json")
    policy = {"limit": lambda start, end: 0 if start >= 16 else None}
    server = _RangeServer(_BODY, lambda start, end: policy["limit"](start, end))
    monkeypatch.setattr(local_models, "_DOWNLOAD_CONNECTIONS", 2)
    monkeypatch.setattr(local_models.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr("urllib.request.urlopen", server.urlopen)

    with pytest.raises(Exception):
        local_models.download_file(_URL, dest, _job())

    kept_part = part.exists()
    kept_ledger = ledger.exists()
    split = len(server.requests)
    policy["limit"] = lambda start, end: None
    local_models.download_file(_URL, dest, _job())
    resumed = _data_requests(server.requests[split:])

    assert kept_part and kept_ledger and resumed and all(start >= 16 for start, _end in resumed), (
        f"part_kept={kept_part} ledger_kept={kept_ledger} resume_requests={resumed} "
        f"(expected the .part and span ledger kept, and a retry that fetches only the hole)"
    )
    assert dest.read_bytes() == _BODY
    assert not part.exists()
    assert not ledger.exists()


def test_mid_span_short_read_retries_the_remainder_inside_the_attempt(tmp_path, monkeypatch):
    """One dropped range retries from the last byte written. Other workers' bytes stay."""
    dest = tmp_path / "model.gguf"
    part = dest.with_suffix(".part")
    dropped = {"done": False}

    def limit(start, end):
        if (start, end) == (16, 31) and not dropped["done"]:
            dropped["done"] = True
            return 4
        return None

    server = _RangeServer(_BODY, limit)
    monkeypatch.setattr(local_models, "_DOWNLOAD_CONNECTIONS", 2)
    monkeypatch.setattr(local_models.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr("urllib.request.urlopen", server.urlopen)
    job = _job()

    try:
        local_models.download_file(_URL, dest, job)
    except Exception as exc:
        pytest.fail(
            f"short read aborted the attempt instead of retrying the remainder; "
            f"part_exists={part.exists()} err={exc}"
        )

    data = _data_requests(server.requests)
    assert dest.read_bytes() == _BODY
    assert job["done_bytes"] == len(_BODY)
    assert (16, 31) in data
    assert (20, 31) in data
    assert data.count((0, 15)) == 1
    assert not part.exists()
