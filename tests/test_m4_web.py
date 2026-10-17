"""M4 web UI (PLAN.md tasks 4.1 + 4.2): progress snapshot + SSE stream,
review screen items, and human decision write-backs."""

import http.client
import json
import threading
import time
from pathlib import Path

import pytest

from book_harness import web
from book_harness.config import STAGE_DIRS

# ---------------------------------------------------------------------------
# fixtures (same minimal shapes as test_m3_artifacts)
# ---------------------------------------------------------------------------

CERTIFIED = {
    "term_en": "lift", "term_ar": "الرفع",
    "source": "ksaa_aerospace", "status": "certified",
    "decided_by": "auto", "alternatives": [],
}

CHUNKS = {
    "chunk-0001": {
        "index": 1,
        "chunk_id": "chunk-0001",
        "section_id": "/page/1/SectionHeader/0",
        "section_path": [{"number": "1", "title": "Design"}],
        "section_title": "1 > Design",
        "pages": [1],
        "source_hash": "abc",
        "blocks": [{"block_id": "/page/1/Text/0", "page": 1,
                    "type": "Text", "units": ["u0", "u1"]}],
        "units": [
            {"uid": "u0", "block_id": "/page/1/Text/0", "kind": "text",
             "text": "The lift equation predicts the lift coefficient.",
             "placeholders": {}},
            {"uid": "u1", "block_id": "/page/1/Text/0", "kind": "text",
             "text": "The thrust reversal engages on landing.",
             "placeholders": {}},
        ],
    },
    "chunk-0002": {
        "index": 2,
        "chunk_id": "chunk-0002",
        "section_id": "/page/2/SectionHeader/0",
        "section_path": [{"number": "2", "title": "Controls"}],
        "section_title": "2 > Controls",
        "pages": [2],
        "source_hash": "def",
        "blocks": [{"block_id": "/page/2/Text/0", "page": 2,
                    "type": "Text", "units": ["u0"]}],
        "units": [
            {"uid": "u0", "block_id": "/page/2/Text/0", "kind": "text",
             "text": "Control surfaces respond to stick input.",
             "placeholders": {}},
        ],
    },
}

DRAFTS = {
    "chunk-0001": {
        "chunk_id": "chunk-0001",
        "cache_key": "key-1",
        "model": "hy-mt2-7b",
        "prompt_version": "m1-v1",
        "glossary_version": "gv1",
        "section": "1 > Design",
        "pages": [1],
        "finished": "2026-01-01T00:00:00",
        "units": [
            {"uid": "u0", "kind": "text", "text": "معادلة الرفع تتنبأ بمعامل الرفع."},
            {"uid": "u1", "kind": "text", "text": "الدفع العكسي يشتغل عند الهبوط."},
        ],
    },
    "chunk-0002": {
        "chunk_id": "chunk-0002",
        "cache_key": "key-2",
        "model": "hy-mt2-7b",
        "prompt_version": "m1-v1",
        "glossary_version": "gv1",
        "section": "2 > Controls",
        "pages": [2],
        "finished": "2026-01-01T00:00:01",
        "units": [
            {"uid": "u0", "kind": "text", "text": "تستجيب أسطح التحكم لعصا القيادة."},
        ],
    },
}

QE_SCORES = [
    {"chunk_id": "chunk-0002", "method": "judge", "model": "q",
     "score": 90.0, "unit_scores": {"u0": 90},
     "finished": 1.0, "cache_key": "q2"},
    {"chunk_id": "chunk-0001", "method": "judge", "model": "q",
     "score": 55.0, "unit_scores": {"u0": 60, "u1": 50},
     "finished": 1.0, "cache_key": "q1"},      # worst first in output
]

STATE = {
    "version": 1,
    "book_id": "test-book",
    "created": 1.0,
    "stages": {
        "source": {"status": "done", "started": 1.0, "finished": 1.0},
        "translate": {"status": "started", "started": 2.0},
    },
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    root = tmp_path / "runs" / "test-book"
    chunks_dir = root / STAGE_DIRS["chunks"]
    drafts_dir = root / STAGE_DIRS["drafts"]
    chunks_dir.mkdir(parents=True)
    drafts_dir.mkdir(parents=True)
    for chunk in CHUNKS.values():
        (chunks_dir / f"{chunk['chunk_id']}.json").write_text(
            json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
    for cid, draft in DRAFTS.items():
        (drafts_dir / f"{cid}.meta.json").write_text(
            json.dumps(draft, ensure_ascii=False), encoding="utf-8")
        (drafts_dir / f"{cid}.md").write_text(f"# {cid}\n", encoding="utf-8")
    _write_jsonl(root / STAGE_DIRS["qa"] / "qe-scores.jsonl", QE_SCORES)
    _write_jsonl(root / "02_glossary" / "drift-candidates.jsonl", [
        {"term": "thrust reversal", "freq": 4, "df": 2,
         "chunks": ["chunk-0001"], "page": 1, "score": 9.0,
         "status": "proposed", "decision": None},
    ])
    return root


@pytest.fixture()
def glossary(run_dir: Path):
    from book_harness.glossary import GlossaryEntry, write_glossary

    entries = [GlossaryEntry(CERTIFIED), GlossaryEntry.uncertified(
        "wearout", "no dictionary hit")]
    path, _ = write_glossary(entries, run_dir / "02_glossary" / "glossary.yaml")
    return path


def _init_state(run_dir: Path) -> None:
    (run_dir / "state.json").write_text(
        json.dumps(STATE, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 4.1 — progress snapshot / stream
# ---------------------------------------------------------------------------

def test_overview_counts_stages_and_translation(run_dir):
    _init_state(run_dir)
    o = web.overview(run_dir)
    assert o["book_id"] == "test-book"
    assert o["translation"] == {
        "total": 2, "translated": 2, "running": True, "percent": 100.0,
    }
    statuses = {s["stage"]: s for s in o["stages"]}
    assert statuses["source"]["status"] == "done"
    assert statuses["translate"]["status"] == "started"
    assert statuses["assemble"]["status"] == "pending"
    assert o["review"]["uncertified_terms"] == 1   # wearout (+ drift proposal
    assert o["review"]["low_qe_chunks"] == 1       # is deduped against nothing


def test_overview_is_safe_on_an_empty_run_dir(tmp_path):
    o = web.overview(tmp_path / "empty")
    assert o["book_id"] == "empty"
    assert o["translation"]["total"] == 0
    assert [s["stage"] for s in o["stages"]] == web.STAGE_ORDER


def test_progress_stream_emits_only_on_change(run_dir, glossary):
    _init_state(run_dir)
    before = web.overview(run_dir)
    stream = web.progress_stream(run_dir, poll_seconds=0.01)
    first = next(stream)
    assert first.startswith("event: progress\ndata: {")
    assert '"book_id": "test-book"' in first
    # empty dir -> nothing changes -> second tick is a heartbeat, not a new event
    second = next(stream)
    assert second.startswith(": heartbeat")
    assert before["translation"]["translated"] == 2


# ---------------------------------------------------------------------------
# 4.2 — review items
# ---------------------------------------------------------------------------

def test_uncertified_terms_merges_glossary_and_drift(run_dir, glossary):
    items = web.uncertified_terms(run_dir)
    by_term = {i["term_en"]: i for i in items}
    assert set(by_term) == {"wearout", "thrust reversal"}
    wearout = by_term["wearout"]
    assert wearout["origin"] == "glossary"
    assert wearout["decided"] is False
    proposal = by_term["thrust reversal"]
    assert proposal["origin"] == "drift"
    assert proposal["freq"] == 4


def test_low_qe_chunks_worst_first_with_pairs(run_dir, glossary):
    items = web.low_qe_chunks(run_dir, threshold=80.0)
    assert [i["chunk_id"] for i in items] == ["chunk-0001"]
    item = items[0]
    assert item["score"] == 55.0
    assert item["section"] == "1 > Design"
    units = {u["uid"]: u for u in item["units"]}
    assert units["u0"]["source"] == CHUNKS["chunk-0001"]["units"][0]["text"]
    assert units["u0"]["text"] == DRAFTS["chunk-0001"]["units"][0]["text"]
    assert units["u0"]["score"] == 60


def test_low_qe_chunks_above_threshold_are_excluded(run_dir):
    items = web.low_qe_chunks(run_dir, threshold=50.0)
    assert items == []


# ---------------------------------------------------------------------------
# Write-back: requeue
# ---------------------------------------------------------------------------

def test_requeue_chunk_invalidates_and_audits(run_dir):
    drafts = run_dir / STAGE_DIRS["drafts"]
    assert web.requeue_chunk(run_dir, "chunk-0001", reason="bad QE") is True
    assert not (drafts / "chunk-0001.meta.json").exists()
    assert not (drafts / "chunk-0001.md").exists()
    archived = list((drafts / "requeued").glob("1_chunk-0001*"))
    assert {p.name for p in archived} == {"1_chunk-0001.meta.json",
                                          "1_chunk-0001.md"}
    log = (run_dir / "06_qa" / "review-log.jsonl").read_text(encoding="utf-8")
    rec = json.loads(log.strip())
    assert rec["action"] == "requeue"
    assert rec["chunk_id"] == "chunk-0001"
    # idempotent: nothing left to invalidate
    assert web.requeue_chunk(run_dir, "chunk-0001") is False
    assert web.requeue_chunk(run_dir, "chunk-9999") is False


# ---------------------------------------------------------------------------
# Write-back: term decisions
# ---------------------------------------------------------------------------

def test_approve_writes_glossary_and_requeues_affected_chunks(run_dir, glossary):
    import yaml

    version_before = yaml.safe_load(glossary.read_text(encoding="utf-8"))["version"]

    result = web.apply_term_decision(
        run_dir, "thrust reversal", decision="approve",
        arabic="انعكاس الدفع",
    )
    assert result["action"] == "appended"
    assert result["entry"]["term_ar"] == "انعكاس الدفع"
    assert result["entry"]["decided_by"] == "human"
    assert result["entry"]["status"] == "uncertified"   # never self-certified
    assert result["glossary_version"] != version_before
    assert result["requeued"] == ["chunk-0001"]
    drafts = run_dir / STAGE_DIRS["drafts"]
    assert not (drafts / "chunk-0001.meta.json").exists()  # re-queued
    assert (drafts / "chunk-0002.meta.json").exists()     # untouched

    # drift proposal is resolved in place
    cands = [json.loads(l) for l in
             (run_dir / "02_glossary" / "drift-candidates.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    assert cands[0]["status"] == "confirmed"

    # audit trail
    log = (run_dir / "06_qa" / "review-log.jsonl").read_text(encoding="utf-8")
    entries = [json.loads(l) for l in log.strip().splitlines()]
    assert any(e["action"] == "term" and e["term"] == "thrust reversal"
               for e in entries)

    # the reviewed term is no longer "pending" for the queue
    items = web.uncertified_terms(run_dir)
    pending = [i for i in items if not i["decided"]]
    assert all(i["term_en"] != "thrust reversal" for i in pending)


def test_approve_without_arabic_is_rejected(run_dir, glossary):
    from book_harness.glossary import load_glossary

    before = load_glossary(glossary)[0]
    with pytest.raises(ValueError):
        web.apply_term_decision(run_dir, "wearout", decision="approve")
    assert load_glossary(glossary)[0] == before


def test_reject_keeps_term_catalogued_without_arabic(run_dir, glossary):
    result = web.apply_term_decision(run_dir, "wearout", decision="reject")
    assert result["entry"]["term_ar"] is None
    assert result["entry"]["human_decision"] == "rejected"
    assert result["entry"]["decided_by"] == "human"
    assert result["requeued"] == []
    items = {i["term_en"]: i for i in web.uncertified_terms(run_dir)}
    assert items["wearout"]["human_decision"] == "rejected"
    assert items["wearout"]["decided"] is True


def test_decision_normalises_term_keys(run_dir, glossary):
    import yaml

    # glossary holds 'wearout'; the UI may send variant punctuation
    result = web.apply_term_decision(run_dir, "Wearout", decision="reject")
    assert result["action"] == "updated"
    data = yaml.safe_load(glossary.read_text(encoding="utf-8"))
    assert len(data["entries"]) == 2          # no duplicate appended


# ---------------------------------------------------------------------------
# HTTP shell: routes, writes, and the SSE stream
# ---------------------------------------------------------------------------

@pytest.fixture()
def server(run_dir, glossary):
    _init_state(run_dir)
    httpd = web.make_server(run_dir, host="127.0.0.1", port=0,
                            poll_seconds=0.05)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        yield port
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(port: int, path: str, *, timeout: float = 5.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp, body


def _post(port: int, path: str, payload: dict, *, timeout: float = 5.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request(
        "POST", path, body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp, body


def test_http_index_and_overview(server):
    port = server
    resp, body = _get(port, "/")
    assert resp.status == 200
    assert b"review" in body and b"EventSource" in body
    resp, body = _get(port, "/api/overview")
    data = json.loads(body)
    assert data["book_id"] == "test-book"
    assert data["translation"]["total"] == 2


def test_http_review_items(server):
    port = server
    resp, body = _get(port, "/api/review")
    assert resp.status == 200
    data = json.loads(body)
    assert {t["term_en"] for t in data["terms"]} >= {"wearout",
                                                     "thrust reversal"}
    assert [c["chunk_id"] for c in data["low_qe"]] == ["chunk-0001"]


def test_http_unknown_route_404(server):
    port = server
    resp, body = _get(port, "/api/nope")
    assert resp.status == 404


def test_http_term_decision_end_to_end(server, run_dir):
    port = server
    resp, body = _post(port, "/api/terms", {
        "term_en": "thrust reversal", "decision": "approve",
        "ar": "انعكاس الدفع",
    })
    assert resp.status == 200
    data = json.loads(body)
    assert data["requeued"] == ["chunk-0001"]
    assert "انعكاس الدفع" in (run_dir / "02_glossary" / "glossary.yaml").read_text(encoding="utf-8")
    assert not (run_dir / "04_drafts" / "chunk-0001.meta.json").exists()


def test_http_requeue_endpoint(server, run_dir):
    port = server
    resp, body = _post(port, "/api/requeue", {"chunk_id": "chunk-0002"})
    assert resp.status == 200
    assert json.loads(body)["requeued"] is True
    assert not (run_dir / "04_drafts" / "chunk-0002.meta.json").exists()


def test_http_bad_decision_is_400(server):
    port = server
    resp, body = _post(port, "/api/terms", {
        "term_en": "wearout", "decision": "approve",
    })
    assert resp.status == 400
    assert "Arabic" in json.loads(body)["error"]


def test_http_sse_stream_emits_initial_snapshot(server):
    port = server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    conn.request("GET", "/events")
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type").startswith("text/event-stream")
    # read until the first full progress event
    buf = b""
    deadline = time.time() + 4.0
    while b"event: progress\n" not in buf and time.time() < deadline:
        buf += resp.read1(1024)
    conn.close()
    assert buf.startswith(b"event: progress\ndata: ")


def _write_state(path: Path, stages: dict) -> None:
    path.write_text(json.dumps({"version": 1, "book_id": "test-book",
                                "created": 1.0, "stages": stages}),
                    encoding="utf-8")


def test_http_sse_stream_pushes_updates(server, run_dir):
    port = server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    conn.request("GET", "/events")
    resp = conn.getresponse()

    def read_until(marker: bytes, timeout: float = 5.0) -> bytes:
        buf = b""
        deadline = time.time() + timeout
        while marker not in buf and time.time() < deadline:
            try:
                chunk = resp.read1(1024)
            except TimeoutError:
                break
            if not chunk:
                break
            buf += chunk
        return buf

    assert b"event: progress" in read_until(b"event: progress")
    # mutate the state file in the background -> the stream must push a
    # second, different snapshot
    time.sleep(0.2)
    stages = dict(STATE["stages"])
    stages["translate"] = {"status": "done", "started": 2.0, "finished": 3.0}
    _write_state(run_dir / "state.json", stages)
    buf = b""
    deadline = time.time() + 6.0
    while not buf.startswith(b"event: progress") and time.time() < deadline:
        try:
            chunk = resp.read1(1024)
        except TimeoutError:
            break
        if not chunk:
            break
        buf += chunk
    conn.close()
    # exactly one frame per change; the payload must reflect the update
    assert buf.startswith(b"event: progress\ndata: ")
    assert b'"status": "done"' in buf
