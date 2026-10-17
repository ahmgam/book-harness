"""M3.5 (PLAN.md): MCP tools + stdio server.

Library side (``book_harness.mcp``): the four tools — lookup_term,
get_glossary, translate_chunk, list_runs — tested with fake arabterm/client
backends (no DB download, no Ollama).

Server side (``mcp.server``): JSON-RPC dispatch through ``handle_message``
(initialize / ping / tools/list / tools/call / notifications), stdio loop
excluded — it is three lines of I/O.
"""

import json
from pathlib import Path

import pytest

from book_harness import mcp as lib
from book_harness.config import STAGE_DIRS


# ---------------------------------------------------------------------------
# fixtures (shapes match test_m4_web / test_m3_artifacts conventions)
# ---------------------------------------------------------------------------

CERTIFIED = {
    "term_en": "lift coefficient", "term_ar": "معامل الرفع",
    "source": "ksaa_aerospace", "source_tech": "ksaa_aerospace",
    "wikidata": "Q123", "page": 12, "scan": "https://archive.org/x/page/n12",
    "status": "certified", "decided_by": "auto", "alternatives": [],
}


class FakeArabterm:
    """Minimal Arabterm double."""

    def lookup_term(self, term, *args, **kwargs):
        return self.hits.get(term.lower(), [])

    def lookup_arabic(self, term_ar):
        return self.hits_ar.get(term_ar.strip().lower(), [])


@pytest.fixture()
def fake_arabterm():
    at = FakeArabterm()
    at.hits = {
        "lift coefficient": [{
            "ar": "معامل الرفع", "en": "lift coefficient",
            "source": "ksaa_aerospace", "source_tech": "ksaa_aerospace",
            "wikidata": "Q123", "page": 12,
            "scan": "https://archive.org/x/page/n12", "description": "",
        }],
        "electropedia term": [{
            "ar": "مصطلح", "en": "electropedia term",
            "source": "electropedia", "source_tech": "electropedia",
            "wikidata": None, "page": 1, "scan": None, "description": "",
        }],
    }
    at.hits_ar = {
        "معامل الرفع": at.hits["lift coefficient"],
    }
    return at


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    root = tmp_path / "runs" / "control-systems"
    (root / STAGE_DIRS["chunks"]).mkdir(parents=True)
    (root / STAGE_DIRS["drafts"]).mkdir(parents=True)
    (root / STAGE_DIRS["chunks"] / "chunk-0001.json").write_text(
        json.dumps({"chunk_id": "chunk-0001"}, ensure_ascii=False),
        encoding="utf-8")
    (root / STAGE_DIRS["drafts"] / "chunk-0001.meta.json").write_text(
        json.dumps({"chunk_id": "chunk-0001"}, ensure_ascii=False),
        encoding="utf-8")
    (root / "state.json").write_text(json.dumps({
        "version": 1, "book_id": "control-systems", "created": 1.0,
        "stages": {
            "source": {"status": "done", "started": 1.0, "finished": 2.0},
            "translate": {"status": "started", "started": 3.0},
        },
    }), encoding="utf-8")
    return root


@pytest.fixture()
def glossary(run_dir: Path):
    from book_harness.glossary import GlossaryEntry, write_glossary

    entries = [GlossaryEntry(CERTIFIED), GlossaryEntry.uncertified(
        "wearout", "no dictionary hit")]
    path, version = write_glossary(
        entries, run_dir / STAGE_DIRS["glossary"] / "glossary.yaml")
    return version


# ---------------------------------------------------------------------------
# lookup_term
# ---------------------------------------------------------------------------

def test_lookup_term_returns_citations(fake_arabterm):
    out = lib.lookup_term("Lift Coefficient", arabterm=fake_arabterm)
    assert out["lang"] == "en"
    assert out["domain"] is None
    assert out["hits"][0]["ar"] == "معامل الرفع"
    assert "scan" in out["hits"][0]


def test_lookup_term_domain_filters_dictionaries(fake_arabterm):
    allowed = lib.lookup_term("lift coefficient", domain="aerospace",
                              arabterm=fake_arabterm)
    assert len(allowed["hits"]) == 1
    blocked = lib.lookup_term("electropedia term", domain="aerospace",
                              arabterm=fake_arabterm)
    assert blocked["hits"] == []       # electropedia not on the whitelist
    unfiltered = lib.lookup_term("electropedia term", arabterm=fake_arabterm)
    assert len(unfiltered["hits"]) == 1


def test_lookup_term_reverse_lookup_and_validation(fake_arabterm):
    out = lib.lookup_term("معامل الرفع", lang="ar", arabterm=fake_arabterm)
    assert out["lang"] == "ar"
    assert out["hits"][0]["en"] == "lift coefficient"
    with pytest.raises(ValueError):
        lib.lookup_term("   ", arabterm=fake_arabterm)


# ---------------------------------------------------------------------------
# get_glossary
# ---------------------------------------------------------------------------

def test_get_glossary_returns_version_and_entries(run_dir, glossary):
    out = lib.get_glossary("control-systems", run_root=run_dir.parent)
    assert out["version"] == glossary
    assert out["count"] == 2
    assert out["by_status"] == {"certified": 1, "uncertified": 1}
    assert out["entries"][0]["source"] == "ksaa_aerospace"


def test_get_glossary_single_term_filter_matches_en_and_ar(run_dir, glossary):
    args = {"run_root": run_dir.parent}
    lift = lib.get_glossary("control-systems", "lift coefficient", **args)
    assert lift["count"] == 1 and lift["term"] == "lift coefficient"
    ar = lib.get_glossary("control-systems", "معامل الرفع", **args)
    assert ar["count"] == 1           # reverse (Arabic-form) matching too
    none = lib.get_glossary("control-systems", "thrust vector", **args)
    assert none["count"] == 0


def test_get_glossary_missing_book_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        lib.get_glossary("no-such-book", run_root=tmp_path)


# ---------------------------------------------------------------------------
# translate_chunk
# ---------------------------------------------------------------------------

class FakeClient:
    model = "hy-mt2-7b"
    last_prompt = ""
    last_system = ""

    def generate(self, prompt, *, system=None, options=None):
        FakeClient.last_prompt = prompt
        FakeClient.last_system = system or ""
        return "الترجمة العربية"


def test_translate_chunk_injects_only_matching_terms(monkeypatch):
    from book_harness.config import GLOSSARY_TEMPLATE

    monkeypatch.setattr(lib, "make_client", lambda model=None: FakeClient())
    glossary = [
        {"en": "thrust reversal", "ar": "الدفع العكسي"},
        {"en": "lift coefficient", "ar": "معامل الرفع"},   # not in text
    ]
    out = lib.translate_chunk(
        "The thrust reversal engages on landing.", glossary,
        context="Previous translation: …")
    assert out["model"] == "hy-mt2-7b"
    assert out["translation"] == "الترجمة العربية"
    assert GLOSSARY_TEMPLATE.format(en="thrust reversal", ar="الدفع العكسي") \
        in FakeClient.last_prompt
    assert "lift coefficient" not in FakeClient.last_prompt
    assert "Previous translation" in FakeClient.last_prompt
    assert FakeClient.last_system        # system prompt present


def test_translate_chunk_accepts_alternative_keys(monkeypatch):
    monkeypatch.setattr(lib, "make_client", lambda model=None: FakeClient())
    lib.translate_chunk(
        "lift coefficient rises", [{"term_en": "lift coefficient",
                                    "term_ar": "معامل الرفع"}])
    assert "معامل الرفع" in FakeClient.last_prompt


def test_translate_chunk_wraps_failures(monkeypatch):
    class Boom(FakeClient):
        def generate(self, prompt, *, system=None, options=None):
            raise RuntimeError("model blown")

    monkeypatch.setattr(lib, "make_client", lambda model=None: Boom())
    with pytest.raises(RuntimeError, match="translation failed"):
        lib.translate_chunk("text", [{"en": "lift", "ar": "الرفع"}])
    with pytest.raises(ValueError):
        lib.translate_chunk("   ", [])


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------

def test_list_runs_summarizes_state_and_counts(run_dir, glossary):
    runs = lib.list_runs(run_root=run_dir.parent)
    assert [r["book_id"] for r in runs] == ["control-systems"]
    run = runs[0]
    assert run["stages"]["source"] == "done"
    assert run["stages"]["translate"] == "started"
    assert run["chunks"] == 1 and run["translated"] == 1
    assert run["glossary"]["version"] == glossary
    assert run["glossary"]["certified"] == 1
    assert run["assembled"] is False


def test_list_runs_skips_non_run_dirs(tmp_path):
    (tmp_path / "not-a-run").mkdir()
    (tmp_path / "not-a-run" / "random.txt").write_text("x")
    assert lib.list_runs(run_root=tmp_path) == []
    assert lib.list_runs(run_root=tmp_path / "missing") == []


# ---------------------------------------------------------------------------
# MCP server: JSON-RPC dispatch
# ---------------------------------------------------------------------------

def _rpc(method, params=None, id=1):
    msg = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


@pytest.fixture()
def server():
    from book_harness import mcp_server

    return mcp_server


def test_initialize_negotiates_protocol_version(server):
    out = server.handle_message(
        _rpc("initialize", {"protocolVersion": "2024-11-05"}))
    assert out["result"]["protocolVersion"] == "2024-11-05"
    assert out["result"]["capabilities"]["tools"]
    assert out["result"]["serverInfo"]["name"] == "book-harness"
    unknown = server.handle_message(
        _rpc("initialize", {"protocolVersion": "1999-01-01"}, id=2))
    assert unknown["result"]["protocolVersion"] == \
        server.DEFAULT_PROTOCOL_VERSION


def test_tools_list_exposes_the_four_tools(server):
    out = server.handle_message(_rpc("tools/list"))
    names = [t["name"] for t in out["result"]["tools"]]
    assert sorted(names) == sorted(lib.TOOL_NAMES)
    spec = next(t for t in out["result"]["tools"] if t["name"] == "lookup_term")
    assert spec["inputSchema"]["required"] == ["term"]
    assert {"text", "glossary"} <= set(
        next(t for t in out["result"]["tools"]
             if t["name"] == "translate_chunk")["inputSchema"]["required"])


def test_tools_call_dispatches_and_json_encodes(server, run_dir, glossary):
    out = server.handle_message(_rpc("tools/call", {
        "name": "list_runs",
        "arguments": {"run_root": str(run_dir.parent)},
    }))
    assert "isError" not in out
    payload = json.loads(out["result"]["content"][0]["text"])
    assert payload[0]["book_id"] == "control-systems"


def test_tools_call_tool_execution_errors_are_isError_results(server, tmp_path):
    out = server.handle_message(_rpc("tools/call", {
        "name": "get_glossary",
        "arguments": {"book_id": "nope", "run_root": str(tmp_path)},
    }))
    assert out["result"]["isError"] is True
    assert "no glossary" in out["result"]["content"][0]["text"]


def test_tools_call_translation_error_is_returned_not_raised(server, monkeypatch):
    class Boom(FakeClient):
        def generate(self, prompt, *, system=None, options=None):
            raise RuntimeError("model blown")

    monkeypatch.setattr(lib, "make_client", lambda model=None: Boom())
    out = server.handle_message(_rpc("tools/call", {
        "name": "translate_chunk",
        "arguments": {"text": "lift", "glossary": [{"en": "lift",
                                                    "ar": "الرفع"}]},
    }))
    assert out["result"]["isError"] is True
    assert "translation failed" in out["result"]["content"][0]["text"]


def test_unknown_tool_is_invalid_params(server):
    out = server.handle_message(_rpc("tools/call", {"name": "nope"}))
    assert out["error"]["code"] == server.INVALID_PARAMS
    assert "nope" in out["error"]["message"]
    assert "get_glossary" in out["error"]["data"]["known"]


def test_unknown_method_and_notifications(server):
    err = server.handle_message(_rpc("resources/list"))
    assert err["error"]["code"] == server.METHOD_NOT_FOUND
    # notification (no id): no response at all
    assert server.handle_message({
        "jsonrpc": "2.0", "method": server.INITIALIZED_NOTIFICATION,
    }) is None
    assert server.handle_message("not-a-object") is not None


def test_stdio_loop_translates_messages_to_lines(server):
    import io

    inp = io.StringIO("\n".join([
        json.dumps(_rpc("initialize", {"protocolVersion": "2024-11-05"})),
        json.dumps({"jsonrpc": "2.0",
                    "method": server.INITIALIZED_NOTIFICATION}),
        "",                                  # blank line ignored
        "{not json",
        json.dumps(_rpc("tools/list")),
    ]) + "\n")
    out = io.StringIO()
    code = server.main(inp=inp, out=out)
    assert code == 0
    frames = out.getvalue().strip().splitlines()
    assert len(frames) == 3                 # initialize + parse-error + tools
    assert json.loads(frames[0])["result"]["serverInfo"]["name"] == "book-harness"
    assert json.loads(frames[1])["error"]["code"] == server.PARSE_ERROR
    assert "tools" in json.loads(frames[2])["result"]
