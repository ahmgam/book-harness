"""OpenAI-compatible client + `--api` selection (default: Ollama).

Covers ``OpenAIClient`` (chat-completions wire protocol), the shared
``build_client`` factory (auto-detect: host URL ending in ``/v1`` →
OpenAI-compatible, otherwise Ollama), and host/header normalization.
HTTP is stubbed at ``_get`` / ``_post`` — no server needed.
"""

import json
from urllib.error import URLError

import pytest

from book_harness.translate import (
    OllamaClient,
    OpenAIClient,
    build_client,
    is_openai_base,
    model_matches,
)

FINISHED = {
    "choices": [{"message": {"role": "assistant", "content": "  الإجابة  "}}]
}


def make_openai(models=("meta-llama/Llama-3.1-8B-Instruct",), key=""):
    """OpenAIClient with stubbed transport recording requests."""
    client = OpenAIClient("http://srv:8000", timeout=7, api_key=key)
    client.calls: list[tuple[str, dict]] = []
    client.failed: list[bool] = []

    def fake_get(path):
        if client.failed:
            raise URLError("down")
        return {"data": [{"id": m} for m in models]}

    def fake_post(path, payload, timeout=None):
        client.calls.append((path, payload))
        if client.failed:
            raise URLError("down")
        return FINISHED

    client._get = fake_get
    client._post = fake_post
    return client


def test_classifies_openai_base_urls():
    assert is_openai_base("http://vllm:8000/v1")
    assert is_openai_base("http://vllm:8000/v1/")
    # mid-path /v1 does not count (only the suffix, matching /v1 endpoints)
    assert not is_openai_base("http://vllm:8000/api/v1/back")
    # the default Ollama host stays Ollama
    assert not is_openai_base("http://localhost:11434")


def test_helpers_keep_their_api_paths_and_augment_headers():
    c = make_openai(key="sekret")
    c.pick_model(None)                                # use the first model
    c.generate("hi", system="sys", options={"temperature": 0.2})
    path, payload = c.calls[-1]
    assert path == "/chat/completions"
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["temperature"] == 0.2
    assert payload["stream"] is False
    assert payload["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    # context-window size is an Ollama option; not a chat-completions field
    assert "num_ctx" not in payload
    assert json.loads(json.dumps(payload))            # JSON-serializable
    assert c.generate("hi") == "الإجابة"             # parsed content, stripped
    assert c.api_key == "sekret"


def test_host_normalization_appends_v1_once():
    assert OpenAIClient("http://srv:8000").host == "http://srv:8000/v1"
    assert OpenAIClient("http://srv:8000/").host == "http://srv:8000/v1"
    assert OpenAIClient("http://srv:8000/v1").host == "http://srv:8000/v1"
    assert OpenAIClient("http://srv:8000/v1/").host == "http://srv:8000/v1"
    with pytest.raises(ValueError):
        OpenAIClient("")


def test_list_models_picks_by_preference_and_override():
    c = make_openai()
    assert c.list_models() == ["meta-llama/Llama-3.1-8B-Instruct"]
    c.pick_model("meta-llama/Llama-3.1-8B-Instruct")
    assert c.model == "meta-llama/Llama-3.1-8B-Instruct"
    c2 = make_openai(models=("m2",))
    with pytest.raises(RuntimeError, match="not installed"):
        c2.pick_model("nope")


def test_unreachable_list_models_raises_connection_error(monkeypatch):
    c = OpenAIClient("http://srv:8000", timeout=7)     # real transport, no stub
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", _raise_urlopen)
    with pytest.raises(ConnectionError, match="not reachable"):
        c.list_models()


def _raise_urlopen(*args, **kwargs):
    raise URLError("down")


def test_factory_defaults_to_ollama():
    class FakeOllama(OllamaClient):
        def list_models(self):
            return ["stub-a", "stub-b"]

    import book_harness.translate as tr

    orig = tr.OllamaClient
    tr.OllamaClient = FakeOllama
    try:
        c = build_client("http://localhost:11434")          # default: Ollama
        assert isinstance(c, FakeOllama)
        assert c.model == "stub-a"
        # explicit ollama even on a /v1 host
        c2 = build_client("http://srv:8000/v1", api="ollama")
        assert not isinstance(c2, OpenAIClient)
    finally:
        tr.OllamaClient = orig


def test_factory_selects_openai_for_v1_hosts(monkeypatch):
    monkeypatch.setattr(
        OpenAIClient, "list_models",
        lambda self: ["llama-4b"], raising=False)
    monkeypatch.setattr(
        OpenAIClient, "pick_model",
        OllamaClient.pick_model, raising=False)
    c = build_client("http://vllm:8000/v1", api_key="k")
    assert isinstance(c, OpenAIClient)
    assert c.model == "llama-4b" and c.api_key == "k"
    # forced selection works regardless of host shape
    c2 = build_client("http://host:1", api="openai")
    assert isinstance(c2, OpenAIClient) and c2.host.endswith("/v1")
    with pytest.raises(ValueError, match="unknown api kind"):
        build_client("http://host:1", api="grpc")


def test_model_matches_org_and_tag_shapes():
    # full id matching is case-insensitive
    installed = ["meta-llama/Llama-3.1-8B", "qwen2.5:14b-instruct"]
    assert model_matches(installed, "meta-llama/Llama-3.1-8B")
    assert model_matches(installed, "QWEN2.5:14b-INSTRUCT")
    # bare-name matching works for the part before the tag (no dot)
    assert model_matches(["unsloth/Llama-4B-Instruct"], "llama-4b-instruct")
    assert model_matches(["qwen2.5:14b-instruct"], "qwen2.5:14b-instruct")
    assert not model_matches(installed, "hy-mt2-30b-a3b")
