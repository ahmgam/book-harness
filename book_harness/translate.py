"""Ollama translation client (task 1.8).

Features per plan:
* model preference list from config with automatic fallback to whatever is
  installed locally (found via ``/api/tags``);
* glossary injection in the Hy-MT2 official terminology template
  (```X` translates to `Y``) — only terms that actually occur in the chunk;
* prior-translated-paragraph context + rolling chapter summary refreshed
  every ``SUMMARY_PERIOD`` chunks;
* per-chunk cache key ``hash(source_chunk + glossary_version + model +
  prompt_version)``; a fresh draft short-circuits the request (resume-safe);
* drafts on disk: ``chunk-NNNN.md`` (human preview) + ``.meta.json``
  (model, prompt version, glossary version, per-unit texts, timings).
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from book_harness.config import (
    GENERATE_OPTIONS,
    GLOSSARY_TEMPLATE,
    OPENAI_API_KEY,
    OLLAMA_HOST,
    OLLAMA_TIMEOUT,
    SYSTEM_PROMPT,
    TRANSLATION_MODELS,
)
from book_harness.state import cache_key

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"§(?P<kind>[A-Z]+)-(?P<n>\d+)§")

#: Options keys understood by OpenAI-style chat-completions endpoints.
#: ``num_ctx`` is llama.cpp/Ollama context-window sizing and is dropped here.
OPENAI_OPTION_KEYS = (
    "temperature", "top_p", "seed", "stop", "max_tokens",
    "presence_penalty", "frequency_penalty",
)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """Minimal stdlib Ollama client (non-streaming generate)."""

    def __init__(
        self,
        host: str,
        timeout: int = 900,
        preferred: list[str] | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.preferred = preferred or []
        self.model: str | None = None

    # -- transport -----------------------------------------------------------
    def _post(self, path: str, payload: dict, timeout: int | None = None) -> dict:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> Any:
        with urllib.request.urlopen(self.host + path, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # -- api -----------------------------------------------------------
    def list_models(self) -> list[str]:
        try:
            tags = self._get("/api/tags")
        except (urllib.error.URLError, ConnectionError) as exc:
            raise ConnectionError(
                f"Ollama not reachable at {self.host}: {exc}"
            ) from exc
        return sorted(m.get("name", "") for m in tags.get("models", []))

    def pick_model(self, model_override: str | None = None) -> str:
        return _pick_model_installed(self, model_override)

    @staticmethod
    def _matches(installed: list[str], want: str) -> bool:
        return model_matches(installed, want)

    def generate(
        self, prompt: str, *, system: str | None = None, options: dict | None = None
    ) -> str:
        if not self.model:
            raise RuntimeError("pick_model() must be called before generate()")
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {**GENERATE_OPTIONS, **(options or {})},
        }
        if system:
            payload["system"] = system
        out = self._post("/api/generate", payload)
        return (out.get("response") or "").strip()


def model_matches(installed: list[str], want: str) -> bool:
    """Case-insensitive model matching tolerant of ``:tag`` and ``org/name``."""
    want = (want or "").lower()
    for name in installed:
        n = name.lower()
        bare = n.split("/")[-1].split(".")[0]
        if (
            n == want
            or n.startswith(want + ":")
            or n == want + ":latest"
            or bare == want
            or bare.startswith(want + ":")
            or bare == want + ":latest"
        ):
            return True
    return False


def _canonical_name(installed: list[str], want: str) -> str:
    """Resolve ``want`` to the *actually installed* model name.

    Must agree with :func:`model_matches`: whatever that matcher accepts has
    to be resolvable here, otherwise the client keeps the bare/unqualified
    name and the server rejects it at generation time (HTTP 404).
    """
    want = (want or "").lower()
    # exact (full id or tag-suffixed) names win over loose matches
    for name in installed:
        if name.lower() in (want, want + ":latest"):
            return name
    for name in installed:
        if model_matches([name], want):
            return name
    return want


def _pick_model_installed(
    client: Any, model_override: str | None = None
) -> str:
    """Shared pick_model implementation (OllamaClient + OpenAIClient)."""
    installed = client.list_models()
    if not installed:
        raise RuntimeError(
            f"no models available at {client.host} "
            f"(endpoint returned an empty list)"
        )
    client._installed = installed
    if model_override:
        if model_matches(installed, model_override):
            client.model = _canonical_name(installed, model_override)
        else:
            raise RuntimeError(
                f"model '{model_override}' not installed on the server at "
                f"{client.host}; available: {installed}"
            )
        return client.model
    for cand in client.preferred:
        if model_matches(installed, cand):
            client.model = _canonical_name(installed, cand)
            logger.info("model (%s): %s",
                        ",".join(client.preferred[:2]) or "unspecified",
                        client.model)
            return client.model
    client.model = installed[0]
    logger.warning(
        "none of the preferred models %s installed; using '%s'",
        client.preferred, client.model,
    )
    return client.model


def app_base(host: str) -> str:
    """Normalize a server base URL, keeping an explicit ``/v1`` suffix."""
    return (host or "").strip().rstrip("/")


def is_openai_base(host: str) -> bool:
    """A host URL ending in ``/v1`` selects the OpenAI-compatible API."""
    return app_base(host).endswith("/v1")


class OpenAIClient:
    """Stdlib client for OpenAI-compatible servers (vLLM, llama.cpp server,
    LM Studio, TGI …): ``GET /v1/models`` + ``POST /v1/chat/completions``.

    Same duck-type surface as :class:`OllamaClient` (``.model``,
    ``list_models``, ``pick_model``, ``generate(prompt, system, options)``),
    so the whole pipeline works unchanged. The base host may be written with
    or without the ``/v1`` suffix; requests go to ``{host}/v1/…`` either way.
    """

    def __init__(
        self,
        host: str,
        timeout: int = 900,
        preferred: list[str] | None = None,
        api_key: str | None = None,
    ) -> None:
        base = app_base(host)
        if not base:
            raise ValueError("openai host must be a base URL")
        self.host = base if base.endswith("/v1") else base + "/v1"
        self.timeout = timeout
        self.preferred = preferred or []
        self.api_key = api_key or ""
        self.model: str | None = None

    # -- transport -----------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: dict, timeout: int | None = None) -> dict:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise ConnectionError(
                f"OpenAI-compatible server not reachable at {self.host}{path}: {exc}"
            ) from exc

    def _get(self, path: str) -> Any:
        req = urllib.request.Request(
            self.host + path, headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise ConnectionError(
                f"OpenAI-compatible server not reachable at {self.host}{path}: {exc}"
            ) from exc

    # -- api ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        data = self._get("/models")
        models = data.get("data") if isinstance(data, dict) else data
        return sorted(
            m.get("id") if isinstance(m, dict) else str(m) for m in (models or [])
        )

    pick_model = OllamaClient.pick_model          # shared implementation
    _matches = OllamaClient._matches              # shared, module-level logic

    def generate(
        self, prompt: str, *, system: str | None = None, options: dict | None = None
    ) -> str:
        if not self.model:
            raise RuntimeError("pick_model() must be called before generate()")
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        passed = {**GENERATE_OPTIONS, **(options or {})}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **{k: passed[k] for k in OPENAI_OPTION_KEYS if k in passed},
        }
        out = self._post("/chat/completions", payload)
        choices = out.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"no choices in chat-completions response: {str(out)[:200]}"
            )
        return (choices[0].get("message", {}).get("content") or "").strip()


# ---------------------------------------------------------------------------
# Client factory (task M2.7-era Ollama default + OpenAI-compatible option)
# ---------------------------------------------------------------------------

def build_client(
    host: str = "",
    *,
    timeout: int | None = None,
    preferred: list[str] | None = None,
    model: str | None = None,
    api: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Construct, model-pick and return the right client for ``host``.

    ``api`` selects the wire protocol:
    * ``None`` / ``"auto"`` — OpenAI-compatible when the host URL ends in
      ``/v1``, otherwise Ollama (**default: Ollama**, per PLAN §3);
    * ``"ollama"`` / ``"openai"`` — force one.
    """
    host = host or OLLAMA_HOST
    timeout = timeout or OLLAMA_TIMEOUT
    preferred = preferred or TRANSLATION_MODELS
    kind = (api or "auto").lower()
    if kind == "auto":
        kind = "openai" if is_openai_base(host) else "ollama"
    if kind == "openai":
        client: Any = OpenAIClient(
            host=host,
            timeout=timeout,
            preferred=preferred,
            api_key=api_key or OPENAI_API_KEY,
        )
    elif kind == "ollama":
        client = OllamaClient(host=host, timeout=timeout, preferred=preferred)
    else:
        raise ValueError(f"unknown api kind: {kind!r} (ollama|openai|auto)")
    client.pick_model(model)
    return client


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def glossary_lines(terms: list[tuple[str, str | None]]) -> list[str]:
    lines = []
    for en, ar in terms:
        if ar:
            lines.append(GLOSSARY_TEMPLATE.format(en=en, ar=ar))
    return lines


def match_terms(
    text: str, entries: list[dict], *, max_terms: int = 40
) -> list[tuple[str, str]]:
    """Glossary entries whose English term occurs in the chunk text (word-boundary)."""
    hay = text.lower()
    out = []
    for e in entries:
        en = e.get("term_en") or ""
        if not en or len(en) < 3 or not e.get("term_ar"):
            continue
        pat = r"\b" + r"\s+".join(re.escape(w) for w in en.lower().split()) + r"\b"
        if re.search(pat, hay):
            out.append((en, e["term_ar"]))
        if len(out) >= max_terms:
            break
    return out


# ---------------------------------------------------------------------------
# Rolling chapter summary
# ---------------------------------------------------------------------------

class SummaryTracker:
    """One English paragraph per chapter, refreshed every N completed chunks."""

    def __init__(
        self,
        client: OllamaClient,
        path: Path,
        period: int = 4,
        max_items: int = 12,
    ) -> None:
        self.client = client
        self.period = max(1, period)
        self.max_items = max_items
        self.path = path
        self.state: dict[str, Any] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self.state = json.load(f)

    def summary_for(self, chapter: str) -> str:
        return self.state.get(chapter, {}).get("summary", "")

    def note_completion(
        self, chapter: str, source: str, translation: str
    ) -> None:
        entry = self.state.setdefault(
            chapter, {"items": [], "since-refresh": 0, "summary": ""}
        )
        entry["items"].append({"src": source[:800], "tr": translation[:800]})
        if len(entry["items"]) > self.max_items:
            entry["items"] = entry["items"][-self.max_items:]
        entry["since-refresh"] += 1
        if entry["since-refresh"] >= self.period:
            self._refresh(chapter, entry)
            entry["since-refresh"] = 0
        self.save()

    def _refresh(self, chapter: str, entry: dict) -> None:
        corpus = "\n---\n".join(it["src"] for it in entry["items"])
        prompt = (
            "Summarize in at most 3 sentences the main topics discussed in "
            "this part of a technical book chapter. Plain text only.\n\n" + corpus
        )
        try:
            summary = self.client.generate(prompt) or ""
        except Exception as exc:
            logger.warning("chapter-summary refresh failed: %s", exc)
            summary = entry.get("summary", "")
        entry["summary"] = summary.strip()[:600]
        if self.client.model:
            entry["model"] = self.client.model

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


# ---------------------------------------------------------------------------
# Chunk translation
# ---------------------------------------------------------------------------

def translate_chunk(
    chunk: dict[str, Any],
    client: OllamaClient,
    glossary_entries: list[dict],
    drafts_dir: Path,
    *,
    prompt_version: str,
    glossary_version: str,
    summaries: SummaryTracker,
    max_units: int | None = None,
    emphasis: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Translate one chunk; skip when a fresh draft already exists.

    ``emphasis`` is the M2 compliance re-queue hook: term pairs that the
    compliance audit flagged. Adding them to the cache-key source naturally
    invalidates the stale draft and forces a keyed re-translation.
    """
    drafts_dir.mkdir(parents=True, exist_ok=True)
    chunk_id = chunk["chunk_id"]
    source = json.dumps(
        {"section": chunk["section_title"], "units": [u["text"] for u in chunk["units"]]},
        ensure_ascii=False,
    )
    if emphasis:
        source += "\x1eemphasis:" + json.dumps(emphasis, ensure_ascii=False)
    key = cache_key(source, glossary_version, client.model or "", prompt_version)

    meta_path = drafts_dir / f"{chunk_id}.meta.json"
    md_path = drafts_dir / f"{chunk_id}.md"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("cache_key") == key:
            logger.debug("%s: fresh draft exists, skipping", chunk_id)
            return meta

    units = chunk.get("units", [])
    if max_units:
        units = units[:max_units]

    chapter = chunk["section_title"]
    results: list[dict[str, Any]] = []
    started = time.time()
    prior_translation = ""
    for unit in units:
        # glossary restricted to terms appearing in this unit (plus the whole
        # chunk's matched set at low volume would blow context; per-unit only)
        pairs = match_terms(unit["text"], glossary_entries)
        g_lines = glossary_lines(pairs)
        context_parts = []
        summary = summaries.summary_for(chapter)
        if summary:
            context_parts.append("Context so far in this section:\n" + summary)
        if prior_translation:
            context_parts.append("Previous paragraph translation:\n" + prior_translation)
        context = "\n\n".join(context_parts)
        text_out = translate_unit(
            client, unit["text"], g_lines, context, unit.get("kind", "text"),
            emphasis=emphasis,
        )
        restored = restore_placeholders(unit, text_out)
        results.append({
            "uid": unit["uid"],
            "block_id": unit["block_id"],
            "kind": unit["kind"],
            "number": unit.get("number", ""),
            "text": restored,
            "raw": text_out,
        })
        if restored:
            prior_translation = restored
        summaries.note_completion(chapter, unit["text"], restored)

    meta = {
        "chunk_id": chunk_id,
        "cache_key": key,
        "source_hash": chunk.get("source_hash"),
        "model": client.model,
        "prompt_version": prompt_version,
        "glossary_version": glossary_version,
        "section": chapter,
        "pages": chunk.get("pages", []),
        "units": results,
        "duration_sec": round(time.time() - started, 2),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path.write_text(draft_markup(chunk, results), encoding="utf-8")
    return meta


def translate_unit(
    client: OllamaClient,
    text: str,
    glossary_lines_: list[str],
    context: str,
    kind: str,
    emphasis: list[tuple[str, str]] | None = None,
) -> str:
    kind_instr = {
        "title": "a section heading",
        "caption": "an image or figure caption",
        "list_item": "one list item",
    }.get(kind, "a paragraph from a technical book")
    user = []
    if glossary_lines_:
        user.append("Terminology glossary:\n" + "\n".join(glossary_lines_) + "\n")
    if emphasis:
        from book_harness.qa import source_matches

        relevant = [
            (en, ar) for en, ar in emphasis
            if source_matches(en, text) and ar
        ]
        if relevant:
            user.append(
                "MANDATORY terminology (previous translations violated these; "
                "every listed Arabic term MUST appear exactly as written):\n"
                + "\n".join(
                    GLOSSARY_TEMPLATE.format(en=en, ar=ar) for en, ar in relevant
                ) + "\n"
            )
    if context:
        user.append(context + "\n")
    user.append(f"Translate into Arabic ({kind_instr}):\n{text}")
    return client.generate("\n\n".join(user), system=SYSTEM_PROMPT)


def restore_placeholders(unit: dict, translated: str) -> str:
    """Replace tokens that survived translation with their original spans."""
    placeholders = unit.get("placeholders", {}) or {}
    if not placeholders:
        return translated
    missing = []
    by_token: dict[str, str] = {}
    out = translated
    for token, original in find_tokens(translated, placeholders):
        by_token[token] = original
    for token in by_token:
        out = out.replace(token, by_token[token])
    for token in placeholders:
        if token not in translated:
            missing.append(token)
    if missing:
        logger.debug(
            "placeholders lost in translation for unit %s: %s", unit.get("uid"), missing
        )
    return out


def find_tokens(text: str, placeholders: dict[str, str]) -> list[tuple[str, str]]:
    found = []
    for m in _TOKEN_RE.finditer(text):
        token = m.group(0)
        if token in placeholders:
            found.append((token, placeholders[token]))
    return found


def draft_markup(chunk: dict, results: list[dict[str, Any]]) -> str:
    """Human-readable preview of the translated draft."""
    lines = [f"<!-- {chunk['chunk_id']}: {chunk['section_title']} -->"]
    for r in results:
        kind = r["kind"]
        if kind == "title":
            lines.append(f"\n## {r['number'] + ' ' if r['number'] else ''}{r['text']}\n")
        elif kind == "list_item":
            lines.append(f"- {r['text']}")
        elif kind == "caption":
            lines.append(f"*{r['text']}*")
        else:
            lines.append(r["text"] + "\n")
    return "\n".join(lines).strip() + "\n"
