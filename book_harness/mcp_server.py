"""MCP stdio server (M3.5, PLAN.md task 3.5.1) — thin shell over
``book_harness.mcp``.

NOTE: this lives inside the ``book_harness`` package (not a top-level
``mcp/``) on purpose — the official ``mcp`` SDK may be installed in the same
environment, and a top-level package would shadow it.

Implements the Model Context Protocol over stdio the same way the M4 review
UI stays dependency-free: newline-delimited JSON-RPC 2.0, no third-party
SDK. Supported requests:

* ``initialize``      → server info + tools capability
* ``ping``            → ``{}``
* ``tools/list``      → the four tools with JSON-Schema input schemas
* ``tools/call``      → dispatch into ``book_harness.mcp`` and return the
                        result as a JSON text block; Python exceptions become
                        tool errors (``isError: true``) per the MCP contract

Notifications (messages without ``id``) never get a response.

Run with::

    book-harness-mcp                 # installed console script, or
    python -m book_harness.mcp_server

Clients spawn this process and talk JSON over stdin/stdout, e.g.::

    {"jsonrpc": "2.0", "id": 1, "method": "initialize",
     "params": {"protocolVersion": "2024-11-05", "capabilities": {}}}
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterable, TextIO

import book_harness as _pkg
from book_harness import config
from book_harness import mcp as lib

SERVER_NAME = "book-harness"
SERVER_VERSION = getattr(_pkg, "__version__", "dev")
#: Newest protocol versions this server can serve. If the client requests a
#: compatible version it is echoed; anything unknown falls back to the
#: broadest baseline.
SUPPORTED_PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

# JSON-RPC / MCP error codes
PARSE_ERROR = -32700
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

INITIALIZED_NOTIFICATION = "notifications/initialized"


def _protocol_version(requested: Any) -> str:
    if isinstance(requested, str) and requested in SUPPORTED_PROTOCOL_VERSIONS:
        return requested
    return DEFAULT_PROTOCOL_VERSION


# ---------------------------------------------------------------------------
# Tool specs (name, description, JSON-Schema input schema, call function)
# ---------------------------------------------------------------------------

_DOMAINS = sorted(d for d in config.DOMAIN_DICTIONARIES if d != "*")

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "lookup_term",
        "description": (
            "Look up a technical term in the certified arabterm dictionaries "
            "and return translation citations (Arabic term, source dictionary, "
            "Wikidata ID, printed page, archive.org scan URL). Pass a "
            "domain to filter through the same domain→dictionary whitelist "
            "the translation pipeline uses; pass lang=ar for reverse lookup. "
            "Returns hits as JSON."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "The term to look up (English or Arabic).",
                },
                "lang": {
                    "type": "string",
                    "enum": ["en", "ar"],
                    "default": "en",
                    "description": "Which column to search: 'en' (default) "
                                   "or 'ar' for a reverse Arabic lookup.",
                },
                "domain": {
                    "type": "string",
                    "enum": _DOMAINS,
                    "description": "Optional domain key; restricts results to "
                                   "dictionaries whitelisted for it.",
                },
            },
            "required": ["term"],
        },
        "call": lambda args: lib.lookup_term(
            args["term"],
            args.get("lang", "en"),
            args.get("domain"),
        ),
    },
    {
        "name": "get_glossary",
        "description": (
            "Return a book run's cited glossary (glossary.yaml): version, "
            "entries with certified/llm_chosen/uncertified status, source "
            "dictionary, Wikidata ID, page and scan URL. Optionally narrow "
            "to one term (matches the English or Arabic form)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "book_id": {
                    "type": "string",
                    "description": "Run directory name under the runs root.",
                },
                "term": {
                    "type": "string",
                    "description": "Optional single-term filter.",
                },
                "run_root": {
                    "type": "string",
                    "description": "Optional runs root directory "
                                   f"(env {lib.RUNS_ENV}, default ./runs).",
                },
            },
            "required": ["book_id"],
        },
        "call": lambda args: lib.get_glossary(
            args["book_id"],
            args.get("term"),
            run_root=args.get("run_root"),
        ),
    },
    {
        "name": "translate_chunk",
        "description": (
            "Translate one unit of English technical text to Arabic through "
            "the local model server (Ollama by default; an OpenAI-compatible "
            "endpoint is used when BOOK_HARNESS_API=openai or the host URL "
            "ends in /v1), injecting a terminology glossary (Hy-MT2 "
            "official template) and optional prior context, exactly like "
            "the pipeline does. Only glossary pairs actually occurring in "
            "the text are injected. glossary items accept {en, ar} or "
            "{term_en, term_ar}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string",
                         "description": "English text to translate."},
                "glossary": {
                    "type": "array",
                    "description": "Terminology pairs to inject.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "en": {"type": "string"},
                            "ar": {"type": "string"},
                        },
                    },
                },
                "context": {
                    "type": "string",
                    "description": "Optional prior-paragraph / chapter "
                                   "summary context.",
                },
                "model": {
                    "type": "string",
                    "description": "Optional Ollama model override "
                                   "(default: configured preference list).",
                },
                "kind": {
                    "type": "string",
                    "enum": ["text", "title", "caption", "list_item"],
                    "default": "text",
                    "description": "Unit kind; tweaks the instruction.",
                },
            },
            "required": ["text", "glossary"],
        },
        "call": lambda args: lib.translate_chunk(
            args["text"],
            args["glossary"],
            args.get("context", ""),
            model=args.get("model"),
            kind=args.get("kind", "text"),
        ),
    },
    {
        "name": "list_runs",
        "description": (
            "List book translation runs under the runs root: per-run stage "
            "statuses from state.json, chunk/translated-draft counts, "
            "glossary version and whether book.md was assembled."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_root": {
                    "type": "string",
                    "description": "Optional runs root directory "
                                   f"(env {lib.RUNS_ENV}, default ./runs).",
                },
            },
        },
        "call": lambda args: lib.list_runs(args.get("run_root")),
    },
]

_TOOLS_BY_NAME: dict[str, dict[str, Any]] = {t["name"]: t for t in _TOOLS}


def tool_specs() -> list[dict[str, Any]]:
    """tools/list payload: everything except the callable."""
    return [
        {k: v for k, v in t.items() if k != "call"}
        for t in _TOOLS
    ]


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """MCP tools/call result with JSON-encoded text content."""
    spec = _TOOLS_BY_NAME.get(name)
    if spec is None:
        raise KeyError(f"unknown tool: {name}")
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    result = spec["call"](arguments)
    return {
        "content": [{
            "type": "text",
            "text": result if isinstance(result, str) else json.dumps(
                result, ensure_ascii=False, indent=2
            ),
        }],
    }


# ---------------------------------------------------------------------------
# JSON-RPC dispatch
# ---------------------------------------------------------------------------

def _result(msg_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str,
           data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


def handle_message(msg: Any) -> dict[str, Any] | None:
    """Handle one parsed JSON-RPC message; None for notifications."""
    if not isinstance(msg, dict):
        return _error(None, PARSE_ERROR, "each message must be a JSON object")
    if msg.get("jsonrpc") != "2.0":
        return _error(msg.get("id"), PARSE_ERROR, 'missing jsonrpc: "2.0"')

    method = msg.get("method")
    if "id" not in msg:
        # notifications (initialized, cancelled, …) never get a response
        return None
    msg_id = msg["id"]
    params = msg.get("params") or {}

    if method == "initialize":
        return _result(msg_id, {
            "protocolVersion": _protocol_version(params.get("protocolVersion")),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        })
    if method == "ping":
        return _result(msg_id, {})
    if method == "tools/list":
        return _result(msg_id, {"tools": tool_specs()})
    if method == "tools/call":
        name = params.get("name")
        if name not in _TOOLS_BY_NAME:
            return _error(msg_id, INVALID_PARAMS, f"unknown tool: {name}",
                          data={"known": sorted(_TOOLS_BY_NAME)})
        try:
            result = call_tool(name, params.get("arguments") or {})
        except (KeyError, ValueError, TypeError) as exc:
            return _result(msg_id, {
                "content": [{"type": "text", "text": f"error: {exc}"}],
                "isError": True,
            })
        except FileNotFoundError as exc:
            return _result(msg_id, {
                "content": [{"type": "text", "text": f"error: {exc}"}],
                "isError": True,
            })
        except Exception as exc:            # model/network/parse failures
            return _result(msg_id, {
                "content": [{"type": "text", "text": f"error: {exc}"}],
                "isError": True,
            })
        return _result(msg_id, result)
    return _error(msg_id, METHOD_NOT_FOUND, f"method not found: {method}",
                  data={"supported": ["initialize", "ping", "tools/list",
                                      "tools/call"]})


def main(inp: Iterable[str] | None = None, out: TextIO | None = None) -> int:
    """Stdio loop: one JSON message per line, one response per request."""
    out = out or sys.stdout
    for line in (inp if inp is not None else sys.stdin):
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit(out, _error(None, PARSE_ERROR, f"invalid JSON: {exc}"))
            continue
        response = handle_message(msg)
        if response is not None:
            _emit(out, response)
    return 0


def _emit(out: TextIO, obj: dict[str, Any]) -> None:
    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    out.flush()


if __name__ == "__main__":
    sys.exit(main())
