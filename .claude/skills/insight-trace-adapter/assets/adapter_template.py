#!/usr/bin/env -S uv run
"""Adapter: <your source> -> Insight Agent canonical JSONL (insight-trace/v1).

Verify loop — run these after every change, never batch them:

    ./adapters/<your source>.py > traces.jsonl
    uv run insight-agent validate traces.jsonl     # must exit 0
    uv run insight-agent coverage traces.jsonl     # what can actually fire?
    uv run insight-agent run-ia3 traces.jsonl -o out

Then open out/ia3/cards.json and check three findings against the raw source by
hand. A rule that fires on 100% of calls is an adapter bug, not a discovery.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from typing import Any

SCHEMA_VERSION = "insight-trace/v1"


def load_corpus_context(path: str) -> dict[str, Any]:
    """Corpus-level data that individual records need.

    Tool definitions often live once at the top of an export rather than on
    every record, so they are read here and threaded into `to_canonical`.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    tools = data.get("tools") if isinstance(data, dict) else None
    return {"tool_catalog": {t["name"]: t.get("parameters") for t in tools} if tools else None}


def iter_source_records(path: str) -> Iterator[Any]:
    """Yield one source record per trace. Adjust to your input format."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    yield from (data if isinstance(data, list) else [data])


def to_canonical(source: Any, index: int, context: dict[str, Any]) -> dict[str, Any]:
    """Map one source record to one canonical trace.

    Start with the required fields only, get `validate` to exit 0, then add one
    optional field at a time and re-check `coverage`.
    """
    trace_id = str(source.get("id") or f"trace-{index:05d}")

    calls = []
    for position, raw in enumerate(source.get("tool_calls", [])):
        call: dict[str, Any] = {
            # Required four:
            "call_id": str(raw.get("id") or f"{trace_id}#{position}"),
            "call_index": position,
            "tool_name": str(raw["name"]),
            "arguments": raw.get("arguments"),  # emit RAW, do not re-parse
        }
        # Omit "result" entirely when no result was recorded: absence is what
        # fires missing_tool_result. "result": None means the tool returned null.
        # Textual results go under "content" -- IA3 unwraps nothing else.
        if "result" in raw:
            found = raw["result"]
            call["result"] = {"content": found} if isinstance(found, str) else found
        calls.append(call)

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "calls": calls,
        # Add next, in this order:
        #   "logical_case_id": "...",       makes card eligibility honest
        #   "source_pointer": {...},        makes evidence reopenable
        #   "steps": [...],                 real IA2 trajectory tokens
    }
    # Highest-value optional field: unlocks six argument-contract rules.
    if context.get("tool_catalog"):
        record["tool_catalog"] = context["tool_catalog"]
    return record


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} SOURCE", file=sys.stderr)
        return 2
    context = load_corpus_context(argv[1])
    for index, source in enumerate(iter_source_records(argv[1])):
        record = to_canonical(source, index, context)
        print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
