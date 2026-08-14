"""Reference adapter: Anthropic / OpenAI message lists -> canonical JSONL.

This is the worked example the adapter-authoring skill points at. It is a real
adapter, tested and shipped, not pseudocode — but more importantly it is a
demonstration of the decisions every adapter has to make:

* **Linking a call to its result.** In both formats the call and its result are
  in *different messages*, joined by an id. Getting that join right is the
  whole job; everything else is renaming.
* **Absence is meaningful.** A ``tool_use`` block with no matching
  ``tool_result`` must produce a record with **no** ``result`` key, because
  that is what makes ``missing_tool_result`` fire. Emitting ``None`` would
  claim the tool returned null.
* **Leaving evidence raw.** OpenAI puts tool arguments in a JSON *string*. If
  it does not parse, we pass the string through unchanged so
  ``malformed_tool_call`` can report it, rather than quietly dropping the call.
* **Not inventing signals.** ``explicit_error`` is set only when the source
  actually carries an error flag. Setting it to ``False`` on success would
  disable all text-based failure decoding downstream.

Both input shapes are a JSON file containing either one conversation or a list
of them. A conversation is ``{"messages": [...]}`` with an optional ``"tools"``
array and optional ``"id"`` / ``"task"`` metadata.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["adapt_conversation", "adapt_file", "adapt_many", "detect_format"]

SCHEMA_VERSION = "insight-trace/v1"


# -- format detection ------------------------------------------------------


def detect_format(conversations: Sequence[Mapping[str, Any]]) -> str:
    """Guess whether these are Anthropic-style or OpenAI-style messages.

    Anthropic puts ``tool_use``/``tool_result`` blocks inside ``content``
    lists; OpenAI puts ``tool_calls`` on the assistant message and replies with
    ``role: "tool"``.
    """

    for conversation in conversations:
        for message in conversation.get("messages", []):
            if message.get("tool_calls") or message.get("role") == "tool":
                return "openai"
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, Mapping) and block.get("type") in {
                        "tool_use",
                        "tool_result",
                    }:
                        return "anthropic"
    # No tool activity either way; the Anthropic reader handles plain text.
    return "anthropic"


# -- shared helpers --------------------------------------------------------


def _text_of(content: Any) -> str:
    """Flatten a message's content into plain text."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return ""


def _result_payload(raw: Any, *, is_error: bool | None = None) -> Any:
    """Normalise a tool result into the canonical shape.

    Text always lands under ``content``. This is not cosmetic: IA3's decoder
    unwraps ``content`` and nothing else, so text placed under ``output`` or
    ``message`` would be invisible to it while still being decoded by IA2.
    """

    if isinstance(raw, str):
        payload: dict[str, Any] = {"content": raw}
    elif isinstance(raw, list):
        payload = {"content": _text_of(raw) or json.dumps(raw, ensure_ascii=False)}
    elif isinstance(raw, Mapping):
        payload = dict(raw)
        if "content" not in payload:
            # Preserve the original structure but surface something textual
            # under the key IA3 reads.
            for key in ("output", "message", "text", "result", "error"):
                if isinstance(payload.get(key), str):
                    payload["content"] = payload[key]
                    break
            else:
                payload["content"] = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    else:
        payload = {"content": "" if raw is None else json.dumps(raw, ensure_ascii=False)}

    if is_error:
        payload.setdefault("isError", True)
    return payload


def _tool_catalog(conversation: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract a catalog from a conversation's declared tools.

    Both APIs ship the tool definitions alongside the conversation, which is
    the single highest-value thing an adapter can capture: without it six
    IA3 contract rules abstain.
    """

    tools = conversation.get("tools")
    if not isinstance(tools, list) or not tools:
        return None

    catalog: dict[str, Any] = {}
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        # OpenAI: {"type": "function", "function": {"name", "parameters"}}
        if isinstance(tool.get("function"), Mapping):
            function = tool["function"]
            name = function.get("name")
            schema = function.get("parameters")
        else:
            # Anthropic: {"name", "input_schema"}
            name = tool.get("name")
            schema = tool.get("input_schema") or tool.get("parameters")
        if not name:
            continue
        # None means "known tool, schema unavailable": enables unknown_tool
        # while abstaining from argument checks.
        catalog[str(name)] = dict(schema) if isinstance(schema, Mapping) else None
    return catalog or None


def _finish(
    conversation: Mapping[str, Any],
    trace_id: str,
    calls: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    orphans: list[dict[str, Any]],
    task_text: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "calls": calls,
        "source_pointer": {"conversation_id": conversation.get("id", trace_id)},
    }
    if steps:
        record["steps"] = steps
    if task_text:
        record["task_text"] = task_text
    if orphans:
        record["orphan_results"] = orphans

    catalog = _tool_catalog(conversation)
    if catalog:
        record["tool_catalog"] = catalog

    # Only forward metadata the source actually provides. Inventing a
    # logical_case_id would overstate independent-case counts downstream.
    for source_key, target_key in (
        ("logical_case_id", "logical_case_id"),
        ("case_id", "logical_case_id"),
        ("verdict", "observed_verdict"),
        ("observed_verdict", "observed_verdict"),
        ("cost", "cost"),
    ):
        if conversation.get(source_key) is not None and target_key not in record:
            record[target_key] = conversation[source_key]

    return record


# -- Anthropic -------------------------------------------------------------


def _adapt_anthropic(conversation: Mapping[str, Any], trace_id: str) -> dict[str, Any]:
    messages = conversation.get("messages", [])

    # Pass 1: collect tool_result blocks by the id they answer, so a call can
    # be matched to a result that appears in a later message.
    results: dict[str, dict[str, Any]] = {}
    result_counts: dict[str, int] = {}
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if not isinstance(block, Mapping) or block.get("type") != "tool_result":
                continue
            use_id = str(block.get("tool_use_id", ""))
            result_counts[use_id] = result_counts.get(use_id, 0) + 1
            results.setdefault(
                use_id,
                {
                    "payload": _result_payload(
                        block.get("content"), is_error=block.get("is_error") is True
                    ),
                    "is_error": block.get("is_error"),
                    "pointer": {"message_index": message_index, "block_index": block_index},
                },
            )

    calls: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    task_text = ""
    pending_user_text = ""
    matched: set[str] = set()

    for message_index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")

        if role == "user":
            text = _text_of(content)
            if text:
                pending_user_text = text
                task_text = task_text or text
                steps.append(
                    {"step_index": len(steps), "step_type": "user", "name": "message",
                     "content": text}
                )
            continue

        if role != "assistant":
            continue

        reasoning = _text_of(content)
        if reasoning:
            steps.append(
                {"step_index": len(steps), "step_type": "agent", "name": "message",
                 "content": reasoning}
            )

        if not isinstance(content, list):
            continue

        for block_index, block in enumerate(content):
            if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                continue
            use_id = str(block.get("id") or f"{trace_id}#{len(calls)}")
            call: dict[str, Any] = {
                "call_id": use_id,
                "call_index": len(calls),
                "tool_name": str(block.get("name", "")),
                "arguments": block.get("input"),
                "source_pointer": {
                    "message_index": message_index,
                    "block_index": block_index,
                    "tool_use_id": use_id,
                },
            }
            if pending_user_text:
                call["prior_user_text"] = pending_user_text

            found = results.get(use_id)
            if found is not None:
                matched.add(use_id)
                call["result"] = found["payload"]
                count = result_counts.get(use_id, 1)
                if count > 1:
                    call["result_count"] = count
                if found["is_error"] is True:
                    # Only assert the flag when the source really set it.
                    call["explicit_error"] = True
                    call["outcome_marker"] = "tool_result_is_error"
            # else: leave "result" absent -> MISSING -> missing_tool_result

            calls.append(call)
            steps.append(
                {"step_index": len(steps), "step_type": "tool",
                 "name": str(block.get("name", ""))}
            )

    orphans = [
        {
            "result_id": use_id,
            "content": found["payload"].get("content"),
            "source_pointer": found["pointer"],
        }
        for use_id, found in results.items()
        if use_id not in matched
    ]

    return _finish(conversation, trace_id, calls, steps, orphans,
                   conversation.get("task") or task_text)


# -- OpenAI ----------------------------------------------------------------


def _adapt_openai(conversation: Mapping[str, Any], trace_id: str) -> dict[str, Any]:
    messages = conversation.get("messages", [])

    results: dict[str, dict[str, Any]] = {}
    result_counts: dict[str, int] = {}
    for message_index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id", ""))
        result_counts[call_id] = result_counts.get(call_id, 0) + 1
        results.setdefault(
            call_id,
            {
                "payload": _result_payload(message.get("content")),
                "pointer": {"message_index": message_index},
            },
        )

    calls: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    task_text = ""
    pending_user_text = ""
    matched: set[str] = set()

    for message_index, message in enumerate(messages):
        role = message.get("role")

        if role in {"user", "system"}:
            text = _text_of(message.get("content"))
            if text and role == "user":
                pending_user_text = text
                task_text = task_text or text
                steps.append(
                    {"step_index": len(steps), "step_type": "user", "name": "message",
                     "content": text}
                )
            continue

        if role != "assistant":
            continue

        reasoning = _text_of(message.get("content"))
        if reasoning:
            steps.append(
                {"step_index": len(steps), "step_type": "agent", "name": "message",
                 "content": reasoning}
            )

        for position, tool_call in enumerate(message.get("tool_calls") or []):
            if not isinstance(tool_call, Mapping):
                continue
            function = tool_call.get("function") or {}
            call_id = str(tool_call.get("id") or f"{trace_id}#{len(calls)}")

            # OpenAI serialises arguments as a JSON string. Parse it when the
            # model produced valid JSON; when it did not, pass the raw string
            # through so malformed_tool_call can report the real defect.
            raw_arguments = function.get("arguments")
            if isinstance(raw_arguments, str):
                try:
                    arguments: Any = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = raw_arguments
            else:
                arguments = raw_arguments

            call: dict[str, Any] = {
                "call_id": call_id,
                "call_index": len(calls),
                "tool_name": str(function.get("name", "")),
                "arguments": arguments,
                "source_pointer": {
                    "message_index": message_index,
                    "tool_call_position": position,
                    "tool_call_id": call_id,
                },
            }
            if pending_user_text:
                call["prior_user_text"] = pending_user_text

            found = results.get(call_id)
            if found is not None:
                matched.add(call_id)
                call["result"] = found["payload"]
                count = result_counts.get(call_id, 1)
                if count > 1:
                    call["result_count"] = count

            calls.append(call)
            steps.append(
                {"step_index": len(steps), "step_type": "tool",
                 "name": str(function.get("name", ""))}
            )

    orphans = [
        {
            "result_id": call_id,
            "content": found["payload"].get("content"),
            "source_pointer": found["pointer"],
        }
        for call_id, found in results.items()
        if call_id not in matched
    ]

    return _finish(conversation, trace_id, calls, steps, orphans,
                   conversation.get("task") or task_text)


# -- public API ------------------------------------------------------------


def adapt_conversation(
    conversation: Mapping[str, Any],
    *,
    fmt: str = "auto",
    trace_id: str | None = None,
    index: int = 0,
    trace_id_prefix: str = "trace",
) -> dict[str, Any]:
    """Convert one conversation into one canonical trace record."""

    if fmt == "auto":
        fmt = detect_format([conversation])
    resolved_id = str(trace_id or conversation.get("id") or f"{trace_id_prefix}-{index:05d}")

    if fmt == "openai":
        return _adapt_openai(conversation, resolved_id)
    if fmt == "anthropic":
        return _adapt_anthropic(conversation, resolved_id)
    raise ValueError(f"unknown message format {fmt!r}; expected 'anthropic', 'openai' or 'auto'")


def adapt_many(
    conversations: Iterable[Mapping[str, Any]],
    *,
    fmt: str = "auto",
    trace_id_prefix: str = "trace",
) -> list[dict[str, Any]]:
    conversations = list(conversations)
    if fmt == "auto":
        fmt = detect_format(conversations)
    return [
        adapt_conversation(c, fmt=fmt, index=i, trace_id_prefix=trace_id_prefix)
        for i, c in enumerate(conversations)
    ]


def _load_conversations(path: str | Path) -> list[Mapping[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, Mapping):
        if isinstance(data.get("conversations"), list):
            return data["conversations"]
        return [data]
    raise ValueError(f"{path}: expected a conversation object or a list of them")


def adapt_file(
    path: str | Path,
    *,
    fmt: str = "auto",
    tool_catalog_path: str | Path | None = None,
    trace_id_prefix: str = "trace",
) -> list[dict[str, Any]]:
    """Convert a file of conversations into canonical records."""

    conversations = _load_conversations(path)
    records = adapt_many(conversations, fmt=fmt, trace_id_prefix=trace_id_prefix)

    if tool_catalog_path is not None:
        catalog = json.loads(Path(tool_catalog_path).read_text(encoding="utf-8"))
        for record in records:
            record.setdefault("tool_catalog", catalog)

    return records
