"""Concrete InsightsGeneration stage backed by an LLM Analyst.

Evidence streams surface candidate Problems without authoring conclusions. The
Analyst synthesizes those Problems into customer-facing Insights and can fetch
their supporting normalized traces when it needs more context.

Three consequences of adding a model to an otherwise deterministic flow are
handled here:

* **Nothing is validated after the fact.** The prompt states the citation
  constraint; the CLI reports cited IDs that are neither in the evidence nor
  among the traces the model fetched, and warns loudly when one does not exist
  in the corpus at all. It does not gate. Treat the output as authored, not
  proven.
* **The prompt is a versioned artifact.** It lives in ``prompts/`` and every
  run writes the assembled copy beside its result, so a change to it shows up
  in the outputs rather than silently altering behaviour.
* **Cost is driven by what the model opens, not by corpus size.** The traces
  worth opening are the ones IA2 flagged for large outputs, which are exactly
  the expensive ones — a single trace can run to well over 100k tokens. Results
  are trimmed per call and the loop is capped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Any

import litellm

from insight_agent.evidence_streams.contracts import EvidenceStreamResult
from insight_agent.traces import Trace, TraceSnapshot

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_TOOL_ROUNDS",
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT_VERSION",
    "InsightsGeneration",
    "InsightsGenerationError",
    "InsightsGenerationResult",
    "ResponseParseError",
]

#: litellm takes ``provider/model``. Claude Opus 5 is the default; any litellm
#: model string works via ``--model``.
DEFAULT_MODEL = "anthropic/claude-opus-5"

#: Generous on purpose. The input is small (~3k tokens) and a truncated response
#: is an unparseable one, which wastes the whole call.
DEFAULT_MAX_TOKENS = 16_000

#: The default prompt follows the source revision; its name is recorded in run
#: metadata alongside the assembled text.
DEFAULT_PROMPT_VERSION = "analyst"

#: A prompt containing this marker gets the evidence substituted in place, and
#: the user turn becomes a short kickoff. Without it the template is the system
#: prompt and the evidence is the user turn (v1/v2 behaviour).
EVIDENCE_PLACEHOLDER = "{evidence}"

#: A prompt naming this tool is asking for the trace-lookup loop. The prompt
#: declares its own requirement rather than the caller having to remember a
#: flag that, if forgotten, would promise the model tools it does not have.
TRACE_LOOKUP_TOOL = "fetch_traces"

#: Per-result cap when a trace is fetched. The traces an Analyst most wants to
#: open are the ones IA2 flagged for large outputs — which are exactly the ones
#: that would blow the context if returned whole. Single traces well over 100k
#: tokens are routine.
DEFAULT_RESULT_CHARS = 2_000

#: Ceiling on one tool response, whatever the model asks for.
MAX_TOOL_RESPONSE_CHARS = 120_000

#: Stops a runaway loop. Each round trip re-sends the whole conversation.
DEFAULT_MAX_TOOL_ROUNDS = 8


class InsightsGenerationError(RuntimeError):
    """Base class for InsightsGeneration failures."""


class ResponseParseError(InsightsGenerationError):
    """The model replied, but not with parseable Insights.

    Carries the raw text so the caller can persist it for inspection instead of
    discarding a response that was already paid for.
    """

    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


def _prompts_dir():
    return files("insight_agent.insights_generation").joinpath("prompts")


@lru_cache(maxsize=8)
def _prompt_template(version: str = DEFAULT_PROMPT_VERSION) -> str:
    """One named Analyst prompt, read from the packaged resources."""

    try:
        return _prompts_dir().joinpath(f"{version}.md").read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        known = ", ".join(_available_prompt_versions()) or "(none found)"
        raise InsightsGenerationError(
            f"unknown prompt version {version!r}; available: {known}"
        ) from exc


def _available_prompt_versions() -> list[str]:
    """Every prompt shipped with the package, oldest name first."""

    return sorted(p.name[: -len(".md")] for p in _prompts_dir().iterdir() if p.name.endswith(".md"))


@dataclass(frozen=True)
class _AnalystRequest:
    """Everything the Analyst is allowed to see."""

    #: Name of the agent under test, substituted into the prompt.
    agent: str

    #: Generic candidate problems from every configured evidence stream.
    evidence: Sequence[EvidenceStreamResult] = ()

    #: Trace-loader summary, for corpus-level grounding.
    corpus: Mapping[str, Any] = field(default_factory=dict)

    #: Which packaged prompt to use. Recorded in `run.json` so a given
    #: `insights.json` can always be traced back to the text that produced it.
    prompt_version: str = DEFAULT_PROMPT_VERSION


@dataclass(frozen=True)
class InsightsGenerationResult:
    """One Analyst run."""

    insights: list[dict[str, Any]]
    model: str
    system_prompt: str
    user_prompt: str
    prompt_version: str = DEFAULT_PROMPT_VERSION
    raw_response: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)

    #: How much raw evidence the model chose to open. Zero on a prompt without
    #: the lookup tool, and a useful signal on one that has it.
    tool_calls: int = 0
    traces_fetched: tuple[str, ...] = ()

    @property
    def cited_trace_ids(self) -> set[str]:
        return {t for insight in self.insights for t in insight.get("trace_ids", [])}


class InsightsGeneration:
    """Generate Insights from a normalized snapshot and evidence-stream results."""

    def __init__(
        self,
        *,
        snapshot: TraceSnapshot,
        evidence: Sequence[EvidenceStreamResult],
        agent: str,
        corpus: Mapping[str, Any],
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> None:
        """Prepare the concrete stage from generic evidence-stream results."""

        names = [result.stream_name for result in evidence]
        if len(names) != len(set(names)):
            raise InsightsGenerationError("evidence stream names must be unique")

        self._request = _AnalystRequest(
            agent=agent,
            evidence=tuple(evidence),
            corpus=corpus,
            prompt_version=prompt_version,
        )
        self.snapshot = snapshot
        self.evidence = tuple(evidence)
        self.problems_presented = sum(len(result.problems) for result in evidence)

    def build_prompt(self) -> tuple[str, str]:
        return _build_prompt(self._request)

    @property
    def known_trace_ids(self) -> set[str]:
        """Trace IDs included in the evidence presented to the Analyst."""

        return {
            trace_id
            for result in self._request.evidence
            for problem in result.problems
            for trace_id in problem.supporting_trace_ids
        }

    def generate(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_base: str | None = None,
        api_key: str | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        **litellm_kwargs: Any,
    ) -> InsightsGenerationResult:
        return _author_insights(
            self._request,
            model=model,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            snapshot=self.snapshot,
            max_tool_rounds=max_tool_rounds,
            **litellm_kwargs,
        )


def _corpus_summary(corpus: Mapping[str, Any]) -> str:
    if not corpus:
        return ""
    parts = []
    for key, label in (
        ("trace_count", "traces"),
        ("call_count", "tool calls"),
        ("distinct_logical_cases", "distinct logical cases"),
    ):
        if corpus.get(key) is not None:
            parts.append(f"{corpus[key]:,} {label}")
    return ", ".join(parts)


OUTPUT_CONTRACT = """\
Return **only** a JSON array. No prose before or after it, no markdown fence.
Each element has exactly these three fields:

```json
[
  {
    "name": "Short header, a few words",
    "description": "2-4 sentences naming the failure mode, the affected tool or call, and the conditions that trigger it.",
    "trace_ids": ["trace-id-1", "trace-id-2", "trace-id-3"]
  }
]
```

- `name` — a short few-word header.
- `description` — 2 to 4 sentences.
- `trace_ids` — the trace IDs you used as evidence, copied exactly from the
  evidence or from a fetched trace. Never invent or reconstruct one.

Return `[]` if nothing meets the bar."""


def _build_prompt(request: _AnalystRequest) -> tuple[str, str]:
    """Assemble the (system, user) prompt pair.

    The template owns the whole system prompt and says where the evidence goes
    with a ``{evidence}`` marker; the user turn is a short kickoff. The output
    contract is appended to that kickoff only when the template does not state
    one itself — a prompt specifying ``trace_ids`` owns its contract and must
    not get a second, possibly conflicting, copy stapled on.
    """

    template = _prompt_template(request.prompt_version)
    if EVIDENCE_PLACEHOLDER not in template:
        raise InsightsGenerationError(
            f"prompt {request.prompt_version!r} has no {EVIDENCE_PLACEHOLDER} marker, so the "
            "evidence has nowhere to go. Add it where the evidence should appear."
        )
    system = template.replace("{agent}", request.agent)

    sections: list[str] = [f"# Evidence for {request.agent}", ""]

    summary = _corpus_summary(request.corpus)
    if summary:
        sections += [f"Corpus: {summary}.", ""]

    sections += ["---", "", "# Candidate problems", ""]
    if not request.evidence:
        sections.append(
            "No evidence streams ran. Return no Insights rather than inventing a problem."
        )

    for result in request.evidence:
        sections += [
            f"## {result.stream_name}",
            "",
        ]
        if result.problems:
            sections += [
                f"{len(result.problems)} candidate problem(s), as JSON:",
                "",
                "```json",
                json.dumps(
                    [problem.model_dump(mode="json") for problem in result.problems],
                    indent=2,
                    ensure_ascii=False,
                ),
                "```",
            ]
        else:
            sections.append(
                "This stream returned no candidate problems. Do not compensate by lowering "
                "the bar or inferring a problem from the absence of evidence."
            )
        sections += ["", "---", ""]

    kickoff = ["Author Insights for this corpus."]
    if '"trace_ids"' not in template and "`trace_ids`" not in template:
        kickoff += ["", OUTPUT_CONTRACT]

    return system.replace(EVIDENCE_PLACEHOLDER, "\n".join(sections)), "\n".join(kickoff)


# -- trace lookup ----------------------------------------------------------


TRACE_LOOKUP_SCHEMA = {
    "type": "function",
    "function": {
        "name": TRACE_LOOKUP_TOOL,
        "description": (
            "Fetch one or more normalized traces by id, so you can read the actual "
            "span tree, LLM messages, tool inputs and tool outputs rather than relying on the "
            "preprocessed evidence alone. Pass several ids in one call when you want to "
            "compare traces. Large span inputs and outputs are truncated; raise "
            "max_chars_per_result when you need to see a large payload in full."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "trace_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exact trace ids, as they appear in the evidence.",
                },
                "max_chars_per_result": {
                    "type": "integer",
                    "description": (
                        f"Characters of each tool result to return (default "
                        f"{DEFAULT_RESULT_CHARS}). Raise it to inspect large outputs."
                    ),
                },
            },
            "required": ["trace_ids"],
        },
    },
}


def _trim(value: Any, limit: int) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return value
    return f"{text[:limit]}… [truncated: {len(text):,} chars total, showing {limit:,}]"


def _fetch_traces(
    traces_by_id: Mapping[str, Trace],
    trace_ids: Sequence[str],
    *,
    max_chars_per_result: int = DEFAULT_RESULT_CHARS,
) -> dict[str, Any]:
    """Return normalized traces for the model, trimmed to fit a context window.

    Unknown ids are reported rather than silently dropped: a model that
    mistyped an id needs to see that it mistyped it, not an empty result it
    might read as "this trace has no spans".
    """

    limit = max(
        200, min(int(max_chars_per_result or DEFAULT_RESULT_CHARS), MAX_TOOL_RESPONSE_CHARS)
    )
    found, missing, budget = [], [], MAX_TOOL_RESPONSE_CHARS

    for trace_id in trace_ids:
        normalized = traces_by_id.get(trace_id)
        if normalized is None:
            missing.append(trace_id)
            continue
        trace = normalized.model_dump(mode="json")
        if "input" in trace:
            trace["input"] = _trim(trace["input"], limit)
        if "output" in trace:
            trace["output"] = _trim(trace["output"], limit)
        for span in trace["spans"]:
            if "input" in span:
                span["input"] = _trim(span["input"], limit)
            if "output" in span:
                span["output"] = _trim(span["output"], limit)
        rendered = json.dumps(trace, ensure_ascii=False, default=str)
        if len(rendered) > budget:
            found.append(
                {
                    "id": trace_id,
                    "error": (
                        "omitted: returning this trace would exceed the response budget. "
                        "Fetch it on its own, or lower max_chars_per_result."
                    ),
                }
            )
            break
        budget -= len(rendered)
        found.append(trace)

    payload: dict[str, Any] = {"traces": found}
    if missing:
        payload["not_found"] = missing
        payload["note"] = (
            "These trace_ids are not in the corpus. Use ids exactly as they appear in "
            "the evidence; do not reconstruct or abbreviate them."
        )
    return payload


def _tool_calls_of(message: Any) -> list[Any]:
    calls = (
        message.get("tool_calls")
        if isinstance(message, Mapping)
        else getattr(message, "tool_calls", None)
    )
    return list(calls or [])


def _call_field(call: Any, *path: str) -> Any:
    node = call
    for key in path:
        node = node.get(key) if isinstance(node, Mapping) else getattr(node, key, None)
        if node is None:
            return None
    return node


def _message_to_dict(message: Any) -> dict[str, Any]:
    """Normalise an assistant message for replay in the next request."""
    if isinstance(message, Mapping):
        return dict(message)
    if hasattr(message, "model_dump"):
        return {k: v for k, v in message.model_dump().items() if v is not None}
    return {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", None),
        "tool_calls": getattr(message, "tool_calls", None),
    }


# -- response parsing ------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def _coerce_insight(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    name = str(value.get("name") or "").strip()
    if not name:
        return None

    raw_ids = value.get("trace_ids")
    if isinstance(raw_ids, str):
        # A single id sent unwrapped. Accept it rather than dropping the
        # Insight; the shape is obvious and the content is what matters.
        raw_ids = [raw_ids]
    elif not isinstance(raw_ids, Sequence):
        raw_ids = []

    trace_ids = []
    for item in raw_ids:
        text = str(item).strip()
        if text and text not in trace_ids:
            trace_ids.append(text)

    return {
        "name": name,
        "description": str(value.get("description") or "").strip(),
        "trace_ids": trace_ids,
    }


def _parse_insights(text: str) -> list[dict[str, Any]]:
    """Extract the Insight list from a model response.

    Deliberately tolerant. There is no validator downstream, and a strict parser
    that rejected a fenced or wrapped array would throw away a response that was
    already paid for. Accepts a bare array, an object with an ``insights`` key,
    either of those inside a ``` fence, or an array embedded in prose.
    """

    if not text or not text.strip():
        raise ResponseParseError("the model returned an empty response", text or "")

    candidates: list[str] = [text.strip()]
    candidates.extend(match.group(1).strip() for match in _FENCE.finditer(text))

    # Last resort: the outermost bracketed span, for a model that wrapped the
    # array in explanatory prose despite being told not to.
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, Mapping):
            for key in ("insights", "Insights", "results"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
            else:
                # A single Insight object returned unwrapped.
                parsed = [parsed] if "name" in parsed else None

        if isinstance(parsed, list):
            return [i for i in (_coerce_insight(item) for item in parsed) if i is not None]

    raise ResponseParseError(
        "could not extract a JSON array of Insights from the model response", text
    )


# -- the call --------------------------------------------------------------


def _extract_message(response: Any) -> Any:
    """Pull the assistant message out of a litellm response.

    litellm normalises providers into the OpenAI response shape, but returns
    either a pydantic model or a dict depending on version and provider.
    """

    if isinstance(response, Mapping):
        choices = response.get("choices") or []
        return (choices[0].get("message") or {}) if choices else {}
    choices = getattr(response, "choices", None) or []
    return getattr(choices[0], "message", None) if choices else None


def _extract_text(response: Any) -> str:
    message = _extract_message(response)
    if message is None:
        return ""
    content = (
        message.get("content")
        if isinstance(message, Mapping)
        else getattr(message, "content", None)
    )
    return str(content or "")


def _extract_usage(response: Any) -> dict[str, Any]:
    usage = (
        response.get("usage") if isinstance(response, Mapping) else getattr(response, "usage", None)
    )
    if usage is None:
        return {}
    if isinstance(usage, Mapping):
        source = usage
    elif hasattr(usage, "model_dump"):
        source = usage.model_dump()
    else:
        source = {
            k: getattr(usage, k)
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            if hasattr(usage, k)
        }
    return {k: v for k, v in source.items() if isinstance(v, (int, float, str))}


def _author_insights(
    request: _AnalystRequest,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_base: str | None = None,
    api_key: str | None = None,
    snapshot: TraceSnapshot,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    **litellm_kwargs: Any,
) -> InsightsGenerationResult:
    """Send the evidence to the model and return the authored Insights.

    ``api_base`` points litellm at an OpenAI-compatible gateway (an internal
    inference endpoint, a proxy, a self-hosted server). Combined with an
    ``openai/`` model prefix it is how any such endpoint is reached.

    No sampling parameters are set. ``temperature``, ``top_p`` and ``top_k`` are
    *removed* on Claude Opus 5 and return a 400 — passing one to be "safe" is
    the single easiest way to break this call. Steer through the prompt instead.
    Callers on a provider that accepts them can pass them explicitly through
    ``litellm_kwargs``.
    """

    system, user = _build_prompt(request)

    # Only forward what was actually supplied: passing api_base=None would stop
    # litellm resolving the provider's own default endpoint.
    if api_base:
        litellm_kwargs["api_base"] = api_base
    if api_key:
        litellm_kwargs["api_key"] = api_key

    # The prompt declares whether it wants trace lookup by naming the tool, so a
    # prompt promising tools can never be run without them.
    wants_tools = TRACE_LOOKUP_TOOL in system
    if wants_tools:
        litellm_kwargs["tools"] = [TRACE_LOOKUP_SCHEMA]
        traces_by_id = {trace.id: trace for trace in snapshot.scan()}
    else:
        traces_by_id = {}

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    usage: dict[str, Any] = {}
    tool_calls_made = 0
    traces_fetched: list[str] = []

    for _ in range(max_tool_rounds if wants_tools else 1):
        response = litellm.completion(
            model=model, max_tokens=max_tokens, messages=messages, **litellm_kwargs
        )
        for key, value in _extract_usage(response).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
            else:
                usage.setdefault(key, value)

        message = _extract_message(response)
        pending = _tool_calls_of(message) if wants_tools else []
        if not pending:
            break

        messages.append(_message_to_dict(message))
        for call in pending:
            tool_calls_made += 1
            name = _call_field(call, "function", "name")
            raw_args = _call_field(call, "function", "arguments") or "{}"
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                arguments = {}

            if name == TRACE_LOOKUP_TOOL:
                ids = [str(t) for t in (arguments.get("trace_ids") or [])]
                traces_fetched.extend(ids)
                payload = _fetch_traces(
                    traces_by_id,
                    ids,
                    max_chars_per_result=arguments.get(
                        "max_chars_per_result", DEFAULT_RESULT_CHARS
                    ),
                )
            else:
                payload = {"error": f"unknown tool {name!r}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _call_field(call, "id"),
                    "name": name,
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                }
            )
    else:
        # Loop ran to exhaustion still asking for tools. Ask once more with the
        # tools withdrawn so the turn ends in an answer rather than a timeout.
        if wants_tools:
            litellm_kwargs.pop("tools", None)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"You have used the {max_tool_rounds} lookups available. "
                        "Author your Insights now from what you have read."
                    ),
                }
            )
            response = litellm.completion(
                model=model, max_tokens=max_tokens, messages=messages, **litellm_kwargs
            )
            for key, value in _extract_usage(response).items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value

    raw = _extract_text(response)
    insights = _parse_insights(raw)

    return InsightsGenerationResult(
        insights=insights,
        model=model,
        system_prompt=system,
        user_prompt=user,
        prompt_version=request.prompt_version,
        raw_response=raw,
        usage=usage,
        tool_calls=tool_calls_made,
        traces_fetched=tuple(dict.fromkeys(traces_fetched)),
    )
