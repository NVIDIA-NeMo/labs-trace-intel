# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load Langfuse v3 traces into the provider-neutral trace contracts."""

from __future__ import annotations

import heapq
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin

from pydantic import JsonValue

from trace_ingest.loaders.trace_loaders import TraceDescription
from trace_ingest.models import (
    UNSET,
    Span,
    SpanKind,
    ToolCall,
    Trace,
    TraceAggregate,
    TraceSnapshot,
)
from trace_ingest.source_links import file_source_url, http_source_url

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.api.resources.commons.types.observations_view import ObservationsView
    from langfuse.api.resources.commons.types.trace_with_details import TraceWithDetails
    from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails

LANGFUSE_DEFAULT_MAX_TRACES = 100

_API_PAGE_SIZE = 100
_LIST_FIELDS = "core"

__all__ = [
    "LANGFUSE_DEFAULT_MAX_TRACES",
    "LangfuseFileTraceConfig",
    "LangfuseFileTraceDescription",
    "LangfuseFileTraceLoader",
    "LangfuseTraceConfig",
    "LangfuseTraceDescription",
    "LangfuseTraceLoader",
    "LangfuseTraceLoadError",
    "validate_langfuse_time_window",
]


class LangfuseTraceLoadError(RuntimeError):
    """A Langfuse source could not be selected or normalized safely."""


class _LangfuseCorpusDescription(TraceDescription):
    """Corpus facts shared by live and file sources."""

    observation_count: int
    unresolved_parent_count: int


class LangfuseTraceDescription(_LangfuseCorpusDescription):
    """Run metadata for a live Langfuse v3 trace query."""

    base_url: str
    filter: str | None
    from_timestamp: str
    to_timestamp: str
    max_traces: int


@dataclass(frozen=True)
class LangfuseTraceConfig:
    """Configuration for loading complete traces through the Langfuse v3 API."""

    from_timestamp: datetime
    to_timestamp: datetime
    base_url: str | None = None
    filter_string: str | None = None
    max_traces: int = LANGFUSE_DEFAULT_MAX_TRACES

    def __post_init__(self) -> None:
        validate_langfuse_time_window(self.from_timestamp, self.to_timestamp)
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")


def validate_langfuse_time_window(from_timestamp: datetime, to_timestamp: datetime) -> None:
    """Require a timezone-aware, nonempty window for CLI and direct API loading."""
    for name, value in (("from_timestamp", from_timestamp), ("to_timestamp", to_timestamp)):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must include a timezone")
    if to_timestamp <= from_timestamp:
        raise ValueError("to_timestamp must be after from_timestamp")


@dataclass
class LangfuseTraceLoader:
    """Load a bounded selection of complete traces from one Langfuse v3 project.

    Langfuse API credentials identify the project. A client can be injected for
    tests; otherwise the optional SDK is imported only when this loader runs.
    """

    config: LangfuseTraceConfig
    client: Langfuse | None = field(default=None, repr=False)
    _snapshot: TraceSnapshot = field(
        default_factory=lambda: TraceSnapshot([]), init=False, repr=False
    )
    _base_url: str | None = field(default=None, init=False, repr=False)

    def load(self) -> TraceSnapshot:
        """Select trace summaries, fetch their complete details, and normalize them."""

        base_url = _resolved_base_url(self.config.base_url)
        client = self.client or _new_langfuse_client(base_url)
        selected = self._select_traces(client)

        traces: list[Trace] = []
        # Query-summary timestamps determine ordering, even if detail timestamps differ.
        for summary in sorted(
            selected, key=lambda trace: (_trace_timestamp(trace), _trace_id(trace))
        ):
            trace_id = _trace_id(summary)
            provider_trace = client.api.trace.get(trace_id)
            trace = _normalize_trace(
                provider_trace,
                source_pointer={
                    "provider": "langfuse",
                    "base_url": base_url,
                    "trace_id": trace_id,
                },
            )
            if provider_trace.html_path:
                trace.source_url = http_source_url(
                    urljoin(base_url.rstrip("/") + "/", provider_trace.html_path)
                )
            traces.append(trace)

        snapshot = TraceSnapshot(traces)
        self._base_url = base_url
        self._snapshot = snapshot
        return snapshot

    def _select_traces(self, client: Langfuse) -> list[TraceWithDetails]:
        selected: dict[str, TraceWithDetails] = {}
        page = 1
        page_size = min(self.config.max_traces, _API_PAGE_SIZE)
        filter_string = _bounded_filter(
            self.config.filter_string,
            from_timestamp=self.config.from_timestamp,
            to_timestamp=self.config.to_timestamp,
        )

        while len(selected) < self.config.max_traces:
            response = client.api.trace.list(
                page=page,
                limit=page_size,
                from_timestamp=self.config.from_timestamp,
                to_timestamp=self.config.to_timestamp,
                order_by="timestamp.desc",
                fields=_LIST_FIELDS,
                filter=filter_string,
            )
            for trace in response.data:
                selected.setdefault(_trace_id(trace), trace)
                if len(selected) == self.config.max_traces:
                    break

            if page >= response.meta.total_pages or len(selected) == self.config.max_traces:
                break
            if not response.data:
                raise LangfuseTraceLoadError(
                    "Langfuse returned an empty trace page before the final page"
                )
            page += 1

        return list(selected.values())

    def describe(self) -> LangfuseTraceDescription:
        """Describe the configured source and most recently loaded corpus."""

        base_url = self._base_url or _resolved_base_url(self.config.base_url)
        return {
            **_describe_corpus(f"langfuse:{base_url}", self._snapshot),
            "base_url": base_url,
            "filter": self.config.filter_string,
            "from_timestamp": self.config.from_timestamp.isoformat(),
            "to_timestamp": self.config.to_timestamp.isoformat(),
            "max_traces": self.config.max_traces,
        }


class LangfuseFileTraceDescription(_LangfuseCorpusDescription):
    """Run metadata for a local native trace-detail export."""

    export_path: str
    export_trace_count: int
    max_traces: int
    truncated: bool


@dataclass(frozen=True)
class LangfuseFileTraceConfig:
    """A native trace-detail JSON/JSONL file or directory of such files."""

    path: Path
    max_traces: int = LANGFUSE_DEFAULT_MAX_TRACES

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.max_traces < 1:
            raise ValueError("max_traces must be at least 1")


@dataclass
class LangfuseFileTraceLoader:
    """Read full v3 trace responses without contacting Langfuse.

    Accepts raw API bodies and the CLI's status/headers/body envelopes. All
    records are validated before selecting the newest complete traces.
    """

    config: LangfuseFileTraceConfig
    _snapshot: TraceSnapshot = field(
        default_factory=lambda: TraceSnapshot([]), init=False, repr=False
    )
    _export_trace_count: int = field(default=0, init=False, repr=False)

    def load(self) -> TraceSnapshot:
        """Validate native exports and reuse live trace normalization."""
        from langfuse.api.resources.commons.types.trace_with_full_details import (
            TraceWithFullDetails,
        )

        export_path = self.config.path.resolve()
        files = (
            sorted(
                p for p in export_path.iterdir() if p.suffix in {".json", ".jsonl"} and p.is_file()
            )
            if export_path.is_dir()
            else [export_path]
        )
        if not files:
            raise LangfuseTraceLoadError(
                f"Langfuse export {export_path} contains no JSON/JSONL files"
            )
        traces: dict[tuple[datetime, str], Trace] = {}
        seen: set[str] = set()
        for path in files:
            contents = path.read_bytes()
            for index, record in enumerate(_parse_trace_export(contents, path), start=1):
                try:
                    provider = TraceWithFullDetails.parse_obj(record)
                    trace_id = _trace_id(provider)
                    if trace_id in seen:
                        raise LangfuseTraceLoadError(f"duplicate trace id {trace_id!r}")
                    seen.add(trace_id)
                    if any(obs.trace_id != trace_id for obs in provider.observations):
                        raise LangfuseTraceLoadError("observation traceId does not match its trace")
                    trace = _normalize_trace(
                        provider,
                        require_complete_tree=True,
                        source_pointer={
                            "provider": "langfuse",
                            "export_path": str(path),
                            "record": index,
                            "trace_id": trace_id,
                        },
                    )
                    trace.source_url = file_source_url(path)
                    traces[(_trace_timestamp(provider), trace_id)] = trace
                except (ValueError, LangfuseTraceLoadError) as error:
                    raise LangfuseTraceLoadError(
                        f"Invalid Langfuse export {path}, record {index}: {error}"
                    ) from error

        snapshot = TraceSnapshot(traces[key] for key in sorted(traces)[-self.config.max_traces :])
        self._export_trace_count = len(traces)
        self._snapshot = snapshot
        return snapshot

    def describe(self) -> LangfuseFileTraceDescription:
        """Describe the last load and its source export."""
        return {
            **_describe_corpus(f"langfuse-export:{self.config.path.resolve()}", self._snapshot),
            "export_path": str(self.config.path.resolve()),
            "export_trace_count": self._export_trace_count,
            "max_traces": self.config.max_traces,
            "truncated": self._export_trace_count > self._snapshot.trace_count,
        }


def _describe_corpus(source: str, snapshot: TraceSnapshot) -> _LangfuseCorpusDescription:
    observation_count = call_count = unresolved_parent_count = 0
    pending = [span for trace in snapshot for span in trace.root_spans]
    while pending:
        span = pending.pop()
        observation_count += 1
        call_count += span.kind is SpanKind.TOOL
        pointer = span.attributes.get("source_pointer")
        unresolved_parent_count += (
            isinstance(pointer, dict) and "unresolved_parent_observation_id" in pointer
        )
        pending.extend(span.children)
    return {
        "source": source,
        "trace_count": snapshot.trace_count,
        "call_count": call_count,
        "distinct_logical_cases": len(
            {str(trace.attributes.get("logical_case_id") or trace.id) for trace in snapshot}
        ),
        "observation_count": observation_count,
        "unresolved_parent_count": unresolved_parent_count,
    }


def _parse_trace_export(contents: bytes, path: Path) -> list[dict[str, object]]:
    try:
        text = contents.decode("utf-8")
        payloads = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else [json.loads(text)]
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LangfuseTraceLoadError(f"Invalid Langfuse JSON export {path}: {error}") from error
    if not payloads:
        raise LangfuseTraceLoadError(f"Langfuse export {path} contains no traces")
    records: list[dict[str, object]] = []
    for index, payload in enumerate(payloads, start=1):
        if isinstance(payload, dict) and "body" in payload:
            if payload.get("status") != 200:
                raise LangfuseTraceLoadError(
                    f"Langfuse export {path}, record {index}: unsuccessful CLI response"
                )
            payload = payload["body"]
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("observations"), list)
            or not all(isinstance(obs, dict) for obs in payload["observations"])
        ):
            raise LangfuseTraceLoadError(
                f"Langfuse export {path}, record {index}: expected a complete v3 trace-detail "
                "response with embedded observations; trace-list, UI table, and CSV exports "
                "are not supported. Export with langfuse --api-version 3 api traces get <id> --json"
            )
        records.append(payload)
    return records


def _new_langfuse_client(base_url: str) -> Langfuse:
    try:
        from langfuse import Langfuse
    except ImportError as error:
        raise ModuleNotFoundError(
            "Langfuse support is not installed; install the project with the 'langfuse' extra"
        ) from error
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        raise LangfuseTraceLoadError(
            "Langfuse loading requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
        )
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        tracing_enabled=False,
    )


def _resolved_base_url(configured: str | None) -> str:
    base_url = configured or os.environ.get("LANGFUSE_BASE_URL")
    if not base_url:
        raise LangfuseTraceLoadError(
            "Langfuse v3 loading requires trace.langfuse.base_url or LANGFUSE_BASE_URL"
        )
    return base_url


def _bounded_filter(
    filter_string: str | None,
    *,
    from_timestamp: datetime,
    to_timestamp: datetime,
) -> str | None:
    if filter_string is None:
        return None
    try:
        conditions = json.loads(filter_string)
    except json.JSONDecodeError as error:
        raise LangfuseTraceLoadError(
            f"Langfuse filter must be a JSON array: {error.msg}"
        ) from error
    if not isinstance(conditions, list) or not all(
        isinstance(condition, Mapping) for condition in conditions
    ):
        raise LangfuseTraceLoadError("Langfuse filter must be a JSON array of conditions")
    time_conditions = [
        {
            "type": "datetime",
            "column": "timestamp",
            "operator": ">=",
            "value": from_timestamp.isoformat(),
        },
        {
            "type": "datetime",
            "column": "timestamp",
            "operator": "<",
            "value": to_timestamp.isoformat(),
        },
    ]
    return json.dumps([*conditions, *time_conditions], separators=(",", ":"))


def _trace_id(provider_trace: TraceWithDetails | TraceWithFullDetails) -> str:
    if not provider_trace.id:
        raise LangfuseTraceLoadError("Langfuse returned a trace without an id")
    return str(provider_trace.id)


def _trace_timestamp(provider_trace: TraceWithDetails | TraceWithFullDetails) -> datetime:
    value = provider_trace.timestamp
    if value.tzinfo is None or value.utcoffset() is None:
        raise LangfuseTraceLoadError(
            f"Langfuse trace {provider_trace.id!r} timestamp must include a timezone"
        )
    return value.astimezone(timezone.utc)


def _normalize_trace(
    provider_trace: TraceWithFullDetails,
    *,
    source_pointer: dict[str, JsonValue],
    require_complete_tree: bool = False,
) -> Trace:
    trace_id = _trace_id(provider_trace)
    observations = provider_trace.observations
    by_id: dict[str, ObservationsView] = {}
    for observation in observations:
        observation_id = str(observation.id)
        if observation_id in by_id:
            raise LangfuseTraceLoadError(
                f"Langfuse trace {trace_id!r} contains duplicate observation id {observation_id!r}"
            )
        by_id[observation_id] = observation

    effective_parents: dict[str, str | None] = {}
    unresolved: dict[str, str] = {}
    for observation_id, observation in by_id.items():
        parent_id = (
            str(observation.parent_observation_id) if observation.parent_observation_id else None
        )
        if parent_id is not None and parent_id not in by_id:
            if require_complete_tree:
                raise LangfuseTraceLoadError("incomplete observation tree: missing parent")
            unresolved[observation_id] = parent_id
            parent_id = None
        effective_parents[observation_id] = parent_id

    ordered = _canonical_observation_order(trace_id, by_id, effective_parents)
    normalized_by_id: dict[str, Span] = {}
    root_spans: list[Span] = []
    tool_call_index = 0
    for observation in ordered:
        observation_id = str(observation.id)
        normalized = _normalize_observation(
            observation,
            unresolved_parent_id=unresolved.get(observation_id),
            base_pointer=source_pointer,
        )
        if normalized.kind is SpanKind.TOOL:
            normalized.tool_call = ToolCall(
                call_id=observation_id,
                index=tool_call_index,
                result_count=0 if normalized.output is UNSET else 1,
            )
            tool_call_index += 1
        normalized_by_id[observation_id] = normalized
        parent_id = effective_parents[observation_id]
        if parent_id is None:
            root_spans.append(normalized)
        else:
            normalized_by_id[parent_id].children.append(normalized)

    attributes: dict[str, JsonValue] = {
        "source_pointer": source_pointer,
        "langfuse": {
            "trace_name": provider_trace.name,
            "timestamp": _trace_timestamp(provider_trace).isoformat(),
            "input": _json_value(provider_trace.input),
            "output": _json_value(provider_trace.output),
            "user_id": provider_trace.user_id,
            "session_id": provider_trace.session_id,
            "release": provider_trace.release,
            "version": provider_trace.version,
            "environment": provider_trace.environment,
            "tags": _json_value(provider_trace.tags),
            "public": provider_trace.public,
            "metadata": _json_value(provider_trace.metadata),
            "html_path": provider_trace.html_path,
            "scores": _json_value(provider_trace.scores),
        },
    }
    tool_catalog = _tool_catalog(ordered, effective_parents)
    if tool_catalog is not None:
        attributes["tool_catalog"] = tool_catalog
    if provider_trace.session_id:
        attributes["logical_case_id"] = str(provider_trace.session_id)

    # Keep repeated evaluations and observation associations rather than choosing a winner.
    scores_by_name: dict[str, list[JsonValue]] = {}
    for score in sorted(provider_trace.scores, key=lambda score: (score.name, score.id)):
        scores_by_name.setdefault(score.name, []).append(_json_value(score))

    return Trace(
        id=trace_id,
        root_spans=root_spans,
        aggregate=TraceAggregate(
            cost_usd=_trace_cost(provider_trace, ordered),
            latency_ms=_trace_latency_ms(provider_trace, ordered),
        ),
        attributes=attributes,
        evaluator_results={name: scores for name, scores in scores_by_name.items()},
    )


def _tool_catalog(
    observations: Sequence[ObservationsView],
    effective_parents: Mapping[str, str | None],
) -> dict[str, JsonValue] | None:
    """Extract one safely trace-wide catalog from API-visible Langfuse inputs.

    Langfuse normalizes tool definitions from supported instrumentation metadata,
    including ``gen_ai.tool.definitions``, into an observation's top-level
    ``input.tools`` array. GENERATION catalogs describe the tools actually
    offered to the model and therefore take precedence; repeated catalogs must
    agree because the canonical Trace contract cannot represent catalogs that
    change per call. A root AGENT catalog is only a fallback for reduced traces
    with no generation observations.
    """

    root_agent_catalogs: list[dict[str, JsonValue]] = []
    generation_catalogs: list[dict[str, JsonValue] | None] = []
    for observation in observations:
        kind, _ = _map_observation_kind(observation.type)
        if kind not in {SpanKind.AGENT, SpanKind.LLM}:
            continue
        catalog = _tool_catalog_from_input(observation.input)
        observation_id = str(observation.id)
        if kind is SpanKind.LLM:
            generation_catalogs.append(catalog)
        elif catalog is not None and effective_parents[observation_id] is None:
            root_agent_catalogs.append(catalog)

    if generation_catalogs:
        first = generation_catalogs[0]
        if first is None:
            return None
        return first if all(catalog == first for catalog in generation_catalogs[1:]) else None
    if root_agent_catalogs:
        first = root_agent_catalogs[0]
        return first if all(catalog == first for catalog in root_agent_catalogs[1:]) else None
    return None


def _tool_catalog_from_input(value: object) -> dict[str, JsonValue] | None:
    """Convert Langfuse ``input.tools`` definitions to canonical JSON Schemas."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Mapping):
        return None
    tools = value.get("tools")
    if not isinstance(tools, list):
        return None

    catalog: dict[str, JsonValue] = {}
    for tool in tools:
        definition = _tool_definition(tool)
        if definition is None:
            continue
        name, schema = definition
        catalog.setdefault(name, schema)
    return catalog if catalog or not tools else None


def _tool_definition(value: object) -> tuple[str, JsonValue] | None:
    """Flatten the tool-definition forms accepted by Langfuse v3 ingestion."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Mapping):
        return None
    nested = value.get("function")
    definition = nested if isinstance(nested, Mapping) else value
    name = definition.get("name")
    if not isinstance(name, str) or not name:
        return None

    schema = definition.get("parameters")
    if schema is None:
        schema = definition.get("parameters_json_schema")
    if schema is None:
        schema = definition.get("inputSchema")
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            schema = None
    return name, _json_value(schema) if isinstance(schema, Mapping) else None


def _canonical_observation_order(
    trace_id: str,
    by_id: Mapping[str, ObservationsView],
    effective_parents: Mapping[str, str | None],
) -> list[ObservationsView]:
    children: dict[str, list[str]] = {observation_id: [] for observation_id in by_id}
    indegree = {observation_id: 0 for observation_id in by_id}
    for observation_id, parent_id in effective_parents.items():
        if parent_id is not None:
            children[parent_id].append(observation_id)
            indegree[observation_id] += 1

    ready: list[tuple[datetime, str]] = []
    for observation_id, degree in indegree.items():
        if degree == 0:
            heapq.heappush(ready, (_started_at(by_id[observation_id]), observation_id))

    ordered: list[ObservationsView] = []
    while ready:
        _, observation_id = heapq.heappop(ready)
        ordered.append(by_id[observation_id])
        for child_id in children[observation_id]:
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                heapq.heappush(ready, (_started_at(by_id[child_id]), child_id))
    if len(ordered) != len(by_id):
        raise LangfuseTraceLoadError(f"Langfuse trace {trace_id!r} contains a parent cycle")
    return ordered


def _normalize_observation(
    observation: ObservationsView,
    *,
    unresolved_parent_id: str | None,
    base_pointer: dict[str, JsonValue],
) -> Span:
    observation_id = str(observation.id)
    kind, subtype = _map_observation_kind(observation.type)
    started_at = _started_at(observation)
    ended_at = _optional_datetime(observation.end_time)
    pointer: dict[str, JsonValue] = {**base_pointer, "observation_id": observation_id}
    if unresolved_parent_id is not None:
        pointer["unresolved_parent_observation_id"] = unresolved_parent_id

    level = _enum_value(observation.level)
    status_message = str(observation.status_message) if observation.status_message else None
    error = status_message or "Langfuse observation level ERROR" if level == "ERROR" else None
    name = str(observation.name) if observation.name else None
    return Span(
        id=observation_id,
        kind=kind,
        start_time=started_at,
        end_time=ended_at,
        input=_io_value(observation.input),
        output=_io_value(observation.output),
        cost_usd=_observation_cost(observation),
        model=str(observation.model) if observation.model else None,
        tool_name=name if kind is SpanKind.TOOL else None,
        error=error,
        attributes={
            "name": name,
            "subtype": subtype,
            "duration_ms": ((ended_at - started_at).total_seconds() * 1_000 if ended_at else None),
            "source_pointer": pointer,
            "status": level,
            "status_message": status_message,
            "langfuse": {
                "metadata": _json_value(observation.metadata),
                "usage_details": _json_value(observation.usage_details),
                "cost_details": _json_value(observation.cost_details),
                "model_parameters": _json_value(observation.model_parameters),
                "version": observation.version,
                "environment": observation.environment,
                "prompt_id": observation.prompt_id,
                "prompt_name": observation.prompt_name,
                "prompt_version": observation.prompt_version,
            },
        },
    )


def _map_observation_kind(value: object) -> tuple[SpanKind, str | None]:
    provider_type = str(value or "UNKNOWN").upper()
    if provider_type == "GENERATION":
        return SpanKind.LLM, provider_type
    try:
        return SpanKind(provider_type), None
    except ValueError:
        return SpanKind.UNKNOWN, provider_type


def _started_at(observation: ObservationsView) -> datetime:
    value = observation.start_time
    if value.tzinfo is None or value.utcoffset() is None:
        raise LangfuseTraceLoadError(
            f"Langfuse observation {observation.id!r} start time must include a timezone"
        )
    return value.astimezone(timezone.utc)


def _optional_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise LangfuseTraceLoadError("Langfuse observation end time must include a timezone")
    return value.astimezone(timezone.utc)


def _observation_cost(observation: ObservationsView) -> float | None:
    details = observation.cost_details or {}
    cost = details.get("total")
    if cost is None:
        cost = observation.calculated_total_cost
    return float(cost) if cost is not None and cost >= 0 else None


def _trace_cost(
    provider_trace: TraceWithFullDetails,
    observations: Sequence[ObservationsView],
) -> float | None:
    if provider_trace.total_cost is not None and provider_trace.total_cost >= 0:
        return float(provider_trace.total_cost)
    costs = [
        cost for observation in observations if (cost := _observation_cost(observation)) is not None
    ]
    return sum(costs) if costs else None


def _trace_latency_ms(
    provider_trace: TraceWithFullDetails,
    observations: Sequence[ObservationsView],
) -> float | None:
    if provider_trace.latency is not None and provider_trace.latency >= 0:
        return float(provider_trace.latency) * 1_000
    starts = [_started_at(observation) for observation in observations]
    ends = [
        end
        for observation in observations
        if (end := _optional_datetime(observation.end_time)) is not None
    ]
    if not starts or not ends:
        return None
    return (max(ends) - min(starts)).total_seconds() * 1_000


def _io_value(value: object) -> JsonValue | UNSET:
    if value is None:
        # The v3 API does not distinguish a missing value from an explicitly
        # recorded JSON null.
        return UNSET
    if isinstance(value, str):
        try:
            return _json_value(json.loads(value))
        except json.JSONDecodeError:
            pass
    return _json_value(value)


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw).upper()


def _json_default(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", by_alias=True)
    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict):
        return legacy_dict(by_alias=True)
    return str(value)


def _json_value(value: object) -> JsonValue:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default))
