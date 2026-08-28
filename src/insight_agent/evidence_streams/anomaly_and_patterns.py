"""Anomaly and recurring-pattern evidence stream.

This module owns the stream entry point and its feature extraction, anomaly
selection, clustering, recurring-failure analysis, and evidence rendering.
It surfaces inspectable evidence with exact source pointers; it does not author
Insights.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Literal

import numpy as np
from pydantic import Field, FiniteFloat, field_validator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from insight_agent.evidence_streams.contracts import EvidenceStreamResult, Problem
from insight_agent.evidence_streams.venue import DEFAULT_PROFILE, VenueProfile
from insight_agent.traces import UNSET, ContractModel, Span, SpanKind, Trace, TraceSnapshot

N_ESTIMATORS = 300
CONTAMINATION = 0.02
RANDOM_STATE = 0
RECURRENCE_THRESHOLD = 3

STRICT_ERROR_PREFIX = re.compile(r"\A\s*(?:error|failed|failure)\s*[:\-]", re.I)
PYTHON_TRACEBACK = re.compile(r"(?m)^Traceback \(most recent call last\):")
WHITESPACE = re.compile(r"\s+")
LONG_ID = re.compile(r"\b(?:0x[0-9a-f]+|[0-9a-f]{8,}|\d{4,})\b", re.I)
PATH = re.compile(r"(?:[A-Za-z]:)?[/\\](?:[^\s:/\\]+[/\\])+[^\s:]*")
NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.I)

DEFAULT_FEATURES = (
    "tool_call_count",
    "distinct_tool_count",
    "trajectory_step_count",
    "dominant_tool_share",
    "repeated_identical_call_rate",
    "explicit_failure_rate",
    "returned_data_false_rate",
    "code_execution_share",
    "output_kb_per_call",
    "largest_output_kb",
    "tool_duration_sec_total",
)

FEATURE_LABELS = {
    "tool_call_count": "tool-call count",
    "distinct_tool_count": "distinct-tool count",
    "trajectory_step_count": "trajectory length",
    "dominant_tool_share": "dominant-tool share",
    "repeated_identical_call_rate": "identical-call repetition",
    "explicit_failure_rate": "explicit tool-failure rate",
    "returned_data_false_rate": "no-structured-data rate",
    "code_execution_share": "code-execution share",
    "output_kb_per_call": "average tool-output size",
    "largest_output_kb": "largest tool output",
    "tool_duration_sec_total": "total recorded tool time",
}


@dataclass(frozen=True)
class NormalizedCall:
    """One ordered tool call in the platform-neutral IA2 contract."""

    call_id: str
    call_index: int
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    result: Any = None
    duration_ms: float | None = None
    source_pointer: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedStep:
    """One observable trajectory step used for ordered sequence grouping."""

    step_index: int
    step_type: str
    name: str = ""
    content: str = ""
    source_pointer: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedTrace:
    """Complete IA2 input unit after a dataset adapter maps source records."""

    trace_id: str
    calls: Sequence[NormalizedCall]
    steps: Sequence[NormalizedStep] = field(default_factory=tuple)
    source_pointer: Mapping[str, Any] = field(default_factory=dict)
    observed_verdict: str | None = None
    cost: float | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceFeatures:
    """Model-ready numeric and ordered evidence for one complete trace."""

    trace_id: str
    numeric: Mapping[str, float]
    sequence_tokens: Sequence[str]
    source_pointer: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedTrace:
    """Structured IA2 evidence produced before selection and grouping."""

    trace: NormalizedTrace
    features: TraceFeatures
    tool_names: Sequence[str]
    failure_events: Sequence[Mapping[str, Any]]
    last_agent_excerpt: str


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        for key in ("content", "output", "message", "error", "summary"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return _stable_json(result) if result is not None else ""


def cap_head_tail(text: str, *, max_chars: int = 8_000) -> str:
    """Cap a large output while retaining equal head and tail evidence."""

    if max_chars < 80:
        raise ValueError("max_chars must be at least 80")
    if len(text) <= max_chars:
        return text
    marker = "\n… <middle omitted by IA2 head+tail cap> …\n"
    remaining = max_chars - len(marker)
    head = (remaining + 1) // 2
    tail = remaining // 2
    return text[:head] + marker + text[-tail:]


def normalize_error_template(text: str, *, limit: int = 180) -> str:
    """Fold volatile paths, long identifiers, and numbers into one signature."""

    value = WHITESPACE.sub(" ", text).strip()
    value = PATH.sub("<path>", value)
    value = LONG_ID.sub("<id>", value)
    value = NUMBER.sub("<n>", value)
    return value[:limit]


def denoise_output(text: str, *, max_chars: int = 8_000) -> str:
    """Optional unmeasured Stage-1 utility: cap and remove adjacent duplicates."""

    capped = cap_head_tail(text, max_chars=max_chars)
    output: list[str] = []
    previous: str | None = None
    duplicate_count = 0
    for line in capped.splitlines():
        normalized = WHITESPACE.sub(" ", line).strip()
        if normalized == previous:
            duplicate_count += 1
            continue
        if duplicate_count:
            output.append(f"<previous line repeated {duplicate_count} times>")
            duplicate_count = 0
        output.append(line)
        previous = normalized
    if duplicate_count:
        output.append(f"<previous line repeated {duplicate_count} times>")
    return "\n".join(output)


def decode_explicit_failure(
    tool_name: str,
    result: Any,
    *,
    profile: VenueProfile = DEFAULT_PROFILE,
) -> tuple[bool, str | None, str]:
    """Conservatively decode only structured or native explicit failure evidence."""

    text = _result_text(result)
    if isinstance(result, Mapping):
        if result.get("is_error") is True or result.get("isError") is True:
            return True, "structured_error_flag", text
        if result.get("success") is False or result.get("ok") is False:
            return True, "structured_unsuccessful", text
        status = str(result.get("status") or "").strip().lower()
        if status in {"error", "failed", "failure", "crashed", "timeout"}:
            return True, f"structured_status_{status}", text
        exit_code = result.get("exit_code", result.get("returncode"))
        if isinstance(exit_code, (int, float)) and int(exit_code) != 0:
            return True, "nonzero_exit_code", text
    if STRICT_ERROR_PREFIX.search(text):
        return True, "tool_output_error_prefix", text
    if profile.is_code_execution_tool(tool_name) and PYTHON_TRACEBACK.search(text):
        return True, "python_traceback", text
    return False, None, text


def normalized_failure_signature(tool_name: str, text: str, marker: str) -> str:
    """Create a deterministic tool/error signature from an explicit failure."""

    lines = [WHITESPACE.sub(" ", line).strip() for line in text.splitlines() if line.strip()]
    if marker == "python_traceback":
        candidates = [line for line in lines if re.search(r"(?:Error|Exception|Failed)\b", line)]
        message = candidates[-1] if candidates else "Python traceback"
    else:
        message = lines[0] if lines else marker
    return f"{tool_name}: {normalize_error_template(message)}"


def _sequence_tokens(
    trace: NormalizedTrace, *, profile: VenueProfile = DEFAULT_PROFILE
) -> list[str]:
    # The emitted token stays the literal "evaluation:boundary" regardless of
    # what the venue calls its terminal step: it is a fixed vocabulary item in
    # the clustering feature space, not a passthrough of the source value.
    evaluation_type = profile.evaluation_step_type.strip().lower()
    if trace.steps:
        tokens = []
        for step in sorted(trace.steps, key=lambda item: item.step_index):
            step_type = (step.step_type or "unknown").strip().lower()
            name = (step.name or step_type).strip().replace(" ", "_")
            tokens.append(
                "evaluation:boundary" if step_type == evaluation_type else f"{step_type}:{name}"
            )
        return tokens or ["trajectory:empty"]
    return [
        f"tool:{call.tool_name.strip().replace(' ', '_')}"
        for call in sorted(trace.calls, key=lambda item: item.call_index)
    ] or ["trajectory:empty"]


def _last_agent_excerpt(
    trace: NormalizedTrace, limit: int = 520, *, profile: VenueProfile = DEFAULT_PROFILE
) -> str:
    for step in sorted(trace.steps, key=lambda item: item.step_index, reverse=True):
        if step.step_type in profile.agent_step_types and step.content:
            text = WHITESPACE.sub(" ", step.content).strip()
            return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
    return ""


def extract_trace_features(
    trace: NormalizedTrace, *, profile: VenueProfile = DEFAULT_PROFILE
) -> PreparedTrace:
    """Build Stage-3 events and default numeric features from one trace."""

    calls = sorted(trace.calls, key=lambda item: item.call_index)
    tool_names = [call.tool_name for call in calls]
    counts = Counter(tool_names)
    fingerprints = {_stable_json([call.tool_name, call.arguments]) for call in calls}
    output_sizes: list[int] = []
    durations: list[float] = []
    failures: list[dict[str, Any]] = []
    returned_false = 0
    for call in calls:
        failed, marker, text = decode_explicit_failure(call.tool_name, call.result, profile=profile)
        output_sizes.append(len(text.encode("utf-8", "ignore")))
        if call.duration_ms is not None and math.isfinite(float(call.duration_ms)):
            durations.append(max(0.0, float(call.duration_ms)))
        if isinstance(call.result, Mapping) and call.result.get(profile.returned_data_key) is False:
            returned_false += 1
        if failed and marker:
            signature = normalized_failure_signature(call.tool_name, text, marker)
            failures.append(
                {
                    "trace_id": trace.trace_id,
                    "call_id": call.call_id,
                    "call_index": call.call_index,
                    "tool_name": call.tool_name,
                    "marker": marker,
                    "signature": signature,
                    "message_signature": signature.split(": ", 1)[-1],
                    "output_excerpt": cap_head_tail(
                        WHITESPACE.sub(" ", text).strip(), max_chars=300
                    )
                    if len(text) > 300
                    else WHITESPACE.sub(" ", text).strip(),
                    "source_pointer": dict(call.source_pointer),
                }
            )
    n_calls = len(calls)
    numeric = {
        "tool_call_count": float(n_calls),
        "distinct_tool_count": float(len(counts)),
        "trajectory_step_count": float(len(trace.steps) or n_calls),
        "dominant_tool_share": counts.most_common(1)[0][1] / n_calls if n_calls else 0.0,
        "repeated_identical_call_rate": 1.0 - len(fingerprints) / n_calls if n_calls else 0.0,
        "explicit_failure_rate": len(failures) / n_calls if n_calls else 0.0,
        "returned_data_false_rate": returned_false / n_calls if n_calls else 0.0,
        "code_execution_share": (
            sum(counts.get(name, 0) for name in profile.code_execution_tools) / n_calls
            if n_calls
            else 0.0
        ),
        "output_kb_per_call": sum(output_sizes) / n_calls / 1024.0 if n_calls else 0.0,
        "largest_output_kb": max(output_sizes) / 1024.0 if output_sizes else 0.0,
        "tool_duration_sec_total": sum(durations) / 1000.0,
    }
    for name, value in trace.metrics.items():
        if math.isfinite(float(value)):
            key = str(name)
            # Caller-supplied metrics share a flat namespace with the eleven
            # built-in features, so a collision silently replaces a measured
            # feature and corrupts anomaly detection. The overwrite is kept for
            # backwards compatibility; only the silence is removed.
            if key in numeric:
                warnings.warn(
                    f"trace {trace.trace_id!r} metric {key!r} shadows a built-in IA2 feature; "
                    f"the built-in value {numeric[key]!r} is being replaced by {float(value)!r}. "
                    "Rename the metric unless this is deliberate.",
                    UserWarning,
                    stacklevel=2,
                )
            numeric[key] = float(value)
    return PreparedTrace(
        trace=trace,
        features=TraceFeatures(
            trace_id=trace.trace_id,
            numeric=numeric,
            sequence_tokens=_sequence_tokens(trace, profile=profile),
            source_pointer=trace.source_pointer,
        ),
        tool_names=tool_names,
        failure_events=failures,
        last_agent_excerpt=_last_agent_excerpt(trace, profile=profile),
    )


def prepare_traces(
    traces: Iterable[NormalizedTrace], *, profile: VenueProfile = DEFAULT_PROFILE
) -> tuple[list[PreparedTrace], list[dict[str, Any]]]:
    """Run Stage 3 over complete traces and return trace rows plus failure events."""

    prepared = [extract_trace_features(trace, profile=profile) for trace in traces]
    prepared.sort(key=lambda item: item.trace.trace_id)
    events = [dict(event) for item in prepared for event in item.failure_events]
    return prepared, events


def _matrix(records: Sequence[TraceFeatures], feature_names: Sequence[str]) -> np.ndarray:
    if not records:
        raise ValueError("at least one trace is required")
    missing = sorted(
        {name for record in records for name in feature_names if name not in record.numeric}
    )
    if missing:
        offenders = sorted(
            {
                record.trace_id
                for record in records
                if any(name not in record.numeric for name in missing)
            }
        )[:10]
        raise ValueError(
            f"missing numeric features: {', '.join(missing)}. "
            f"First traces lacking them: {', '.join(offenders)}. "
            "Custom feature names must be supplied by every trace via "
            "NormalizedTrace.metrics (canonical JSONL: the per-trace 'metrics' object)."
        )
    return np.asarray(
        [[float(record.numeric[name]) for name in feature_names] for record in records],
        dtype=float,
    )


def _robust_coordinates(matrix: np.ndarray) -> np.ndarray:
    centers = np.median(matrix, axis=0)
    mads = np.median(np.abs(matrix - centers), axis=0)
    mads[mads == 0] = 1e-9
    return (matrix - centers) / (1.4826 * mads)


def _robust_reasons(
    records: Sequence[TraceFeatures], feature_names: Sequence[str]
) -> dict[str, list[str]]:
    centers = {
        name: median(float(record.numeric[name]) for record in records) for name in feature_names
    }
    mads = {
        name: median(abs(float(record.numeric[name]) - centers[name]) for record in records) or 1e-9
        for name in feature_names
    }
    output: dict[str, list[str]] = {}
    for record in records:
        ranked = sorted(
            (
                abs((float(record.numeric[name]) - centers[name]) / (1.4826 * mads[name])),
                float(record.numeric[name]) - centers[name],
                name,
            )
            for name in feature_names
        )
        reasons = [
            f"{'high' if delta > 0 else 'low'} {FEATURE_LABELS.get(name, name.replace('_', ' '))}"
            for magnitude, delta, name in reversed(ranked[-3:])
            if magnitude > 1.5
        ]
        output[record.trace_id] = reasons or [FEATURE_LABELS.get(max(ranked)[2], max(ranked)[2])]
    return output


def select_anomalies(
    records: Sequence[TraceFeatures],
    feature_names: Sequence[str] = DEFAULT_FEATURES,
    *,
    contamination: float = CONTAMINATION,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
    input_scaling: str = "none",
) -> list[dict[str, Any]]:
    """Run IA2 Isolation Forest and return inspectable per-trace evidence.

    Outcomes/verdicts are absent from ``TraceFeatures``. ``input_scaling='none'``
    matches the current multi-dataset path; ``'robust'`` reproduces the original
    single-dataset variant. PCA is report-only and never changes decisions.
    """

    if not 0.0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5")
    matrix = _matrix(records, feature_names)
    if input_scaling == "none":
        model_input = matrix
    elif input_scaling == "robust":
        model_input = _robust_coordinates(matrix)
    else:
        raise ValueError("input_scaling must be 'none' or 'robust'")
    detector = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    ).fit(model_input)
    anomaly_scores = -detector.decision_function(model_input)
    flags = detector.predict(model_input) == -1
    if len(records) >= 2 and len(feature_names) >= 2:
        scaled = StandardScaler().fit_transform(matrix)
        projection = (
            PCA(n_components=2, random_state=random_state).fit_transform(scaled)
            if np.any(np.var(scaled, axis=0) > 0.0)
            else np.zeros((len(records), 2), dtype=float)
        )
    else:
        projection = np.zeros((len(records), 2), dtype=float)
    reasons = _robust_reasons(records, feature_names)
    return [
        {
            "trace_id": record.trace_id,
            "anomaly_score": float(anomaly_scores[index]),
            "is_anomaly": bool(flags[index]),
            "anomaly_reasons": reasons[record.trace_id],
            "pca_x": float(projection[index, 0]),
            "pca_y": float(projection[index, 1]),
            "source_pointer": dict(record.source_pointer),
        }
        for index, record in enumerate(records)
    ]


def _choose_cluster_count(
    matrix: Any, candidates: Sequence[int], *, random_state: int
) -> tuple[int, list[dict[str, float]]]:
    valid = [k for k in candidates if 2 <= k < matrix.shape[0]]
    if not valid:
        raise ValueError("trajectory grouping needs at least three traces")
    scored: list[dict[str, float]] = []
    for k in valid:
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=20).fit_predict(matrix)
        score = float(
            silhouette_score(
                matrix,
                labels,
                metric="cosine",
                sample_size=min(1000, matrix.shape[0]),
                random_state=random_state,
            )
        )
        scored.append({"k": float(k), "silhouette_cosine": score})
    best = max(scored, key=lambda item: item["silhouette_cosine"])
    return int(best["k"]), scored


def _rank_top_terms(
    center: Sequence[float], vocabulary: Sequence[Any], *, limit: int = 6
) -> list[str]:
    """Rank display terms consistently across numerical backends.

    BLAS implementations can differ below the precision relevant to this
    diagnostic output. Round weights before using the term as a deterministic
    tie-breaker so equivalent clusters render byte-identically across hosts.
    """

    weighted_terms = [
        (round(float(weight), 12), str(vocabulary[index]))
        for index, weight in enumerate(center)
        if weight > 0
    ]
    weighted_terms.sort(reverse=True)
    return [term for _, term in weighted_terms[:limit]]


def group_trajectories(
    records: Sequence[TraceFeatures],
    *,
    cluster_candidates: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
    max_features: int = 240,
    random_state: int = RANDOM_STATE,
    on_insufficient_traces: Literal["raise", "skip"] = "raise",
) -> dict[str, Any]:
    """Group recurring ordered action patterns without outcome labels.

    Clustering needs some ``k`` with ``2 <= k < len(records)``, so a corpus of
    fewer than three traces cannot be grouped. The default is to raise, which
    is what the measured pipeline did. Pass ``on_insufficient_traces="skip"``
    to get an explicit abstention instead — useful when running the stage over
    a small smoke corpus.
    """

    if on_insufficient_traces not in ("raise", "skip"):
        raise ValueError("on_insufficient_traces must be 'raise' or 'skip'")
    if on_insufficient_traces == "skip" and not [
        k for k in cluster_candidates if 2 <= k < len(records)
    ]:
        return {
            "status": "not_evaluable",
            "reason": (
                f"trajectory grouping needs at least three traces and a candidate k with "
                f"2 <= k < {len(records)}; got {len(records)} trace(s) and candidates "
                f"{list(cluster_candidates)}"
            ),
            "trace_count": len(records),
            "clusters": [],
            "assignments": {},
        }

    documents = [" ".join(record.sequence_tokens) for record in records]
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(documents)
    selected_k, selection_scores = _choose_cluster_count(
        matrix, cluster_candidates, random_state=random_state
    )
    model = KMeans(n_clusters=selected_k, random_state=random_state, n_init=30).fit(matrix)
    labels = model.labels_
    distances = model.transform(matrix)
    vocabulary = vectorizer.get_feature_names_out()
    clusters: list[dict[str, Any]] = []
    assignments: dict[str, int] = {}
    for cluster_id in range(selected_k):
        indexes = np.flatnonzero(labels == cluster_id)
        center = model.cluster_centers_[cluster_id]
        representative = int(indexes[np.argmin(distances[indexes, cluster_id])])
        terms = _rank_top_terms(center, vocabulary)
        trace_ids = [records[int(index)].trace_id for index in indexes]
        assignments.update({trace_id: cluster_id for trace_id in trace_ids})
        clusters.append(
            {
                "cluster_id": cluster_id,
                "trace_count": len(trace_ids),
                "trace_ids": trace_ids,
                "representative_trace_id": records[representative].trace_id,
                "representative_source_pointer": dict(records[representative].source_pointer),
                "top_sequence_terms": terms,
            }
        )
    clusters.sort(key=lambda item: item["trace_count"], reverse=True)
    return {
        "method": "TF-IDF ordered unigrams/bigrams + KMeans",
        "fit_used_outcome_labels": False,
        "selected_k": selected_k,
        "selection_scores": selection_scores,
        "assignments": assignments,
        "clusters": clusters,
    }


def group_observed_verdicts(
    traces: Iterable[NormalizedTrace], *, minimum_independent_traces: int = 1
) -> list[dict[str, Any]]:
    """Group explicit terminal verdicts, the canonical Stage 5."""

    grouped: dict[str, list[NormalizedTrace]] = defaultdict(list)
    for trace in traces:
        if trace.observed_verdict:
            grouped[normalize_error_template(trace.observed_verdict)].append(trace)
    patterns: list[dict[str, Any]] = []
    for verdict, members in grouped.items():
        if len(members) < minimum_independent_traces:
            continue
        representatives = sorted(
            members,
            key=lambda trace: trace.cost if trace.cost is not None else -1.0,
            reverse=True,
        )[:5]
        known_costs = [float(trace.cost) for trace in members if trace.cost is not None]
        patterns.append(
            {
                "verdict": verdict,
                "independent_trace_count": len(members),
                "trace_ids": sorted(trace.trace_id for trace in members),
                "total_known_cost": sum(known_costs) if known_costs else None,
                "representatives": [
                    {"trace_id": trace.trace_id, "source_pointer": dict(trace.source_pointer)}
                    for trace in representatives
                ],
            }
        )
    return sorted(patterns, key=lambda item: item["independent_trace_count"], reverse=True)


def group_failures(
    events: Iterable[Mapping[str, Any]], *, minimum_independent_traces: int = RECURRENCE_THRESHOLD
) -> list[dict[str, Any]]:
    """Compress strict failure events into recurring tool-specific signatures."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event["signature"])].append(event)
    patterns: list[dict[str, Any]] = []
    for signature, members in grouped.items():
        trace_ids = sorted({str(member["trace_id"]) for member in members})
        if len(trace_ids) < minimum_independent_traces:
            continue
        representatives: dict[str, Mapping[str, Any]] = {}
        for member in members:
            representatives.setdefault(str(member["trace_id"]), member)
        tools = Counter(str(member["tool_name"]) for member in members)
        patterns.append(
            {
                "signature": signature,
                "event_count": len(members),
                "independent_trace_count": len(trace_ids),
                "trace_ids": trace_ids,
                "top_tools": [name for name, _ in tools.most_common(5)],
                "representatives": [dict(item) for item in list(representatives.values())[:5]],
            }
        )
    return sorted(
        patterns,
        key=lambda item: (item["independent_trace_count"], item["event_count"]),
        reverse=True,
    )


def group_failures_cross_tool(
    events: Iterable[Mapping[str, Any]], *, minimum_independent_traces: int = RECURRENCE_THRESHOLD
) -> list[dict[str, Any]]:
    """Merge the same normalized failure message across different tool names."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event.get("message_signature") or event["signature"])].append(event)
    patterns: list[dict[str, Any]] = []
    for message, members in grouped.items():
        trace_ids = sorted({str(member["trace_id"]) for member in members})
        if len(trace_ids) < minimum_independent_traces:
            continue
        representatives: dict[str, Mapping[str, Any]] = {}
        for member in members:
            representatives.setdefault(str(member["trace_id"]), member)
        patterns.append(
            {
                "message_signature": message,
                "event_count": len(members),
                "independent_trace_count": len(trace_ids),
                "tool_names": sorted({str(member["tool_name"]) for member in members}),
                "trace_ids": trace_ids,
                "representatives": [dict(item) for item in list(representatives.values())[:5]],
            }
        )
    return sorted(
        patterns,
        key=lambda item: (item["independent_trace_count"], item["event_count"]),
        reverse=True,
    )


def _markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _pointer(value: Mapping[str, Any]) -> str:
    return _markdown(_stable_json(value)) if value else "not supplied"


def build_evidence_digest(
    prepared: Sequence[PreparedTrace],
    anomalies: Sequence[Mapping[str, Any]],
    trajectory_groups: Mapping[str, Any] | None,
    verdict_groups: Sequence[Mapping[str, Any]],
    failure_groups: Sequence[Mapping[str, Any]],
    cross_tool_failure_groups: Sequence[Mapping[str, Any]],
    *,
    max_anomalies: int = 50,
    max_patterns_per_section: int = 24,
) -> str:
    """Build the compact Stage-6 packet read by the separate Analyst LLM."""

    prepared_by_id = {item.trace.trace_id: item for item in prepared}
    ranked = sorted(
        (item for item in anomalies if item["is_anomaly"]),
        key=lambda item: float(item["anomaly_score"]),
        reverse=True,
    )
    # The inventory must report the true flag count, not the rendered row count.
    # Truncating first made a corpus with 200 flags report "50", which a reader
    # ranking issues by how widespread they are would take as the real total.
    flagged_total = len(ranked)
    flagged = ranked[:max_anomalies]
    lines = [
        "# IA2 cited evidence digest",
        "",
        "## Reader contract",
        "",
        "IA2 answers **what is unusual?** and **what recurs?** It does not author an Insight, prove causality, or turn an anomaly into an error. The Analyst must inspect cited traces and may file zero Insights.",
        "",
        "## Inventory",
        "",
        f"- Complete traces: {len(prepared):,}",
        f"- Ordered tool calls: {sum(len(item.trace.calls) for item in prepared):,}",
        f"- Isolation Forest flags: {flagged_total:,}",
        f"- Trajectory groups: {len((trajectory_groups or {}).get('clusters', [])):,}",
        f"- Observed-verdict groups: {len(verdict_groups):,}",
        f"- Recurring strict-failure signatures: {len(failure_groups):,}",
        "",
        "## Unusual traces",
        "",
    ]
    if flagged_total > len(flagged):
        lines.extend(
            [
                f"Showing the {len(flagged):,} highest-scoring of {flagged_total:,} flagged "
                "traces. The remaining flags are in the anomaly rows, not this table.",
                "",
            ]
        )
    lines.extend(
        [
            "| Trace | Score | Feature reasons | Source pointer |",
            "|---|---:|---|---|",
        ]
    )
    for item in flagged:
        lines.append(
            f"| `{_markdown(item['trace_id'])}` | {float(item['anomaly_score']):.5f} | {_markdown('; '.join(item['anomaly_reasons']))} | `{_pointer(item.get('source_pointer') or {})}` |"
        )
    if trajectory_groups:
        lines.extend(
            [
                "",
                "## Recurring ordered trajectory patterns",
                "",
                "| Group | Traces | Representative | Characteristic sequence terms |",
                "|---:|---:|---|---|",
            ]
        )
        for group in trajectory_groups.get("clusters", [])[:max_patterns_per_section]:
            lines.append(
                f"| {group['cluster_id']} | {group['trace_count']} | `{_markdown(group['representative_trace_id'])}` | {_markdown(', '.join(group['top_sequence_terms']))} |"
            )
    lines.extend(["", "## Explicit terminal-verdict patterns", ""])
    if verdict_groups:
        lines.extend(["| Verdict | Traces | Representative pointers |", "|---|---:|---|"])
        for group in verdict_groups[:max_patterns_per_section]:
            refs = "; ".join(
                f"`{item['trace_id']}` `{_pointer(item['source_pointer'])}`"
                for item in group["representatives"][:3]
            )
            lines.append(
                f"| {_markdown(group['verdict'])} | {group['independent_trace_count']} | {refs} |"
            )
    else:
        lines.append("No explicit terminal verdicts were supplied; no verdict claim is made.")
    lines.extend(
        [
            "",
            "## Recurring strict tool-failure signatures",
            "",
            "These are observed tool/runtime outputs, not proof that the agent chose the wrong tool.",
            "",
            "| Signature | Traces | Events | Representative calls |",
            "|---|---:|---:|---|",
        ]
    )
    for group in failure_groups[:max_patterns_per_section]:
        calls = "; ".join(
            f"`{item['trace_id']}/{item['call_id']}` `{_pointer(item.get('source_pointer') or {})}`"
            for item in group["representatives"][:3]
        )
        lines.append(
            f"| {_markdown(group['signature'])} | {group['independent_trace_count']} | {group['event_count']} | {calls} |"
        )
    if cross_tool_failure_groups:
        lines.extend(
            [
                "",
                "### Cross-tool recurring messages",
                "",
                "| Message | Tools | Traces | Events |",
                "|---|---|---:|---:|",
            ]
        )
        for group in cross_tool_failure_groups[:max_patterns_per_section]:
            lines.append(
                f"| {_markdown(group['message_signature'])} | {_markdown(', '.join(group['tool_names']))} | {group['independent_trace_count']} | {group['event_count']} |"
            )
    lines.extend(
        [
            "",
            "## Analyst authoring rules",
            "",
            "1. File only recurring, operationally useful, evidence-backed Insights.",
            "2. Cite exact trace IDs and call IDs, then open raw evidence when needed.",
            "3. Distinguish observed tool/runtime failure from inferred agent misuse.",
            "4. Do not infer root cause or quality from anomaly or cluster membership.",
            "5. Withhold weak or duplicate claims and state what remains unproven.",
            "",
        ]
    )
    # Guards against an adapter error silently dropping cited traces. This was
    # a bare `assert`, which disappears entirely under `python -O` — precisely
    # when a long batch run would be least likely to notice the corruption.
    uncited = sorted(
        {str(item["trace_id"]) for item in flagged if str(item["trace_id"]) not in prepared_by_id}
    )
    if uncited:
        raise RuntimeError(
            f"digest cites traces absent from the prepared set: {', '.join(uncited)}. "
            "This indicates the anomaly rows and the prepared traces came from "
            "different corpora."
        )
    return "\n".join(lines)


class Anomaly(ContractModel):
    trace_id: str = Field(min_length=1)
    anomaly_score: FiniteFloat
    is_anomaly: bool
    anomaly_reasons: tuple[str, ...]
    pca_x: FiniteFloat
    pca_y: FiniteFloat
    source_pointer: Mapping[str, Any]


class FailureGroup(ContractModel):
    signature: str = Field(min_length=1)
    event_count: int = Field(ge=1)
    independent_trace_count: int = Field(ge=1)
    trace_ids: tuple[str, ...] = Field(min_length=1)
    top_tools: tuple[str, ...]
    representatives: tuple[Mapping[str, Any], ...] = Field(min_length=1)


class CrossToolFailureGroup(ContractModel):
    message_signature: str = Field(min_length=1)
    event_count: int = Field(ge=1)
    independent_trace_count: int = Field(ge=1)
    tool_names: tuple[str, ...]
    trace_ids: tuple[str, ...] = Field(min_length=1)
    representatives: tuple[Mapping[str, Any], ...] = Field(min_length=1)


class AnomalyAndPatternsAnalysis(ContractModel):
    prepared: tuple[PreparedTrace, ...]
    failure_events: tuple[Mapping[str, Any], ...]
    anomalies: tuple[Anomaly, ...]
    trajectory_groups: Mapping[str, Any] | None
    verdict_groups: tuple[Mapping[str, Any], ...]
    failure_groups: tuple[FailureGroup, ...]
    cross_tool_failure_groups: tuple[CrossToolFailureGroup, ...]
    digest: str


def run_ia2(
    traces: Sequence[NormalizedTrace],
    *,
    feature_names: Sequence[str] = DEFAULT_FEATURES,
    contamination: float = CONTAMINATION,
    cluster_candidates: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
    minimum_independent_traces: int = RECURRENCE_THRESHOLD,
    input_scaling: str = "none",
    profile: VenueProfile = DEFAULT_PROFILE,
) -> AnomalyAndPatternsAnalysis:
    """Execute Stages 3–6 and return every intermediate plus the cited digest."""

    prepared, failure_events = prepare_traces(traces, profile=profile)
    records = [item.features for item in prepared]
    anomalies = select_anomalies(
        records,
        feature_names,
        contamination=contamination,
        input_scaling=input_scaling,
    )
    trajectory_groups: dict[str, Any] | None
    if len(records) >= 3:
        trajectory_groups = group_trajectories(records, cluster_candidates=cluster_candidates)
    else:
        trajectory_groups = None
    verdict_groups = group_observed_verdicts(
        traces, minimum_independent_traces=minimum_independent_traces
    )
    failure_groups = group_failures(
        failure_events, minimum_independent_traces=minimum_independent_traces
    )
    cross_tool_groups = group_failures_cross_tool(
        failure_events, minimum_independent_traces=minimum_independent_traces
    )
    digest = build_evidence_digest(
        prepared,
        anomalies,
        trajectory_groups,
        verdict_groups,
        failure_groups,
        cross_tool_groups,
    )
    return AnomalyAndPatternsAnalysis(
        prepared=tuple(prepared),
        failure_events=tuple(failure_events),
        anomalies=tuple(anomalies),
        trajectory_groups=trajectory_groups,
        verdict_groups=tuple(verdict_groups),
        failure_groups=tuple(failure_groups),
        cross_tool_failure_groups=tuple(cross_tool_groups),
        digest=digest,
    )


@dataclass(frozen=True)
class AnomalyAndPatternsArtifacts:
    result: AnomalyAndPatternsAnalysis
    config: AnomalyAndPatternsConfig


def problems_from_analysis(result: AnomalyAndPatternsAnalysis) -> tuple[Problem, ...]:
    """Project IA2's native analysis into candidate problems for synthesis."""

    problems: list[Problem] = []
    anomalies = [row for row in result.anomalies if row.is_anomaly]
    if anomalies:
        ranked = sorted(
            anomalies,
            key=lambda row: (-row.anomaly_score, row.trace_id),
        )
        reason_counts = Counter(str(reason) for row in anomalies for reason in row.anomaly_reasons)
        common_reasons = (
            ", ".join(f"{reason} ({count})" for reason, count in reason_counts.most_common(5))
            or "no single dominant feature"
        )
        trace_subject = "trace was" if len(anomalies) == 1 else "traces were"
        description = (
            f"{len(anomalies)} {trace_subject} statistical outliers in the corpus. "
            f"The most common feature-level reasons were {common_reasons}. "
            "An anomaly is evidence for investigation, not proof of a defect."
        )
        problems.append(
            Problem(
                description=description,
                supporting_trace_ids=tuple(row.trace_id for row in ranked[:50]),
            )
        )

    for group in result.failure_groups:
        tools = ", ".join(group.top_tools) or "unknown tools"
        problems.append(
            Problem(
                description=(
                    f"Recurring strict tool failure `{group.signature}` appeared in "
                    f"{group.event_count} event(s) across "
                    f"{group.independent_trace_count} independent trace(s). "
                    f"Affected tools: {tools}."
                ),
                supporting_trace_ids=group.trace_ids,
            )
        )

    for group in result.cross_tool_failure_groups:
        tools = group.tool_names
        if len(tools) < 2:
            continue
        problems.append(
            Problem(
                description=(
                    f"The normalized failure `{group.message_signature}` appeared in "
                    f"{group.event_count} event(s) across "
                    f"{group.independent_trace_count} independent trace(s) and multiple "
                    f"tools: {', '.join(tools)}. Inspect the supporting traces to determine "
                    "whether those occurrences share a cause."
                ),
                supporting_trace_ids=group.trace_ids,
            )
        )

    return tuple(problems)


def _duration_ms(span: Span) -> float | None:
    if span.duration_ms is not None:
        return span.duration_ms
    if span.started_at is None or span.ended_at is None:
        return None
    return (span.ended_at - span.started_at).total_seconds() * 1_000


def _span_content(span: Span) -> str:
    value = span.output
    if value is UNSET or value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content")
        if isinstance(content, str):
            return content
        messages = value.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            for message in reversed(messages):
                if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                    return str(message["content"])
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _native_step_type(span: Span) -> str:
    return {
        SpanKind.TOOL: "tool",
        SpanKind.AGENT: "agent",
        SpanKind.EVALUATOR: "evaluation",
    }.get(span.kind, span.kind.value.lower())


def to_ia2_trace(trace: Trace, *, profile: VenueProfile = DEFAULT_PROFILE) -> NormalizedTrace:
    """Project one normalized trace into the anomaly-and-pattern analysis model."""

    calls: list[NormalizedCall] = []
    tool_spans = [span for span in trace.spans if span.kind is SpanKind.TOOL]

    for call_index, span in enumerate(tool_spans):
        details = span.tool_call
        result = None if span.output is UNSET else span.output
        returned_data = details.returned_data if details is not None else UNSET
        if returned_data is not UNSET:
            if isinstance(result, Mapping):
                if profile.returned_data_key not in result:
                    result = {**result, profile.returned_data_key: returned_data}
            else:
                warnings.warn(
                    f"call {(details.call_id if details else span.span_id)!r} in trace "
                    f"{trace.id!r} sets 'returned_data' but its result is not a JSON object, "
                    "so anomaly-and-pattern analysis cannot read it; "
                    "returned_data_false_rate will stay 0 for this call. Wrap the result "
                    'as {"content": ...} to make it count.',
                    UserWarning,
                    stacklevel=2,
                )
        arguments = span.input if isinstance(span.input, Mapping) else {}
        calls.append(
            NormalizedCall(
                call_id=str(details.call_id if details and details.call_id else span.span_id),
                call_index=(
                    details.index
                    if details is not None and details.index is not None
                    else call_index
                ),
                tool_name=str(span.tool_name or span.name or ""),
                arguments=arguments,
                result=result,
                duration_ms=_duration_ms(span),
                source_pointer=dict(span.source_pointer),
            )
        )

    steps = tuple(
        NormalizedStep(
            step_index=index,
            step_type=span.subtype or _native_step_type(span),
            name=str(span.name or span.tool_name or ""),
            content=span.summary if span.summary is not None else _span_content(span),
            source_pointer=dict(span.source_pointer),
        )
        for index, span in enumerate(trace.spans)
    )
    return NormalizedTrace(
        trace_id=trace.id,
        calls=tuple(calls),
        steps=steps,
        source_pointer=dict(trace.source_pointer),
        observed_verdict=trace.observed_verdict,
        cost=trace.cost_usd,
        metrics=dict(trace.metrics),
    )


class AnomalyAndPatternsConfig(ContractModel):
    """Typed configuration owned by anomaly-and-pattern analysis."""

    contamination: float = CONTAMINATION
    input_scaling: Literal["none", "robust"] = "none"
    cluster_candidates: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    minimum_independent_traces: int = Field(default=RECURRENCE_THRESHOLD, ge=1)
    feature_names: tuple[str, ...] | None = None

    @field_validator("contamination")
    @classmethod
    def contamination_is_finite_and_bounded(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 < value < 0.5:
            raise ValueError("contamination must be between 0 and 0.5")
        return value

    @field_validator("cluster_candidates")
    @classmethod
    def cluster_candidates_are_valid(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(candidate < 2 for candidate in value):
            raise ValueError("cluster_candidates must contain integers greater than or equal to 2")
        return value

    @field_validator("feature_names")
    @classmethod
    def feature_names_are_valid(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is not None and (not value or any(not name.strip() for name in value)):
            raise ValueError("feature_names must contain non-empty names")
        if value is not None and len(set(value)) != len(value):
            raise ValueError("feature_names must not contain duplicates")
        return value


@dataclass(frozen=True)
class AnomalyAndPatternsEvidenceStream:
    name = "anomaly-and-patterns"

    config: AnomalyAndPatternsConfig = field(default_factory=AnomalyAndPatternsConfig)
    profile: VenueProfile = DEFAULT_PROFILE

    def validate_configuration(self) -> None:
        if not isinstance(self.config, AnomalyAndPatternsConfig):
            raise TypeError("anomaly-and-patterns requires AnomalyAndPatternsConfig")

    def analyze(self, snapshot: TraceSnapshot) -> EvidenceStreamResult:
        traces = [to_ia2_trace(trace, profile=self.profile) for trace in snapshot.scan()]
        result = run_ia2(
            traces,
            feature_names=self.config.feature_names or DEFAULT_FEATURES,
            contamination=self.config.contamination,
            cluster_candidates=self.config.cluster_candidates,
            minimum_independent_traces=self.config.minimum_independent_traces,
            input_scaling=self.config.input_scaling,
            profile=self.profile,
        )
        problems = problems_from_analysis(result)
        return EvidenceStreamResult(
            stream_name=self.name,
            problems=problems,
            artifacts=AnomalyAndPatternsArtifacts(
                result=result,
                config=self.config,
            ),
        )
