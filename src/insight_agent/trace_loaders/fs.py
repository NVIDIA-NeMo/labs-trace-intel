"""Load canonical ``Trace`` objects from JSONL files."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from insight_agent.trace_loaders.trace_loaders import TraceDescription
from insight_agent.traces import Span, SpanKind, Trace, TraceSnapshot

__all__ = ["FSDataLoadError", "FSDataLoader"]


class FSDataLoadError(ValueError):
    """A filesystem trace corpus is not valid canonical Trace JSONL."""


def _spans(trace: Trace) -> Iterator[Span]:
    pending = list(reversed(trace.root_spans))
    while pending:
        span = pending.pop()
        yield span
        pending.extend(reversed(span.children))


def _parse_trace(raw: str, *, path: Path, line_number: int) -> Trace:
    try:
        return Trace.model_validate_json(raw)
    except ValidationError as error:
        problem = (
            "malformed JSON"
            if any(detail["type"] == "json_invalid" for detail in error.errors())
            else "invalid trace"
        )
        raise FSDataLoadError(f"{path}:{line_number}: {problem}: {error}") from error


@dataclass
class FSDataLoader:
    """Read one canonical ``Trace`` JSON object per non-empty line."""

    path: Path
    _snapshot: TraceSnapshot | None = field(default=None, init=False, repr=False)
    _description: TraceDescription | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def load(self) -> TraceSnapshot:
        if self._snapshot is not None:
            return self._snapshot

        traces: list[Trace] = []
        with self.path.open(encoding="utf-8") as corpus:
            for line_number, raw in enumerate(corpus, start=1):
                if raw.strip():
                    traces.append(_parse_trace(raw, path=self.path, line_number=line_number))
        if not traces:
            raise FSDataLoadError(f"{self.path}: contains no traces")
        try:
            self._snapshot = TraceSnapshot(traces)
        except ValueError as error:
            raise FSDataLoadError(f"{self.path}: {error}") from error
        self._description = {
            "source": f"fs:{self.path.resolve()}",
            "trace_count": len(traces),
            "call_count": sum(
                span.kind is SpanKind.TOOL for trace in traces for span in _spans(trace)
            ),
            "distinct_logical_cases": len(
                {str(trace.attributes.get("logical_case_id") or trace.id) for trace in traces}
            ),
        }
        return self._snapshot

    def describe(self) -> TraceDescription:
        self.load()
        assert self._description is not None
        return self._description
