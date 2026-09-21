# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed Insight product contract and artifact loader."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, TypeAdapter
from trace_ingest.source_links import SourceURL

from insight_agent.traces import TraceSnapshot


class Insight(BaseModel):
    """A validated, actionable problem found in the agent's traces."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    trace_refs: list[str] = Field(min_length=2)
    trace_links: dict[str, SourceURL] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
        description="Resolved source URLs keyed by trace ID; populated by the application.",
    )


def resolve_trace_links(
    insights: Sequence[Insight], snapshot: TraceSnapshot, existing: Sequence[Insight] = ()
) -> list[Insight]:
    """Attach only loader or previously saved links, never model-generated URLs."""
    known = {
        ref: url
        for insight in existing
        for ref, url in insight.trace_links.items()
        if ref in insight.trace_refs
    }
    resolved = {}
    for ref in dict.fromkeys(ref for insight in insights for ref in insight.trace_refs):
        try:
            url = snapshot.get_trace_by_id(ref).source_url
        except KeyError:
            url = known.get(ref)
        if url is not None:
            resolved[ref] = url
    return [
        insight.model_copy(
            update={
                "trace_links": {ref: resolved[ref] for ref in insight.trace_refs if ref in resolved}
            }
        )
        for insight in insights
    ]


_INSIGHTS_ADAPTER = TypeAdapter(list[Insight])


def load_insights(path: Path) -> list[Insight]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    return _INSIGHTS_ADAPTER.validate_python(payload)


__all__ = ["Insight", "load_insights"]
