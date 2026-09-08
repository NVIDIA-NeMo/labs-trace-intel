"""The typed Insight product contract and artifact loader."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter


class Insight(BaseModel):
    """A validated, actionable problem found in the agent's traces."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    trace_refs: list[str] = Field(min_length=1)


_INSIGHTS_ADAPTER = TypeAdapter(list[Insight])


def load_insights(path: Path) -> list[Insight]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    return _INSIGHTS_ADAPTER.validate_json(text)


__all__ = ["Insight", "load_insights"]
