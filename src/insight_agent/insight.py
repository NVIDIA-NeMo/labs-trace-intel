"""The typed Insight product contract."""

from pydantic import BaseModel, Field


class Insight(BaseModel):
    """A validated, actionable problem found in the agent's traces."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    trace_refs: list[str] = Field(min_length=1)


__all__ = ["Insight"]
