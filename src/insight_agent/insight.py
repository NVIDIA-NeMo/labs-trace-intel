# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed Insight product contract and artifact loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, TypeAdapter


class Insight(BaseModel):
    """A validated, actionable problem found in the agent's traces."""

    id: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
        description="Opaque storage ID of an existing insight. Preserve it exactly; new insights have no ID.",
    )
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    trace_refs: list[str] = Field(min_length=2)


_INSIGHTS_ADAPTER = TypeAdapter(list[Insight])


def load_insights(path: Path) -> list[Insight]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    return _INSIGHTS_ADAPTER.validate_python(payload)


__all__ = ["Insight", "load_insights"]
