# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed Insight product contract and artifact loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, TypeAdapter


class Insight(BaseModel):
    """A validated, actionable problem found in the agent's traces."""

    name: str = Field(min_length=1)
    description: str = Field(
        min_length=1,
        description=(
            "The finding: what actually happens in the traces and why, defensible "
            "directly from the referenced traces. Do not include a suggested fix here."
        ),
    )
    next_steps: str = Field(
        min_length=1,
        description=(
            "A feeder for further investigation, not a diagnosis or a fix: where "
            "a developer should start looking to pin down the underlying cause, "
            "and what areas of the agent might be involved. Point to plausible "
            "areas and open questions to explore -- for example the agent "
            "harness/orchestration, a specific tool, the prompt or instructions, "
            "ETHOS.md or other configuration, or the underlying model -- rather "
            "than asserting a root cause or prescribing a concrete change. Do "
            "not restate the finding here, and do not prescribe specific "
            "engineering steps like tests, functions, or files to change; leave "
            "the diagnosis and the fix to the developer's investigation."
        ),
    )
    trace_refs: list[str] = Field(min_length=1)


_INSIGHTS_ADAPTER = TypeAdapter(list[Insight])


def load_insights(path: Path) -> list[Insight]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    return _INSIGHTS_ADAPTER.validate_python(payload)


__all__ = ["Insight", "load_insights"]
