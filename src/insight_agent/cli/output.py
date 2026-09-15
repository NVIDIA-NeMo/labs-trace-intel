# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transient progress and a single, outcome-first report for an analysis run."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult
from insight_agent.insight import Insight

_STREAM_LABELS = {
    "anomaly-and-patterns": "Anomalies and patterns",
    "tool-issues": "Tool issues",
    "ethos-divergence": "Ethos divergence",
    "eval-failure-patterns": "Evaluation failures",
    "user-sentiment": "User sentiment",
}
_LOG_INTERVAL = 10


def display_name(name: str) -> str:
    return _STREAM_LABELS.get(name, name.replace("-", " ").capitalize())


def count(value: int, singular: str) -> str:
    return f"{value:,} {singular}{'' if value == 1 else 's'}"


@dataclass
class RunResult:
    trace_count: int
    evidence: list[EvidenceStreamResult]
    insights: list[Insight]
    existing_insights: list[Insight] = field(default_factory=list)
    rejected_by_code: int = 0


class RunOutput:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.trace_count: int | None = None
        self.activity = "Loading traces"
        self.started = time.monotonic()

    def render_progress(self) -> Group:
        lines = []
        if self.trace_count is not None:
            lines.append(Text(f"Loaded {count(self.trace_count, 'trace')}."))
        elapsed = f" · {int(time.monotonic() - self.started)}s"
        activity = Text(self.activity, no_wrap=True, overflow="ellipsis")
        activity.truncate(max(0, self.console.width - len(elapsed)), overflow="ellipsis")
        activity.append(elapsed)
        lines.append(activity)
        return Group(*lines)

    @asynccontextmanager
    async def progress(self) -> AsyncIterator[None]:
        if self.console.is_terminal and not self.console.is_dumb_terminal:
            with Live(
                console=self.console,
                get_renderable=self.render_progress,
                transient=True,
                refresh_per_second=4,
                redirect_stdout=False,
                redirect_stderr=False,
            ):
                yield
        else:
            # Logs and pipes get occasional status lines without terminal control codes.
            async def heartbeat() -> None:
                while True:
                    await asyncio.sleep(_LOG_INTERVAL)
                    self.console.print(f"{self.activity} · {int(time.monotonic() - self.started)}s")

            task = asyncio.create_task(heartbeat())
            try:
                yield
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def report(self, result: RunResult, output_path: Path) -> None:
        previous = {(item.name, item.description) for item in result.existing_insights}
        new_count = sum((item.name, item.description) not in previous for item in result.insights)
        if new_count:
            outcome = f"Produced {count(new_count, 'insight')}"
        else:
            outcome = "No new insights produced" if previous else "No insights produced"
        self.console.print(f"{outcome} from {count(result.trace_count, 'trace')}.")
        candidate_count = sum(len(item.problems) for item in result.evidence)
        if not result.insights and candidate_count:
            reason = (
                "none were supported by the codebase"
                if result.rejected_by_code == candidate_count
                else "none were retained after validation and review"
                if result.rejected_by_code
                else "none met the actionability criteria"
            )
            self.console.print(f"{count(candidate_count, 'candidate issue')} found; {reason}.")
        retained = len(result.insights) - new_count
        if retained:
            self.console.print(f"{count(retained, 'existing insight')} retained.")

        name_width = (
            max((len(display_name(item.stream_name)) for item in result.evidence), default=0) + 2
        )
        for heading, skipped in (("Completed", False), ("Skipped", True)):
            rows = [item for item in result.evidence if (item.skip_reason is not None) == skipped]
            if not rows:
                continue
            self.console.print(f"\n{heading}", style="bold")
            table = Table.grid(padding=(0, 2))
            table.add_column(width=name_width)
            table.add_column(ratio=1)
            for item in rows:
                if item.skip_reason is not None:
                    detail = item.skip_reason
                else:
                    detail = (
                        count(len(item.problems), "candidate issue")
                        if item.problems
                        else f"{count(item.finding_count, 'finding')}; no candidate issues"
                        if item.finding_count
                        else "No findings"
                    )
                    if item.limitations:
                        detail += "; " + "; ".join(item.limitations)
                table.add_row(Text(f"  {display_name(item.stream_name)}"), Text(detail))
            self.console.print(table)
        if result.insights and output_path != Path("-"):
            self.console.print(f"\nSaved: {output_path}", soft_wrap=True)
