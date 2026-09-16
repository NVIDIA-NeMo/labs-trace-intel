# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO
from pathlib import Path

from rich.console import Console

from insight_agent.cli.output import RunOutput, RunResult
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult


def test_report_separates_completed_and_skipped_analyses():
    stream = StringIO()
    output = RunOutput(Console(file=stream))
    output.report(
        RunResult(
            10,
            [
                EvidenceStreamResult(
                    stream_name="tool-issues", problems=(), limitations=("Missing schemas",)
                ),
                EvidenceStreamResult(
                    stream_name="ethos-divergence", problems=(), skip_reason="No ethos document"
                ),
            ],
            [],
        ),
        Path("insights.yml"),
    )

    completed, skipped = stream.getvalue().split("Skipped")
    assert "Completed" in completed and "Tool issues" in completed
    assert "Missing schemas" in completed and "Ethos divergence" not in completed
    assert "Ethos divergence" in skipped and "No ethos document" in skipped
    assert "evidence_streams to false" in skipped and "docs/checks.md" in skipped
    assert "Saved:" not in skipped
