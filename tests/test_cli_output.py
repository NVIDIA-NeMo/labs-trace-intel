# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

import insight_agent.cli.output as output
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult, Problem
from insight_agent.insight import Insight


def report(evidence, *, insights=(), previous=(), rejected=0, width=100, path="insights.yml"):
    stream = StringIO()
    ui = output.RunOutput(Console(file=stream, width=width, markup=False, highlight=False))
    ui.report(output.RunResult(10, evidence, list(insights), list(previous), rejected), Path(path))
    return "\n".join(line.rstrip() for line in stream.getvalue().splitlines())


def test_partial_run_has_one_outcome_and_each_limitation_once():
    result = report(
        [
            EvidenceStreamResult(stream_name="anomaly-and-patterns", problems=()),
            EvidenceStreamResult(
                stream_name="tool-issues",
                problems=(),
                limitations=("tool schemas missing in all 10 traces",),
            ),
            EvidenceStreamResult(
                stream_name="ethos-divergence", problems=(), skip_reason="No ethos document"
            ),
            EvidenceStreamResult(
                stream_name="eval-failure-patterns", problems=(), skip_reason="No evaluator results"
            ),
            EvidenceStreamResult(
                stream_name="user-sentiment",
                problems=(),
                skip_reason="No embedding backend configured",
            ),
        ]
    )
    assert (
        result
        == """No insights produced from 10 traces.

Completed
  Anomalies and patterns  No findings
  Tool issues             No findings; tool schemas missing in all 10 traces

Skipped
  Ethos divergence        No ethos document
  Evaluation failures     No evaluator results
  User sentiment          No embedding backend configured

Saved: insights.yml"""
    )


def test_partial_analysis_stays_completed_and_is_not_a_clean_bill_of_health():
    result = report(
        [
            EvidenceStreamResult(
                stream_name="anomaly-and-patterns",
                problems=(),
                limitations=("trajectory clustering — traces have identical trajectories",),
            )
        ],
        width=45,
    )
    assert "Completed" in result and "Skipped" not in result
    assert "identical trajectories" in " ".join(result.split())
    assert "No evidence found" not in result
    assert all(len(line) <= 45 for line in result.splitlines())


@pytest.mark.parametrize(
    "rejected,reason",
    [
        (0, "none met the actionability criteria"),
        (1, "none were retained after validation and review"),
        (3, "none were supported by the codebase"),
    ],
)
def test_candidates_rejected_are_distinguished_from_no_findings(rejected, reason):
    problem = Problem(description="Timeouts", supporting_trace_ids=("trace-1",))
    result = report(
        [EvidenceStreamResult(stream_name="tool-issues", problems=(problem,) * 3)],
        rejected=rejected,
    )
    assert f"3 candidate issues found; {reason}." in result
    assert "No findings" not in result


def test_raw_findings_do_not_get_counted_as_candidate_issues():
    result = report([EvidenceStreamResult(stream_name="tool-issues", problems=(), finding_count=3)])
    assert "3 findings; no candidate issues" in result
    assert "actionability criteria" not in result


def test_previous_insights_are_not_reported_as_new():
    old = Insight(name="Old issue", description="Existing", trace_refs=["old-trace"])
    new = Insight(name="New issue", description="New", trace_refs=["new-trace"])
    result = report([], insights=[old, new], previous=[old])
    assert result.startswith("Produced 1 insight from 10 traces.\n1 existing insight retained.")
    result = report([], insights=[old], previous=[old], path="-")
    assert result.startswith(
        "No new insights produced from 10 traces.\n1 existing insight retained."
    )
    assert "Saved:" not in result


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_progress_cleanup_and_terminal_behavior(monkeypatch, terminal, fails):
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = StringIO()
    console = Console(file=stream, force_terminal=terminal, color_system=None, width=70)
    ui = output.RunOutput(console)
    ui.trace_count = 10
    ui.activity = "Analyzing: tool issues"
    monkeypatch.setattr(output, "_LOG_INTERVAL", 0.001)

    async def run():
        try:
            async with ui.progress():
                await asyncio.sleep(0.02)
                if fails:
                    raise RuntimeError("model unavailable")
        except RuntimeError:
            assert fails
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    asyncio.run(run())
    text = stream.getvalue()
    assert "Analyzing: tool issues" in text
    assert "Completed" not in text and "Saved:" not in text
    if terminal:
        assert "\x1b[?25l" in text  # Hide cursor while the transient display is active.
        assert "\x1b[?25h" in text  # Always restore it, including on failure.
        assert "\x1b[2K" in text
    else:
        assert "\x1b" not in text
        assert "Loaded 10 traces" not in text  # Keep repeated trace counts out of log heartbeats.


def test_fast_nonterminal_run_only_prints_the_report():
    stream = StringIO()
    ui = output.RunOutput(Console(file=stream))

    async def run():
        async with ui.progress():
            ui.activity = "Analyzing: tool issues"

    asyncio.run(run())
    assert stream.getvalue() == ""


def test_narrow_live_status_keeps_elapsed_time_visible(monkeypatch):
    stream = StringIO()
    ui = output.RunOutput(Console(file=stream, width=35, color_system=None))
    ui.activity = "Analyzing: anomalies and patterns, tool issues, user sentiment"
    ui.started = 0
    monkeypatch.setattr(output.time, "monotonic", lambda: 12)
    ui.console.print(ui.render_progress())
    line = stream.getvalue().strip()
    assert line.endswith("· 12s")
    assert "…" in line
    assert len(line) <= 35


def test_merged_candidates_are_not_reported_as_rejected():
    old = Insight(name="Timeouts", description="Existing", trace_refs=["old-trace", "new-trace"])
    problem = Problem(description="Timeouts", supporting_trace_ids=("new-trace",))
    result = report(
        [EvidenceStreamResult(stream_name="tool-issues", problems=(problem,))],
        insights=[old],
        previous=[old],
    )
    assert "1 existing insight retained" in result
    assert "actionability criteria" not in result
