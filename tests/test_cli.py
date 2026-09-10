# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from nooa.unifiedllm import FakeLLMClient

import insight_agent.cli.main as cli
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult, Problem
from insight_agent.insight import Insight, load_insights
from insight_agent.traces import Trace, TraceAggregate, TraceSnapshot


def test_cli_compiles_selected_evidence_with_existing_insights(tmp_path, monkeypatch, capsys):
    existing = [
        Insight(
            name="Search omits archived documents",
            description="Archived documents disappear from search results.",
            trace_refs=["historical-trace"],
        )
    ]
    existing_path = tmp_path / "existing.json"
    existing_path.write_text(json.dumps([item.model_dump() for item in existing]), encoding="utf-8")
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text('{"id":"unscored","root_spans":[],"aggregate":{}}', encoding="utf-8")
    output_path = tmp_path / "results" / "insights.yml"
    compilation = SimpleNamespace(compile_insights=AsyncMock(return_value=existing))
    monkeypatch.setenv("INSIGHT_AGENT_API_KEY", "test-key-not-real")
    monkeypatch.setattr(cli, "_build_llm", lambda config, api_key: FakeLLMClient())
    monkeypatch.setattr(cli, "InsightCompilation", lambda llm: compilation)

    result = cli.main(
        [
            "--trace.filesystem.path",
            str(trace_path),
            "--evidence-streams.tool-issues.retry-threshold",
            "3",
            "--existing-insights",
            str(existing_path),
            "--output-path",
            str(output_path),
        ]
    )

    evidence, _, prior = compilation.compile_insights.await_args.args
    assert [item.stream_name for item in evidence] == ["tool-issues"]
    assert prior == existing
    assert result == cli.EXIT_OK
    assert output_path.read_text(encoding="utf-8") == capsys.readouterr().out
    assert load_insights(output_path) == existing


def test_code_validation_filters_problems_and_preserves_stream_result(
    tmp_path, monkeypatch
) -> None:
    trace = Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())
    snapshot = TraceSnapshot([trace])
    supported = Problem(description="Supported", supporting_trace_ids=("trace-1",))
    unsupported = Problem(description="Unsupported", supporting_trace_ids=("trace-1",))
    unknown = Problem(description="Unknown", supporting_trace_ids=("trace-1",))
    evidence = [
        EvidenceStreamResult(
            stream_name="test-stream",
            problems=(supported, unsupported, unknown),
            artifacts={"kept": True},
        )
    ]
    received = []

    class FakeValidator:
        def __init__(self, code_base_path, llm):
            assert code_base_path == tmp_path.resolve()

        async def is_supported(self, problem, supporting_traces):
            received.append((problem, supporting_traces))
            return {"Supported": True, "Unsupported": False, "Unknown": None}[problem.description]

    monkeypatch.setattr(cli, "ProblemValidation", FakeValidator)

    result = asyncio.run(
        cli._validate_evidence_with_code(evidence, snapshot, tmp_path, FakeLLMClient())
    )

    assert result[0].problems == (supported, unknown)
    assert result[0].artifacts == {"kept": True}
    assert received == [
        (supported, (trace,)),
        (unsupported, (trace,)),
        (unknown, (trace,)),
    ]
