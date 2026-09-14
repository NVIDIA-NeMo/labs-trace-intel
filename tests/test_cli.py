# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from nooa.unifiedllm import FakeLLMClient

import insight_agent.cli.main as cli
from insight_agent.config import EvidenceStreamsConfig
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult, Problem
from insight_agent.evidence_streams.registry import EvidenceStreamRegistry
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
    captured = capsys.readouterr()
    assert output_path.read_text(encoding="utf-8") == captured.out
    assert captured.err.splitlines() == [
        "[insight-agent] Loading traces from canonical JSONL...",
        "[insight-agent] Loaded 1 trace with 0 tool calls across 1 logical case from canonical JSONL.",
        "[insight-agent] Running 1 evidence stream: tool issues.",
        '[insight-agent] Evidence stream "tool issues" found 0 candidate problems (1/1 complete).',
        "[insight-agent] Synthesizing 0 candidate problems and 1 existing insight into final insights...",
        "[insight-agent] Generated 1 final insight.",
        f"[insight-agent] Wrote 1 final insight to {output_path}.",
    ]
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


def test_evidence_progress_reports_findings_as_streams_finish(monkeypatch):
    release_slow_stream = asyncio.Event()

    class FakeRegistry:
        names = ("slow-stream", "fast-stream")

        async def analyze(self, name, snapshot):
            assert snapshot.trace_count == 0
            if name == "slow-stream":
                await asyncio.wait_for(release_slow_stream.wait(), timeout=1)
            else:
                release_slow_stream.set()
            return EvidenceStreamResult(
                stream_name=name,
                problems=(
                    Problem(
                        description=f"{name} found a recurring issue.",
                        supporting_trace_ids=("trace-1",),
                    ),
                ),
            )

    monkeypatch.setattr(cli, "registered_builtin_streams", lambda **kwargs: FakeRegistry())
    progress = []

    results = asyncio.run(
        cli._run_evidence_streams(
            EvidenceStreamsConfig(tool_issues={}), TraceSnapshot([]), progress=progress.append
        )
    )

    assert [result.stream_name for result in results] == ["slow-stream", "fast-stream"]
    assert progress == [
        'Evidence stream "fast stream" found 1 candidate problem (1/2 complete).',
        "  Candidate: fast-stream found a recurring issue.",
        'Evidence stream "slow stream" found 1 candidate problem (2/2 complete).',
        "  Candidate: slow-stream found a recurring issue.",
    ]


def test_evidence_streams_share_cli_loop_and_run_concurrently(monkeypatch):
    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        registry = EvidenceStreamRegistry()

        class Stream:
            def __init__(self, name):
                self.name = name

            def validate_configuration(self):
                pass

            async def analyze(self, snapshot):
                assert asyncio.get_running_loop() is loop
                if self.name == "first":
                    await asyncio.wait_for(started.wait(), timeout=1)
                else:
                    started.set()
                return EvidenceStreamResult(stream_name=self.name, problems=())

        for stream in (Stream("first"), Stream("second")):
            registry.register(stream)
        monkeypatch.setattr(cli, "registered_builtin_streams", lambda **kwargs: registry)
        results = await cli._run_evidence_streams(
            EvidenceStreamsConfig(tool_issues={}), TraceSnapshot([])
        )
        assert [result.stream_name for result in results] == ["first", "second"]

    asyncio.run(run())
