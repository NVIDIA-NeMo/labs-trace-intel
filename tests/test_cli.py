# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from nooa.unifiedllm import FakeLLMClient

import insight_agent.cli.main as cli
from insight_agent.config import EvidenceStreamsConfig, RunConfig
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult, Problem
from insight_agent.evidence_streams.registry import EvidenceStreamRegistry
from insight_agent.insight import Insight, load_insights
from insight_agent.insights_generation.config import load_dotenv
from insight_agent.traces import Trace, TraceAggregate, TraceSnapshot


@pytest.fixture
def clean_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.os, "environ", {})
    monkeypatch.setattr(cli, "load_dotenv", lambda: load_dotenv(tmp_path / ".env"))


def test_missing_environment_exits_before_loading_traces(
    clean_environment, monkeypatch, tmp_path, capsys
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "trace:\n  langsmith:\n    project: production\n"
        "evidence_streams:\n  user_sentiment:\n    litellm:\n"
        "      model: openai/embedding\n      api_key_env: EMBEDDING_API_KEY\n"
    )
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_API_KEY", " \t")
    loader = Mock()
    monkeypatch.setattr(cli, "_configured_trace_loader", loader)
    output = tmp_path / "insights.yml"

    assert cli.main(["--config", str(config_path), "--output-path", str(output)]) == 2

    loader.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Missing required environment settings:\n"
        "  INSIGHT_AGENT_API_KEY — API key for the configured model\n"
        "  LANGSMITH_API_KEY — API key for LangSmith\n"
        "  EMBEDDING_API_KEY — API key named by evidence_streams.user_sentiment.litellm.api_key_env\n\n"
        "Set these in .env in your working directory (NAME=value),\n"
        "or export them in your shell, then rerun the command.\n"
    )
    assert not output.exists()


def test_langfuse_checks_only_missing_settings_after_cli_overrides(
    clean_environment, monkeypatch, tmp_path, capsys
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "trace:\n  langfuse:\n"
        "    from_timestamp: 2026-01-01T00:00:00Z\n"
        "    to_timestamp: 2026-01-02T00:00:00Z\n"
        "evidence_streams:\n  tool_issues: {}\n"
    )
    monkeypatch.setenv("INSIGHT_AGENT_API_KEY", "private-model-key")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "private-project-key")
    args = ["--config", str(config_path)]
    assert cli.main(args) == 2
    error = capsys.readouterr().err
    assert "LANGFUSE_SECRET_KEY" in error
    assert "LANGFUSE_BASE_URL" in error
    assert "LANGFUSE_PUBLIC_KEY" not in error
    assert "private-" not in error

    args += ["--trace.langfuse.base-url", "https://langfuse.example.com"]
    assert cli.main(args) == 2
    assert "LANGFUSE_BASE_URL" not in capsys.readouterr().err
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "private-secret-key")
    assert cli._check_environment(cli.get_config(args)) == "private-model-key"


def test_environment_accepts_dotenv_and_existing_key_aliases(
    clean_environment, monkeypatch, tmp_path
):
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=dotenv-model-key\nLANGCHAIN_API_KEY=dotenv-trace-key\n"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "exported-model-key")
    config = RunConfig(
        trace={"langsmith": {"project": "production"}}, evidence_streams={"tool_issues": {}}
    )

    assert cli._check_environment(config) == "exported-model-key"


def test_cli_preserves_existing_insights_without_synthesizing_empty_evidence(
    clean_environment, tmp_path, monkeypatch, capsys, select_streams
):
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
            "--evidence-streams",
            json.dumps(select_streams(tool_issues={"retry_threshold": 3})),
            "--existing-insights",
            str(existing_path),
            "--output-path",
            str(output_path),
        ]
    )

    compilation.compile_insights.assert_not_awaited()
    assert result == cli.EXIT_OK
    captured = capsys.readouterr()
    assert output_path.read_text(encoding="utf-8") == captured.out
    assert "No new insights produced from 1 trace." in captured.err
    assert "1 existing insight retained." in captured.err
    assert "Skipped" in captured.err
    assert "Tool issues" in captured.err and "No tool calls" in captured.err
    assert f"Saved: {output_path}" in captured.err
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


def test_evidence_streams_share_cli_loop_and_run_concurrently(monkeypatch):
    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        registry = EvidenceStreamRegistry()

        class Stream:
            def __init__(self, name):
                self.name = name

            def check_prerequisites(self, snapshot):
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
        progress = []
        results = await cli._run_evidence_streams(
            EvidenceStreamsConfig(tool_issues={}),
            TraceSnapshot([Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())]),
            progress=progress.append,
        )
        assert [result.stream_name for result in results] == ["first", "second"]
        assert set().union(*map(set, progress)) == {"first", "second"}
        assert progress[-1] == ()

    asyncio.run(run())


def test_no_candidates_skips_synthesis_and_file_creation(
    clean_environment, tmp_path, monkeypatch, capsys, select_streams
):
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"id":"trace-1","root_spans":[],"aggregate":{}}')
    output = tmp_path / "insights.yml"
    compilation = Mock()
    monkeypatch.setenv("INSIGHT_AGENT_API_KEY", "test-key")
    monkeypatch.setattr(cli, "InsightCompilation", compilation)

    assert (
        cli.main(
            [
                "--trace.filesystem.path",
                str(traces),
                "--evidence-streams",
                json.dumps(select_streams(tool_issues=True)),
                "--output-path",
                str(output),
            ]
        )
        == 0
    )
    compilation.assert_not_called()
    assert not output.exists()
    captured = capsys.readouterr()
    assert captured.out == "[]\n"
    assert "Saved:" not in captured.err
