# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace

import yaml

from insight_agent.config import RunConfig
from insight_agent.insight import Insight, load_insights
from insight_agent.traces import TraceSnapshot


def _config(**overrides: object) -> RunConfig:
    values = {
        "trace": {"filesystem": {"path": "unused.jsonl"}},
        "evidence_streams": {"anomaly_and_patterns": {}},
        **overrides,
    }
    return RunConfig.model_validate(values)


def test_generation_carries_existing_insights_into_compilation(tmp_path, monkeypatch) -> None:
    existing = [
        Insight(
            name="Search omits archived documents",
            description="Archived documents disappear from search results.",
            trace_refs=["historical-trace"],
        )
    ]
    existing_path = tmp_path / "insights.json"
    existing_path.write_text(
        json.dumps([insight.model_dump(mode="json") for insight in existing]),
        encoding="utf-8",
    )
    received: list[list[Insight]] = []

    class FakeCompilation:
        async def compile_insights(self, evidence, snapshot, existing_insights):
            received.append(existing_insights)
            return existing_insights

    cli = importlib.import_module("insight_agent.cli.main")
    monkeypatch.setenv("INSIGHT_AGENT_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        cli,
        "_configured_trace_loader",
        lambda config: SimpleNamespace(load=lambda: TraceSnapshot([])),
    )

    async def fake_evidence(config, snapshot, llm):
        return []

    monkeypatch.setattr(cli, "_run_evidence_streams", fake_evidence)
    monkeypatch.setattr(
        cli, "_build_insight_compilation", lambda config, api_key: FakeCompilation()
    )

    result = asyncio.run(cli._generate_insights(_config(existing_insights=existing_path)))

    assert result == existing
    assert received == [existing]


def test_main_writes_and_prints_final_yaml(tmp_path, monkeypatch, capsys) -> None:
    generated = [
        Insight(
            name="Repeated timeout",
            description="Calls repeatedly time out.",
            trace_refs=["trace-1"],
        )
    ]
    output_path = tmp_path / "results" / "insights.yml"
    cli = importlib.import_module("insight_agent.cli.main")

    async def fake_generation(config):
        return generated

    monkeypatch.setattr(cli, "_generate_insights", fake_generation)

    result = cli.main(
        [
            "--trace.filesystem.path",
            "unused.jsonl",
            "--evidence-streams.anomaly-and-patterns.contamination",
            "0.02",
            "--output-path",
            str(output_path),
        ]
    )

    rendered = capsys.readouterr().out
    assert result == cli.EXIT_OK
    assert output_path.read_text(encoding="utf-8") == rendered
    assert load_insights(output_path) == generated
    assert yaml.safe_load(rendered) == [insight.model_dump(mode="json") for insight in generated]
