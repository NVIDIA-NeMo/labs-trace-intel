"""Loading and reconciling Insights from a previous run."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from nooa import Agent
from pydantic import ValidationError

from insight_agent.cli.main import EXIT_ERROR, EXIT_OK, main
from insight_agent.config import load_run_config
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult
from insight_agent.insight import Insight, load_insights
from insight_agent.traces import TraceSnapshot

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "insight_agent" / "data"
CORPUS = DATA_DIR / "sample_corpus.jsonl"


def write_existing_insights(path: Path) -> list[Insight]:
    insights = [
        Insight(
            name="Search omits archived documents",
            description="Archived documents disappear from otherwise complete search results.",
            trace_refs=["historical-trace"],
        )
    ]
    path.write_text(
        json.dumps([insight.model_dump(mode="json") for insight in insights]),
        encoding="utf-8",
    )
    return insights


def write_run_config(tmp_path: Path, existing_insights: Path) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    config = tmp_path / "analyst.yaml"
    config.write_text(
        f"""trace:
  filesystem:
    path: {CORPUS}
output:
  directory: {tmp_path / "out"}
  quiet: true
evidence_streams:
  anomaly_and_patterns: {{}}
  tool_issues: {{}}
analyst:
  enabled: true
  env_file: {env_file}
  existing_insights: {existing_insights}
""",
        encoding="utf-8",
    )
    return config


def test_load_insights_accepts_a_previous_output_artifact(tmp_path):
    path = tmp_path / "insights.json"
    expected = write_existing_insights(path)

    assert load_insights(path) == expected


@pytest.mark.parametrize("contents", ["not json", "{}", '[{"name": "missing fields"}]'])
def test_load_insights_rejects_invalid_artifacts(tmp_path, contents):
    path = tmp_path / "insights.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValidationError):
        load_insights(path)


def test_config_resolves_existing_insights_relative_to_yaml(tmp_path):
    config_file = tmp_path / "settings" / "analyst.yaml"
    config_file.parent.mkdir()
    config_file.write_text(
        f"""trace:
  filesystem:
    path: {CORPUS}
evidence_streams:
  anomaly_and_patterns: {{}}
analyst:
  existing_insights: previous/insights.json
""",
        encoding="utf-8",
    )

    config = load_run_config(config_file)

    assert config.analyst.existing_insights == config_file.parent / "previous" / "insights.json"


def test_run_reconciles_with_existing_insights(tmp_path, monkeypatch):
    path = tmp_path / "existing-insights.json"
    expected = write_existing_insights(path)
    calls: list[list[Insight]] = []

    class FakeInsightCompilation(Agent):
        async def compile_insights(
            self,
            evidence_streams: list[EvidenceStreamResult],
            trace_snapshot: TraceSnapshot,
            existing_insights: list[Insight],
        ) -> list[Insight]:
            """Compile validated insights from evidence."""

            calls.append(existing_insights)
            return existing_insights

    cli_main = importlib.import_module("insight_agent.cli.main")
    monkeypatch.setattr(cli_main, "InsightCompilation", FakeInsightCompilation)
    monkeypatch.setenv("INSIGHT_AGENT_API_KEY", "test-key-not-real")

    assert main(["--config", str(write_run_config(tmp_path, path))]) == EXIT_OK
    assert calls == [expected]

    output = tmp_path / "out" / "analyst"
    assert load_insights(output / "insights.json") == expected
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["existing_insights"] == {"path": str(path), "count": 1}


def test_invalid_existing_insights_fails_before_compilation(tmp_path, monkeypatch, capsys):
    path = tmp_path / "existing-insights.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("INSIGHT_AGENT_API_KEY", "test-key-not-real")

    assert main(["--config", str(write_run_config(tmp_path, path))]) == EXIT_ERROR
    assert "ValidationError" in capsys.readouterr().err
