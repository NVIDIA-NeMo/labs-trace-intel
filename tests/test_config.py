# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from pathlib import Path

from insight_agent.config import EvidenceStreamsConfig, RunConfig


def test_cli_nested_override_preserves_yaml_siblings(tmp_path: Path) -> None:
    config_path = tmp_path / "analyst.yaml"
    config_path.write_text(
        """
trace:
  max_traces: 100
  mlflow_experiment:
    experiment: deep_research
    tracking_uri: https://example.test
evidence_streams:
  anomaly_and_patterns: {}
""".lstrip(),
        encoding="utf-8",
    )

    config = RunConfig(
        _cli_parse_args=[
            "--config",
            str(config_path),
            "--trace.max-traces",
            "5",
        ]
    )

    assert config.trace.max_traces == 5
    assert config.trace.mlflow_experiment is not None
    assert config.trace.mlflow_experiment.experiment == "deep_research"
    assert config.trace.mlflow_experiment.tracking_uri == "https://example.test"


def test_config_file_is_optional_when_cli_provides_trace_input() -> None:
    config = RunConfig(
        _cli_parse_args=[
            "--trace.filesystem.path",
            "traces.jsonl",
            "--evidence-streams.anomaly-and-patterns.contamination",
            "0.02",
            "--evidence-streams.eval-failure-patterns.max-tool-rounds",
            "72",
        ]
    )

    assert config.config is None
    assert config.trace.filesystem is not None
    assert config.trace.filesystem.path == Path("traces.jsonl")
    assert config.evidence_streams.eval_failure_patterns is not None
    assert config.evidence_streams.eval_failure_patterns.max_tool_rounds == 72


def test_cli_accepts_optional_code_base() -> None:
    config = RunConfig(
        _cli_parse_args=[
            "--trace.filesystem.path",
            "traces.jsonl",
            "--evidence-streams.anomaly-and-patterns.contamination",
            "0.02",
            "--code-base",
            "../agent-source",
        ]
    )

    assert config.code_base == Path("../agent-source")


def test_langfuse_export_yaml_with_cli_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "analyst.yaml"
    config_path.write_text(
        "trace:\n  langfuse_export:\n    path: exports\nevidence_streams:\n  tool_issues: {}\n",
        encoding="utf-8",
    )
    config = RunConfig(_cli_parse_args=["--config", str(config_path), "--trace.max-traces", "200"])
    assert config.trace.langfuse_export is not None
    assert config.trace.langfuse_export.path == Path("exports")
    assert config.trace.max_traces == 200


def test_langfuse_source_can_be_configured_entirely_through_cli() -> None:
    config = RunConfig(
        _cli_parse_args=[
            "--trace.langfuse.from-timestamp",
            "2026-08-01T00:00:00Z",
            "--trace.langfuse.to-timestamp",
            "2026-08-02T00:00:00Z",
            "--trace.langfuse.base-url",
            "https://langfuse.example.com",
            "--trace.max-traces",
            "25",
            "--evidence-streams.tool-issues",
            "{}",
        ]
    )

    assert config.trace.langfuse is not None
    assert config.trace.langfuse.from_timestamp == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert config.trace.langfuse.to_timestamp == datetime(2026, 8, 2, tzinfo=timezone.utc)
    assert config.trace.langfuse.base_url == "https://langfuse.example.com"
    assert config.trace.max_traces == 25


def test_streams_default_on_and_false_disables_only_that_stream(tmp_path):
    config = RunConfig(_cli_parse_args=["--trace.filesystem.path", "traces.jsonl"])
    assert all(
        getattr(config.evidence_streams, name) is not None
        for name in EvidenceStreamsConfig.model_fields
    )
    path = tmp_path / "config.yaml"
    path.write_text(
        "trace:\n  filesystem:\n    path: traces.jsonl\nevidence_streams:\n  ethos_divergence: false\n  tool_issues:\n    retry_threshold: 5\n"
    )
    config = RunConfig(
        _cli_parse_args=[
            "--config",
            str(path),
            "--evidence-streams.user-sentiment",
            "false",
            "--evidence-streams.eval-failure-patterns",
            "true",
        ]
    )
    assert config.evidence_streams.ethos_divergence is None
    assert config.evidence_streams.user_sentiment is None
    assert config.evidence_streams.tool_issues is not None
    assert config.evidence_streams.tool_issues.retry_threshold == 5
    assert config.evidence_streams.anomaly_and_patterns is not None
    assert config.evidence_streams.eval_failure_patterns is not None
