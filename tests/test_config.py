# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from insight_agent.config import RunConfig


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
