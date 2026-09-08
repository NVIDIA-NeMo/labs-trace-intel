# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated, shareable configuration for a complete Analyst run.

The CLI remains useful for one-off invocations, while this module gives a
team one durable configuration shape to put under version control. Secrets are
intentionally excluded from the YAML configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_settings import SettingsConfigDict

from insight_agent.evidence_streams.anomaly_and_patterns.stream import AnomalyAndPatternsConfig
from insight_agent.evidence_streams.tool_issues.stream import ToolIssueConfig
from insight_agent.insights_generation.defaults import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_ROUNDS,
)


class ConfigModel(BaseModel):
    """Strict base for the shareable YAML contract."""

    model_config = SettingsConfigDict(
        extra="forbid",
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_hide_none_type=True,
        cli_avoid_json=True,
        cli_use_class_docs_for_groups=True,
    )


class FilesystemConfig(ConfigModel):
    """Settings owned by the canonical JSONL filesystem loader."""

    path: Path


class MLflowExperimentConfig(ConfigModel):
    """Settings owned by the live MLflow experiment loader."""

    experiment: str
    tracking_uri: str | None = None
    filter: str | None = None


class MLflowExportConfig(ConfigModel):
    """Settings owned by the native MLflow export loader."""

    path: Path


class TraceConfig(ConfigModel):
    """Shared trace settings and exactly one configured loader."""

    max_traces: int | None = Field(
        default=None,
        ge=1,
        description="Maximum traces loaded by an MLflow source",
    )
    filesystem: FilesystemConfig | None = Field(
        default=None,
        description="Canonical JSONL filesystem loader",
    )
    mlflow_experiment: MLflowExperimentConfig | None = Field(
        default=None,
        description="Live MLflow experiment loader",
    )
    mlflow_export: MLflowExportConfig | None = Field(
        default=None,
        description="Native MLflow export loader",
    )

    @model_validator(mode="after")
    def source_has_required_settings(self) -> TraceConfig:
        configured = sum(
            source is not None
            for source in (self.filesystem, self.mlflow_experiment, self.mlflow_export)
        )
        if configured != 1:
            raise ValueError("trace must configure exactly one loader")
        if self.filesystem is not None and self.max_traces is not None:
            raise ValueError("trace.max_traces is not supported by the filesystem loader")
        return self


class OutputConfig(ConfigModel):
    """Filesystem and console output choices."""

    directory: Path = Field(default=Path("out"), description="Output directory")
    quiet: bool = Field(default=False, description="Suppress the stdout summary")


class AnomalyAndPatternsStreamConfig(AnomalyAndPatternsConfig):
    """Strict YAML settings for anomaly-and-pattern evidence."""

    model_config = ConfigDict(extra="forbid")


class ToolIssueStreamConfig(ToolIssueConfig):
    """Strict YAML settings for deterministic tool-issue evidence."""

    model_config = ConfigDict(extra="forbid")


class EvidenceStreamsConfig(ConfigModel):
    """Selected deterministic evidence streams and their configuration."""

    anomaly_and_patterns: AnomalyAndPatternsStreamConfig | None = Field(
        default=None,
        description="Anomaly and recurring-pattern evidence",
    )
    tool_issues: ToolIssueStreamConfig | None = Field(
        default=None,
        description="Deterministic tool-issue evidence",
    )

    @model_validator(mode="after")
    def at_least_one_stream_is_configured(self) -> EvidenceStreamsConfig:
        if self.anomaly_and_patterns is None and self.tool_issues is None:
            raise ValueError("at least one evidence stream must be configured")
        return self


class AnalystGenerationConfig(ConfigModel):
    """Settings that control LLM-authored Insight generation."""

    model: str | None = None
    api_base: str | None = None
    env_file: Path | None = None
    existing_insights: Path | None = Field(
        default=None,
        description="Existing insights.json to reconcile with newly generated Insights",
    )
    max_tool_rounds: int = Field(default=DEFAULT_MAX_TOOL_ROUNDS, ge=0)
    temperature: float | None = None
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1)


class AnalystConfig(AnalystGenerationConfig):
    """Run selection plus settings for the optional LLM-authored stage."""

    enabled: bool = True


class RunConfig(ConfigModel):
    """The complete public configuration contract for an analysis run."""

    trace: TraceConfig = Field(description="Trace source and loading settings")
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output settings",
    )
    evidence_streams: EvidenceStreamsConfig = Field(
        description="Evidence-stream selection and settings"
    )
    analyst: AnalystConfig = Field(
        default_factory=AnalystConfig,
        description="LLM-authored Analyst settings",
    )


def load_run_config(path: Path | str) -> RunConfig:
    """Load one YAML mapping and validate it as a :class:`RunConfig`."""

    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"configuration in {source} must be a YAML mapping")
    try:
        config = RunConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid configuration in {source}: {exc}") from exc
    base = source.parent.resolve()

    def resolve_path(value: Path | None) -> Path | None:
        if value is None or value.is_absolute() or value == Path("-"):
            return value
        return base / value

    filesystem = config.trace.filesystem
    if filesystem is not None:
        filesystem = filesystem.model_copy(update={"path": resolve_path(filesystem.path)})
    export = config.trace.mlflow_export
    if export is not None:
        export = export.model_copy(update={"path": resolve_path(export.path)})
    trace = config.trace.model_copy(update={"filesystem": filesystem, "mlflow_export": export})
    output = config.output.model_copy(update={"directory": resolve_path(config.output.directory)})
    analyst = config.analyst.model_copy(
        update={
            "env_file": resolve_path(config.analyst.env_file),
            "existing_insights": resolve_path(config.analyst.existing_insights),
        }
    )
    return config.model_copy(update={"trace": trace, "output": output, "analyst": analyst})


def dump_run_config(config: RunConfig) -> dict[str, Any]:
    """Return a JSON-safe resolved view suitable for review or automation."""

    return config.model_dump(mode="json", exclude_none=True)
