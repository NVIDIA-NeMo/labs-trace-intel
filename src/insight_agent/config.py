# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from insight_agent.evidence_streams.anomaly_and_patterns.stream import AnomalyAndPatternsConfig
from insight_agent.evidence_streams.ethos_divergence.ethos_divergence_detector import (
    EthosDivergenceConfig,
)
from insight_agent.evidence_streams.eval_failure_patterns import EvalFailurePatternsConfig
from insight_agent.evidence_streams.tool_issues.stream import ToolIssueConfig
from insight_agent.trace_loaders.intake import IntakeTraceLoaderConfig
from insight_agent.trace_loaders.langfuse import validate_langfuse_time_window

_CONFIG_MODEL_SETTINGS = SettingsConfigDict(
    extra="forbid",
    cli_kebab_case=True,
    cli_implicit_flags=True,
    cli_hide_none_type=True,
    cli_avoid_json=True,
    cli_use_class_docs_for_groups=True,
)


class ConfigModel(BaseModel):
    """Strict base for the shareable YAML contract."""

    model_config = _CONFIG_MODEL_SETTINGS


class _ConfigFileSettingsSource(YamlConfigSettingsSource):
    """Load YAML from the path already parsed from the higher-priority CLI source."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        # Defer reading until Pydantic has populated ``current_state`` from the
        # CLI source, which is automatically given the highest priority.
        super().__init__(settings_cls, yaml_file=None)

    def __call__(self) -> dict[str, object]:
        config_path = self.current_state.get("config")
        if config_path is None:
            return {}
        return YamlConfigSettingsSource(self.settings_cls, yaml_file=config_path)()


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


class LangfuseExportConfig(ConfigModel):
    """Settings owned by the native Langfuse v3 export loader."""

    path: Path


class LangfuseConfig(ConfigModel):
    """Settings owned by the live Langfuse v3 trace loader."""

    base_url: str | None = None
    from_timestamp: datetime
    to_timestamp: datetime
    filter: str | None = None

    @model_validator(mode="after")
    def time_window_is_valid(self) -> LangfuseConfig:
        validate_langfuse_time_window(self.from_timestamp, self.to_timestamp)
        return self


class LangSmithTraceSourceConfig(ConfigModel):
    """Settings owned by the LangSmith Trace Loader."""

    project: str
    api_url: str | None = None
    filter: str | None = None
    tree_filter: str | None = None
    start_time: datetime | None = None

    @model_validator(mode="after")
    def start_time_has_timezone(self) -> LangSmithTraceSourceConfig:
        if self.start_time is not None and (
            self.start_time.tzinfo is None or self.start_time.utcoffset() is None
        ):
            raise ValueError("start_time must include a timezone")
        return self


class LangSmithTraceExportFileSourceConfig(ConfigModel):
    """Settings owned by the LangSmith Trace Export File Loader."""

    path: Path


class TraceConfig(ConfigModel):
    """Shared trace settings and exactly one configured loader."""

    max_traces: int | None = Field(
        default=None,
        ge=1,
        description="Maximum complete traces loaded by a provider source",
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
    intake: IntakeTraceLoaderConfig | None = Field(
        default=None,
        description="Bounded NeMo Platform Intake query",
    )
    langfuse: LangfuseConfig | None = Field(
        default=None,
        description="Live Langfuse v3 trace loader",
    )
    langfuse_export: LangfuseExportConfig | None = Field(
        default=None,
        description="Native Langfuse v3 trace-detail export loader",
    )
    langsmith: LangSmithTraceSourceConfig | None = Field(
        default=None,
        description="LangSmith Trace Loader",
    )
    langsmith_trace_export_file: LangSmithTraceExportFileSourceConfig | None = Field(
        default=None,
        description="LangSmith Trace Export File Loader",
    )

    @model_validator(mode="after")
    def source_has_required_settings(self) -> TraceConfig:
        configured = sum(
            source is not None
            for source in (
                self.filesystem,
                self.mlflow_experiment,
                self.mlflow_export,
                self.intake,
                self.langfuse,
                self.langfuse_export,
                self.langsmith,
                self.langsmith_trace_export_file,
            )
        )
        if configured != 1:
            raise ValueError("trace must configure exactly one loader")
        if self.filesystem is not None and self.max_traces is not None:
            raise ValueError("trace.max_traces is not supported by the filesystem loader")
        return self


class AnomalyAndPatternsStreamConfig(AnomalyAndPatternsConfig):
    """Strict YAML settings for anomaly-and-pattern evidence."""

    model_config = ConfigDict(extra="forbid")


class ToolIssueStreamConfig(ToolIssueConfig):
    """Strict YAML settings for deterministic tool-issue evidence."""

    model_config = ConfigDict(extra="forbid")


class EvidenceStreamsConfig(ConfigModel):
    """Selected evidence streams and their configuration."""

    anomaly_and_patterns: AnomalyAndPatternsStreamConfig | None = Field(
        default=None,
        description="Anomaly and recurring-pattern evidence",
    )
    tool_issues: ToolIssueStreamConfig | None = Field(
        default=None,
        description="Deterministic tool-issue evidence",
    )
    eval_failure_patterns: EvalFailurePatternsConfig | None = Field(
        default=None,
        description="LLM review of evaluation-linked failures",
    )

    ethos_divergence: EthosDivergenceConfig | None = None

    @model_validator(mode="after")
    def at_least_one_stream_is_configured(self) -> EvidenceStreamsConfig:
        if all(
            stream is None
            for stream in (
                self.anomaly_and_patterns,
                self.tool_issues,
                self.ethos_divergence,
                self.eval_failure_patterns,
            )
        ):
            raise ValueError("at least one evidence stream must be configured")
        return self


class RunConfig(BaseSettings):
    """The complete public configuration contract for an analysis run."""

    model_config = _CONFIG_MODEL_SETTINGS

    config: Path | None = Field(
        default=None,
        exclude=True,
        description="Optional YAML Analyst configuration",
    )
    trace: TraceConfig = Field(description="Trace source and loading settings")
    output_path: Path = Field(default=Path("insights.yml"), description="Output path")
    evidence_streams: EvidenceStreamsConfig = Field(
        description="Evidence-stream selection and settings"
    )
    code_base: Path | None = Field(
        default=None,
        description="Optional local codebase used to validate trace-derived problems",
    )

    model: str | None = None
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum output tokens per model response. "
            "Set this when your model supports a smaller output limit."
        ),
    )
    api_base: str | None = None
    existing_insights: Path | None = Field(
        default=None,
        description="Existing insights.json to reconcile with newly generated Insights",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Read explicit values first, followed by the selected YAML configuration."""

        return init_settings, _ConfigFileSettingsSource(settings_cls)
