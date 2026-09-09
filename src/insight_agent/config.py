# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
from insight_agent.evidence_streams.tool_issues.stream import ToolIssueConfig

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

    ethos_divergence: EthosDivergenceConfig | None = None

    @model_validator(mode="after")
    def at_least_one_stream_is_configured(self) -> EvidenceStreamsConfig:
        if all(
            stream is None
            for stream in (self.anomaly_and_patterns, self.tool_issues, self.ethos_divergence)
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

    model: str | None = None
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
