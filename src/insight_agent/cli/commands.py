"""Typed models that define the utility-command CLI surface."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import CliPositionalArg, CliSubCommand

from insight_agent.config import (
    AnalystConfig,
    AnalystGenerationConfig,
    AnomalyAndPatternsStreamConfig,
    ConfigModel,
    MLflowExperimentConfig,
    MLflowExportConfig,
    OutputConfig,
    ToolIssueStreamConfig,
)
from insight_agent.trace_loaders import MLFLOW_DEFAULT_MAX_TRACES


class TraceSourceCommandConfig(ConfigModel):
    """Trace source accepted by utilities that operate without a run YAML."""

    traces: CliPositionalArg[Path | None] = None
    mlflow_experiment: MLflowExperimentConfig | None = None
    mlflow_export: MLflowExportConfig | None = None
    max_traces: int = Field(default=MLFLOW_DEFAULT_MAX_TRACES, ge=1)

    @model_validator(mode="after")
    def exactly_one_source(self) -> TraceSourceCommandConfig:
        configured = sum(
            source is not None
            for source in (self.traces, self.mlflow_experiment, self.mlflow_export)
        )
        if configured != 1:
            raise ValueError("configure exactly one trace source")
        return self


class UtilityEvidenceConfig(ConfigModel):
    """Evidence settings shared by utilities that always run both streams."""

    anomaly_and_patterns: AnomalyAndPatternsStreamConfig = Field(
        default_factory=AnomalyAndPatternsStreamConfig
    )
    tool_issues: ToolIssueStreamConfig = Field(default_factory=ToolIssueStreamConfig)


class SchemaCommand(ConfigModel):
    """Print the canonical Trace JSON Schema."""

    out: Path | None = Field(default=None, description="Write the schema to this path")


class ValidateCommand(ConfigModel):
    """Validate a run configuration or canonical Trace JSONL corpus."""

    config: Path | None = Field(default=None, description="YAML run configuration to validate")
    traces: Path | None = Field(
        default=None, description="Canonical Trace JSONL corpus to validate"
    )
    json_output: bool = Field(default=False, description="Emit machine-readable JSON")

    @model_validator(mode="after")
    def exactly_one_target(self) -> ValidateCommand:
        if (self.config is None) == (self.traces is None):
            raise ValueError("configure exactly one of --config or --traces")
        return self


class CoverageCommand(ConfigModel):
    """Report which deterministic tool-issue rules a corpus can support."""

    traces: CliPositionalArg[Path]
    json_output: bool = Field(default=False, description="Emit machine-readable JSON")
    verbose: bool = Field(default=False, description="List every rule")
    with_findings: bool = Field(
        default=False,
        description="Run the tool-issue checks and mark which rules fired",
    )


class ExplainFailuresCommand(TraceSourceCommandConfig):
    """Compare failure decoding between evidence streams call by call."""

    only_disagreements: bool = False
    json_output: bool = Field(default=False, description="Emit machine-readable JSON")


class RunAnalystCommand(TraceSourceCommandConfig):
    """Author Insights from deterministic evidence using an LLM."""

    output: OutputConfig = Field(default_factory=OutputConfig)
    evidence_streams: UtilityEvidenceConfig = Field(default_factory=UtilityEvidenceConfig)
    analyst: AnalystGenerationConfig = Field(default_factory=AnalystGenerationConfig)
    digest: Path | None = Field(
        default=None,
        description="Use an existing anomaly-and-pattern digest.md",
    )
    cards: Path | None = Field(
        default=None,
        description="Use existing tool-issue cards.json",
    )
    dry_run: bool = Field(default=False, description="Write the prompt without making an API call")

    @model_validator(mode="after")
    def complete_analyst_request(self) -> RunAnalystCommand:
        if not self.analyst.agent:
            raise ValueError("analyst.agent is required")
        if bool(self.digest) != bool(self.cards):
            raise ValueError("digest and cards must be given together")
        return self


class DemoCommand(ConfigModel):
    """Run the complete pipeline on the bundled sample corpus."""

    output: OutputConfig = Field(default_factory=OutputConfig)
    evidence_streams: UtilityEvidenceConfig = Field(default_factory=UtilityEvidenceConfig)
    analyst: AnalystConfig = Field(default_factory=AnalystConfig)


class UtilityCommands(ConfigModel):
    """Utility commands that operate independently of a run YAML."""

    schema_command: CliSubCommand[SchemaCommand] = Field(alias="schema")
    validate_command: CliSubCommand[ValidateCommand] = Field(alias="validate")
    coverage: CliSubCommand[CoverageCommand]
    explain_failures: CliSubCommand[ExplainFailuresCommand]
    run_analyst: CliSubCommand[RunAnalystCommand]
    demo: CliSubCommand[DemoCommand]

    def cli_cmd(self) -> None:
        """Let :class:`CliApp` instantiate the selected command for dispatch."""


# Keep established concise spellings while the canonical generated options
# mirror their model paths. New fields need no entry here: their nested option
# is generated automatically.
UTILITY_CLI_SHORTCUTS: dict[str, str | list[str]] = {
    "json-output": "json",
    "verbose": "v",
    "mlflow-experiment.experiment": "mlflow-experiment",
    "mlflow-experiment.tracking-uri": "mlflow-tracking-uri",
    "mlflow-experiment.filter": "mlflow-filter",
    "mlflow-export.path": "mlflow-export",
    "output.directory": ["o", "out"],
    "output.quiet": "quiet",
    "evidence-streams.anomaly-and-patterns.contamination": "contamination",
    "evidence-streams.anomaly-and-patterns.input-scaling": "input-scaling",
    "evidence-streams.anomaly-and-patterns.cluster-candidates": "cluster-candidates",
    "evidence-streams.anomaly-and-patterns.minimum-independent-traces": ("min-independent-traces"),
    "evidence-streams.anomaly-and-patterns.feature-names": "feature",
    "evidence-streams.tool-issues.minimum-independent-cases": "min-independent-cases",
    "evidence-streams.tool-issues.retry-threshold": "retry-threshold",
    "evidence-streams.tool-issues.include-audit-problems": "all-cards",
    "analyst.agent": "agent",
    "analyst.model": "model",
    "analyst.api-base": "api-base",
    "analyst.env-file": "env-file",
    "analyst.prompt-version": "prompt-version",
    "analyst.max-tool-rounds": "max-tool-rounds",
    "analyst.temperature": "temperature",
    "analyst.max-tokens": "max-tokens",
}


UTILITY_COMMAND_NAMES = frozenset(
    (field.alias or name).replace("_", "-") for name, field in UtilityCommands.model_fields.items()
)
