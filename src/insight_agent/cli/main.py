# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the Insights Analyst."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from nooa.unifiedllm import CompletionClient

from insight_agent.config import EvidenceStreamsConfig, RunConfig, TraceConfig
from insight_agent.evidence_streams.builtins import registered_builtin_streams
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult
from insight_agent.insight import Insight, load_insights
from insight_agent.insights_generation.config import (
    ENV_API_BASE,
    ENV_API_KEY,
    ENV_MODEL,
    load_dotenv,
    resolve,
)
from insight_agent.insights_generation.defaults import DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from insight_agent.insights_generation.insight_compilation import InsightCompilation
from insight_agent.trace_loaders.fs import FSDataLoader
from insight_agent.trace_loaders.langsmith import (
    LANGSMITH_DEFAULT_MAX_TRACES,
    LangSmithTraceConfig,
    LangSmithTraceExportFileConfig,
    LangSmithTraceExportFileLoader,
    LangSmithTraceLoader,
)
from insight_agent.trace_loaders.mlflow import (
    MLFLOW_DEFAULT_MAX_TRACES,
    MLflowFileTraceConfig,
    MLflowFileTraceLoader,
    MLflowTraceConfig,
    MLflowTraceLoader,
)
from insight_agent.trace_loaders.trace_loaders import TraceLoader
from insight_agent.traces import TraceSnapshot

EXIT_OK = 0
DEFAULT_REASONING_EFFORT = "high"


def _configured_trace_loader(config: TraceConfig) -> TraceLoader:
    """Construct the trace loader selected by the run configuration."""

    if config.filesystem is not None:
        return FSDataLoader(config.filesystem.path)
    if config.langsmith is not None:
        source = config.langsmith
        return LangSmithTraceLoader(
            LangSmithTraceConfig(
                project_name=source.project,
                api_url=source.api_url,
                filter=source.filter,
                tree_filter=source.tree_filter,
                start_time=source.start_time,
                max_traces=config.max_traces or LANGSMITH_DEFAULT_MAX_TRACES,
            )
        )
    if config.langsmith_trace_export_file is not None:
        return LangSmithTraceExportFileLoader(
            LangSmithTraceExportFileConfig(
                path=config.langsmith_trace_export_file.path,
                max_traces=config.max_traces or LANGSMITH_DEFAULT_MAX_TRACES,
            )
        )
    if config.mlflow_experiment is not None:
        source = config.mlflow_experiment
        return MLflowTraceLoader(
            MLflowTraceConfig(
                experiment_name=source.experiment,
                tracking_uri=source.tracking_uri,
                filter_string=source.filter,
                max_traces=config.max_traces or MLFLOW_DEFAULT_MAX_TRACES,
            )
        )
    assert config.mlflow_export is not None
    return MLflowFileTraceLoader(
        MLflowFileTraceConfig(
            path=config.mlflow_export.path,
            max_traces=config.max_traces or MLFLOW_DEFAULT_MAX_TRACES,
        )
    )


async def _run_evidence_streams(
    config: EvidenceStreamsConfig,
    snapshot: TraceSnapshot,
    llm: CompletionClient | None = None,
) -> list[EvidenceStreamResult]:
    """Run the configured evidence streams concurrently in registration order."""

    registry = registered_builtin_streams(
        anomaly_and_patterns=config.anomaly_and_patterns,
        tool_issues=config.tool_issues,
        ethos_divergence=config.ethos_divergence,
        llm=llm,
    )
    return list(
        await asyncio.gather(
            *(asyncio.to_thread(registry.analyze, name, snapshot) for name in registry.names)
        )
    )


def _build_llm(config: RunConfig, api_key: str) -> CompletionClient:
    """Construct the LLM client used by detection and insight compilation."""

    client_config: dict[str, Any] = {
        "api_base": config.api_base or resolve(ENV_API_BASE),
        "api_key": api_key,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "allowed_openai_params": ["tool_choice", "reasoning_effort"],
        "drop_params": True,
    }
    return CompletionClient(
        model=config.model or resolve(ENV_MODEL) or DEFAULT_MODEL,
        **client_config,
    )


def _build_insight_compilation(config: RunConfig, api_key: str) -> InsightCompilation:
    return InsightCompilation(llm=_build_llm(config, api_key))


async def _generate_insights(config: RunConfig) -> list[Insight]:
    load_dotenv()
    api_key = resolve(ENV_API_KEY)
    if not api_key:
        raise ValueError("Insight compilation needs an API key; set INSIGHT_AGENT_API_KEY")

    loader = _configured_trace_loader(config.trace)
    snapshot = await asyncio.to_thread(loader.load)
    llm = _build_llm(config, api_key) if config.evidence_streams.ethos_divergence else None
    evidence = await _run_evidence_streams(config.evidence_streams, snapshot, llm)
    existing_insights = load_insights(config.existing_insights) if config.existing_insights else []

    compilation = _build_insight_compilation(config, api_key)
    return await compilation.compile_insights(evidence, snapshot, existing_insights)


def _render_insights(insights: Sequence[Insight]) -> str:
    return yaml.safe_dump(
        [insight.model_dump() for insight in insights],
        allow_unicode=True,
        sort_keys=False,
    )


def _write_insights(output_path: Path, rendered: str) -> None:
    if output_path == Path("-"):
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


def get_config(argv: Sequence[str] | None = None) -> RunConfig:
    """Load run settings from optional YAML and generated CLI arguments."""

    return RunConfig(_cli_parse_args=list(sys.argv[1:] if argv is None else argv))


def main(argv: Sequence[str] | None = None) -> int:
    config = get_config(argv)
    rendered = _render_insights(asyncio.run(_generate_insights(config)))
    _write_insights(config.output_path, rendered)
    print(rendered, end="")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
