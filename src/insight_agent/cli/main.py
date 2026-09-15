# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze agent traces and write insights as YAML.

Complete setup example (run from the repository root; supply your OpenAI API key):

  export INSIGHT_AGENT_API_KEY='your-openai-api-key'
  cat > config.yaml <<'YAML'
trace:
  filesystem:
    path: examples/tau_bench_traces.jsonl
evidence_streams:
  tool_issues: {}
model: openai/gpt-5.2
max_tokens: 16384
YAML
  uv run insight-agent --config config.yaml

Nested CLI options override YAML, e.g. --trace.filesystem.path traces.jsonl.
See docs/configuration.md for all trace sources and evidence-stream settings.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from argparse import SUPPRESS, Action, ArgumentParser
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml
from nooa.unifiedllm import CompletionClient, UnifiedLLM
from pydantic_settings import CliSettingsSource

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
from insight_agent.insights_generation.validation import ProblemValidation
from insight_agent.trace_loaders.atif import ATIFTraceLoader
from insight_agent.trace_loaders.fs import FSDataLoader
from insight_agent.trace_loaders.intake import IntakeTraceLoader
from insight_agent.trace_loaders.langfuse import (
    LANGFUSE_DEFAULT_MAX_TRACES,
    LangfuseFileTraceConfig,
    LangfuseFileTraceLoader,
    LangfuseTraceConfig,
    LangfuseTraceLoader,
)
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
EXIT_SETUP = 2
DEFAULT_REASONING_EFFORT = "high"
_LOGGER = logging.getLogger("insight_agent")


class SetupError(ValueError):
    """Required settings are missing from the environment for this run."""


def _check_environment(config: RunConfig) -> str:
    """Check the selected integrations before constructing clients or loading traces."""

    load_dotenv()
    api_key = resolve(ENV_API_KEY)
    required = {f"{ENV_API_KEY} — API key for the configured model": api_key}
    if config.trace.langsmith is not None:
        # Match the LangSmith SDK's legacy alias without importing the optional SDK.
        required["LANGSMITH_API_KEY — API key for LangSmith"] = os.environ.get(
            "LANGSMITH_API_KEY", ""
        ).strip() or os.environ.get("LANGCHAIN_API_KEY")
    if config.trace.langfuse is not None:
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            required[f"{name} — Langfuse project credentials"] = os.environ.get(name)
        if not (config.trace.langfuse.base_url or "").strip():
            required["LANGFUSE_BASE_URL — Langfuse URL; or set trace.langfuse.base_url"] = (
                os.environ.get("LANGFUSE_BASE_URL")
            )
    sentiment = config.evidence_streams.user_sentiment
    if sentiment is not None and sentiment.litellm is not None and sentiment.litellm.api_key_env:
        name = sentiment.litellm.api_key_env
        hint = "API key named by evidence_streams.user_sentiment.litellm.api_key_env"
        required[f"{name} — {hint}"] = os.environ.get(name)

    missing = [f"  {label}" for label, value in required.items() if not (value or "").strip()]
    if missing:
        raise SetupError(
            "Missing required environment settings:\n"
            + "\n".join(missing)
            + "\n\nSet these in .env in your working directory (NAME=value),\n"
            "or export them in your shell, then rerun the command."
        )
    assert api_key is not None
    return api_key


def _configured_trace_loader(config: TraceConfig) -> TraceLoader:
    """Construct the trace loader selected by the run configuration."""

    if config.filesystem is not None:
        return FSDataLoader(config.filesystem.path)
    if config.atif is not None:
        return ATIFTraceLoader(config.atif)
    if config.intake is not None:
        source = config.intake
        if config.max_traces is not None:
            query = source.query.model_copy(update={"max_traces": config.max_traces})
            source = source.model_copy(update={"query": query})
        return IntakeTraceLoader(config=source)
    if config.langfuse is not None:
        source = config.langfuse
        return LangfuseTraceLoader(
            LangfuseTraceConfig(
                base_url=source.base_url,
                from_timestamp=source.from_timestamp,
                to_timestamp=source.to_timestamp,
                filter_string=source.filter,
                max_traces=config.max_traces or LANGFUSE_DEFAULT_MAX_TRACES,
            )
        )
    if config.langfuse_export is not None:
        return LangfuseFileTraceLoader(
            LangfuseFileTraceConfig(
                path=config.langfuse_export.path,
                max_traces=config.max_traces or LANGFUSE_DEFAULT_MAX_TRACES,
            )
        )
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
    llm_factory: Callable[[], UnifiedLLM] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[EvidenceStreamResult]:
    """Run the configured evidence streams concurrently in registration order."""

    registry = registered_builtin_streams(
        anomaly_and_patterns=config.anomaly_and_patterns,
        tool_issues=config.tool_issues,
        ethos_divergence=config.ethos_divergence,
        user_sentiment=config.user_sentiment,
        eval_failure_patterns=config.eval_failure_patterns,
        llm_factory=llm_factory,
    )

    async def analyze(name: str) -> tuple[str, EvidenceStreamResult]:
        result = await registry.analyze(name, snapshot)
        return name, result

    completed: dict[str, EvidenceStreamResult] = {}
    total = len(registry.names)
    tasks = [analyze(name) for name in registry.names]
    for future in asyncio.as_completed(tasks):
        name, result = await future
        completed[name] = result
        if progress is not None:
            progress(
                f'Evidence stream "{_display_name(name)}" found '
                f"{_count(len(result.problems), 'candidate problem')} "
                f"({len(completed)}/{total} complete)."
            )
            for problem in result.problems[:3]:
                progress(f"  Candidate: {_summary(problem.description)}")
            remaining = len(result.problems) - 3
            if remaining > 0:
                progress(f"  {_count(remaining, 'additional candidate')} not shown.")
    return [completed[name] for name in registry.names]


async def _validate_evidence_with_code(
    evidence: list[EvidenceStreamResult],
    snapshot: TraceSnapshot,
    code_base_path: Path,
    llm: UnifiedLLM,
) -> list[EvidenceStreamResult]:
    """Keep only Problems that the supplied codebase supports."""

    code_base_path = code_base_path.expanduser().resolve()
    if not code_base_path.is_dir():
        raise ValueError(f"code_base must be an existing directory: {code_base_path}")
    validator = ProblemValidation(code_base_path, llm)

    async def validate_result(result: EvidenceStreamResult) -> EvidenceStreamResult:
        validations = []
        for problem in result.problems:
            supporting_traces = tuple(
                snapshot.get_trace_by_id(trace_id) for trace_id in problem.supporting_trace_ids
            )
            validations.append(validator.is_supported(problem, supporting_traces))

        decisions = await asyncio.gather(*validations)
        retained_problems = tuple(
            problem
            for problem, decision in zip(result.problems, decisions, strict=True)
            if decision is not False
        )
        return result.model_copy(update={"problems": retained_problems})

    return list(await asyncio.gather(*(validate_result(result) for result in evidence)))


def _build_llm(config: RunConfig, api_key: str) -> CompletionClient:
    """Construct an LLM client with the run's shared model settings."""

    return CompletionClient(
        model=config.model or resolve(ENV_MODEL) or DEFAULT_MODEL,
        api_base=config.api_base or resolve(ENV_API_BASE),
        api_key=api_key,
        max_tokens=config.max_tokens or DEFAULT_MAX_TOKENS,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        allowed_openai_params=["tool_choice", "reasoning_effort"],
        drop_params=True,
    )


async def _generate_insights(config: RunConfig) -> list[Insight]:
    api_key = _check_environment(config)

    loader = _configured_trace_loader(config.trace)
    source_name = _trace_source_name(config.trace)
    _LOGGER.info("Loading traces from %s...", source_name)
    snapshot = await asyncio.to_thread(loader.load)
    corpus = loader.describe()
    _LOGGER.info(
        "Loaded %s with %s across %s from %s.",
        _count(corpus["trace_count"], "trace"),
        _count(corpus["call_count"], "tool call"),
        _count(corpus["distinct_logical_cases"], "logical case"),
        source_name,
    )

    stream_names = tuple(
        name
        for name, stream in (
            ("anomaly and patterns", config.evidence_streams.anomaly_and_patterns),
            ("tool issues", config.evidence_streams.tool_issues),
            ("ethos divergence", config.evidence_streams.ethos_divergence),
            ("user sentiment", config.evidence_streams.user_sentiment),
            ("evaluation failure patterns", config.evidence_streams.eval_failure_patterns),
        )
        if stream is not None
    )
    _LOGGER.info(
        "Running %s: %s.",
        _count(len(stream_names), "evidence stream"),
        ", ".join(stream_names),
    )
    evidence = await _run_evidence_streams(
        config.evidence_streams,
        snapshot,
        lambda: _build_llm(config, api_key),
        _LOGGER.info,
    )
    existing_insights = load_insights(config.existing_insights) if config.existing_insights else []

    async with _build_llm(config, api_key) as llm:
        compilation = InsightCompilation(llm=llm)
        if config.code_base is not None:
            candidate_count = sum(len(result.problems) for result in evidence)
            _LOGGER.info(
                "Validating %s against the codebase...",
                _count(candidate_count, "candidate problem"),
            )
            evidence = await _validate_evidence_with_code(
                evidence,
                snapshot,
                config.code_base,
                llm,
            )
            retained_count = sum(len(result.problems) for result in evidence)
            _LOGGER.info(
                "Code validation retained %s.",
                _count(retained_count, "candidate problem"),
            )

        candidate_count = sum(len(result.problems) for result in evidence)
        existing_context = (
            f" and {_count(len(existing_insights), 'existing insight')}"
            if existing_insights
            else ""
        )
        _LOGGER.info(
            "Synthesizing %s%s into final insights...",
            _count(candidate_count, "candidate problem"),
            existing_context,
        )
        insights = await compilation.compile_insights(evidence, snapshot, existing_insights)
    _LOGGER.info("Generated %s.", _count(len(insights), "final insight"))
    return insights


def _trace_source_name(config: TraceConfig) -> str:
    """Return a concise user-facing name for the selected trace source."""

    sources = (
        (config.filesystem, "canonical JSONL"),
        (config.atif, "ATIF JSONL"),
        (config.intake, "NeMo Platform Intake"),
        (config.langfuse, "Langfuse"),
        (config.langfuse_export, "a Langfuse export"),
        (config.langsmith, "LangSmith"),
        (config.langsmith_trace_export_file, "a LangSmith export"),
        (config.mlflow_experiment, "MLflow"),
        (config.mlflow_export, "an MLflow export"),
    )
    return next(name for source, name in sources if source is not None)


def _display_name(value: str) -> str:
    return value.replace("-", " ")


def _count(value: int, singular: str) -> str:
    suffix = "" if value == 1 else "s"
    return f"{value:,} {singular}{suffix}"


def _summary(value: str, limit: int = 220) -> str:
    text = " ".join(value.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


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


# Pydantic hooks accept parser/group objects and argparse's heterogeneous keyword arguments.
def _add_common_argument(parser: Any, *args: str, **kwargs: Any) -> Action:  # noqa: ANN401
    if (
        kwargs.get("dest") not in ("config", "output_path", "model")
        and kwargs.get("action") != "help"
    ):
        kwargs["help"] = SUPPRESS
    return parser.add_argument(*args, **kwargs)


def _add_common_group(parser: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    kwargs["description"] = None
    return parser.add_argument_group(**kwargs)


def _print_help(*, full: bool) -> None:
    parser = CliSettingsSource(
        RunConfig,
        add_argument_method=ArgumentParser.add_argument if full else _add_common_argument,
        add_argument_group_method=ArgumentParser.add_argument_group if full else _add_common_group,
    ).root_parser
    parser.description = __doc__
    parser.add_argument("--help-all", action="help", help="Show all configuration options")
    parser.print_help()


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or any(arg in argv for arg in ("-h", "--help", "--help-all")):
        _print_help(full="--help-all" in argv)
        return EXIT_OK

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[insight-agent] %(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False
    try:
        config = get_config(argv)
        insights = asyncio.run(_generate_insights(config))
        rendered = _render_insights(insights)
        _write_insights(config.output_path, rendered)
        if config.output_path == Path("-"):
            _LOGGER.info("Final insights follow on standard output.")
        else:
            _LOGGER.info(
                "Wrote %s to %s.",
                _count(len(insights), "final insight"),
                config.output_path,
            )
        print(rendered, end="")
        return EXIT_OK
    except SetupError as error:
        _LOGGER.error("%s", error)
        return EXIT_SETUP
    finally:
        _LOGGER.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
