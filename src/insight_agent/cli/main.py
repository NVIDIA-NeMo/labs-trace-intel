"""Command-line implementations.

Exit codes:
    0  success
    1  usage or runtime error
    2  schema (structural) validation errors
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy
import sklearn
from pydantic_settings import CliApp, CliSettingsSource, get_subcommand

from insight_agent import __version__
from insight_agent.cli.artifacts import prepared_features, write_json
from insight_agent.cli.commands import (
    UTILITY_CLI_SHORTCUTS,
    UTILITY_COMMAND_NAMES,
    CoverageCommand,
    DemoCommand,
    ExplainFailuresCommand,
    RunAnalystCommand,
    SchemaCommand,
    TraceSourceCommandConfig,
    UtilityCommands,
    UtilityEvidenceConfig,
    ValidateCommand,
)
from insight_agent.config import (
    AnalystGenerationConfig,
    EvidenceStreamsConfig,
    FilesystemConfig,
    OutputConfig,
    RunConfig,
    TraceConfig,
    dump_run_config,
    load_run_config,
)
from insight_agent.evidence_streams.anomaly_and_patterns import (
    AnomalyAndPatternsAnalysis,
    AnomalyAndPatternsArtifacts,
    AnomalyAndPatternsConfig,
    decode_explicit_failure,
    problems_from_analysis,
    to_ia2_trace,
)
from insight_agent.evidence_streams.builtins import (
    ANOMALY_AND_PATTERNS,
    BUILTIN_STREAM_NAMES,
    TOOL_ISSUES,
    registered_builtin_streams,
)
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult
from insight_agent.evidence_streams.registry import EvidenceStreamRegistry
from insight_agent.evidence_streams.tool_issues import (
    ToolIssueCard,
    ToolIssueEvidenceArtifacts,
    detect,
    problems_from_cards,
    strict_failure,
    to_ia3_trace,
)
from insight_agent.evidence_streams.tool_issues.coverage import corpus_coverage, format_coverage
from insight_agent.insights_generation import DEFAULT_MODEL as ANALYST_DEFAULT_MODEL
from insight_agent.insights_generation import (
    InsightsGeneration,
    InsightsGenerationError,
    ResponseParseError,
)
from insight_agent.insights_generation.config import (
    ENV_API_BASE,
    ENV_API_KEY,
    ENV_MODEL,
    load_dotenv,
    resolve,
)
from insight_agent.trace_loaders import (
    MLFLOW_DEFAULT_MAX_TRACES,
    FSDataLoader,
    FSDataLoadError,
    MLflowFileTraceConfig,
    MLflowFileTraceLoader,
    MLflowTraceConfig,
    MLflowTraceLoader,
    MLflowTraceLoadError,
    TraceLoader,
)
from insight_agent.traces import Trace, TraceSnapshot

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_SCHEMA = 2

CLI_DESCRIPTION = (
    "Analyze agent traces for recurring problems using configurable evidence streams "
    "and optional LLM-authored Insights."
)
# -- shared plumbing -------------------------------------------------------


def _trace_loader(args: TraceSourceCommandConfig) -> TraceLoader:
    """Construct the selected trace source without leaking it downstream."""
    if args.mlflow_experiment is not None:
        source = args.mlflow_experiment
        return MLflowTraceLoader(
            MLflowTraceConfig(
                experiment_name=source.experiment,
                tracking_uri=source.tracking_uri,
                filter_string=source.filter,
                max_traces=args.max_traces,
            )
        )

    if args.mlflow_export is not None:
        return MLflowFileTraceLoader(
            MLflowFileTraceConfig(
                path=args.mlflow_export.path,
                max_traces=args.max_traces,
            )
        )

    assert args.traces is not None
    return FSDataLoader(args.traces)


def _configured_trace_loader(config: TraceConfig) -> TraceLoader:
    """Construct a trace loader directly from validated run configuration."""

    max_traces = config.max_traces or MLFLOW_DEFAULT_MAX_TRACES
    if config.filesystem is not None:
        return FSDataLoader(config.filesystem.path)
    if config.mlflow_experiment is not None:
        source = config.mlflow_experiment
        return MLflowTraceLoader(
            MLflowTraceConfig(
                experiment_name=source.experiment,
                tracking_uri=source.tracking_uri,
                filter_string=source.filter,
                max_traces=max_traces,
            )
        )
    assert config.mlflow_export is not None
    return MLflowFileTraceLoader(
        MLflowFileTraceConfig(path=config.mlflow_export.path, max_traces=max_traces)
    )


def _run_metadata(
    loader: TraceLoader,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Provenance recorded beside every result.

    Records the library versions because scikit-learn minor releases can change
    IsolationForest and KMeans output.
    """

    payload = {
        "insight_agent_version": __version__,
        "numpy_version": numpy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "corpus": loader.describe(),
    }
    if extra:
        payload["parameters"] = extra
    return payload


# -- commands --------------------------------------------------------------


def cmd_schema(args: SchemaCommand) -> int:
    text = json.dumps(Trace.model_json_schema(), indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return EXIT_OK


def cmd_validate(args: ValidateCommand) -> int:
    if args.config is not None:
        try:
            config = load_run_config(args.config)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(json.dumps(dump_run_config(config), indent=2, sort_keys=True))
        return EXIT_OK

    assert args.traces is not None
    try:
        snapshot = FSDataLoader(args.traces).load()
    except FSDataLoadError as error:
        payload = {"ok": False, "trace_count": 0, "errors": [str(error)]}
        if args.json_output:
            sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        else:
            print(f"error: {error}")
        return EXIT_SCHEMA

    payload = {"ok": True, "trace_count": snapshot.trace_count, "errors": []}
    if args.json_output:
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    else:
        print(f"OK — loaded {snapshot.trace_count} trace(s).")
    return EXIT_OK


def cmd_coverage(args: CoverageCommand) -> int:
    loader = FSDataLoader(args.traces)
    findings = None
    if args.with_findings:
        findings = detect(
            (to_ia3_trace(trace) for trace in loader.load()),
        )

    report = corpus_coverage(loader, findings=findings)
    if args.json_output:
        sys.stdout.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    else:
        print(format_coverage(report, verbose=args.verbose))
    return EXIT_OK


def cmd_explain_failures(args: ExplainFailuresCommand) -> int:
    """Show, per call, how each engine decoded the result.

    The anomaly-and-pattern and tool-issue streams use different failure
    decoders. The former unwraps text from ``content``/``output``/``message``/
    ``error``/``summary`` and accepts ``is_error``/``ok``/``exit_code``. The
    latter unwraps only ``content`` and accepts ``called``/a non-empty
    ``error``/shell exit lines. This command makes the divergence visible.
    """
    loader = _trace_loader(args)
    snapshot = loader.load()
    ia2_traces = {trace.id: to_ia2_trace(trace) for trace in snapshot}
    rows = []

    for trace in (to_ia3_trace(item) for item in snapshot):
        ia2_calls = {c.call_id: c for c in ia2_traces[trace.trace_id].calls}
        for call in trace.calls:
            ia3_failed, ia3_marker = strict_failure(call)
            ia2_call = ia2_calls.get(call.call_id)
            if ia2_call is None:
                continue
            ia2_failed, ia2_marker, _ = decode_explicit_failure(ia2_call.tool_name, ia2_call.result)
            row = {
                "trace_id": trace.trace_id,
                "call_id": call.call_id,
                "call_index": call.call_index,
                "tool_name": call.tool_name,
                "ia2_failed": ia2_failed,
                "ia2_marker": ia2_marker,
                "ia3_failed": ia3_failed,
                "ia3_marker": ia3_marker,
                "agree": ia2_failed == ia3_failed,
            }
            if args.only_disagreements and row["agree"]:
                continue
            rows.append(row)

    if args.json_output:
        sys.stdout.write(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        return EXIT_OK

    if not rows:
        print("No disagreements." if args.only_disagreements else "No calls.")
        return EXIT_OK

    print(f"{'trace':<28} {'call':<26} {'tool':<20} {'patterns':<24} {'tool issues':<24}")
    print("-" * 126)
    for row in rows:
        ia2 = f"{'FAIL' if row['ia2_failed'] else 'ok'} {row['ia2_marker'] or ''}".strip()
        ia3 = f"{'FAIL' if row['ia3_failed'] else 'ok'} {row['ia3_marker'] or ''}".strip()
        flag = "" if row["agree"] else "  <-- disagree"
        print(
            f"{row['trace_id'][:27]:<28} {row['call_id'][:25]:<26} "
            f"{row['tool_name'][:19]:<20} {ia2[:23]:<24} {ia3[:23]:<24}{flag}"
        )

    disagreements = sum(1 for r in rows if not r["agree"])
    print(f"\n{len(rows)} call(s), {disagreements} disagreement(s).")
    if disagreements:
        print(
            "Disagreements usually mean result text is wrapped under a key the tool-issue "
            'stream does not unwrap. Use {"content": ...} for textual results.'
        )
    return EXIT_OK


def _registered_evidence_streams(
    config: UtilityEvidenceConfig,
    *names: str,
) -> EvidenceStreamRegistry:
    requested = names or BUILTIN_STREAM_NAMES
    return registered_builtin_streams(
        anomaly_and_patterns=(
            config.anomaly_and_patterns if ANOMALY_AND_PATTERNS in requested else None
        ),
        tool_issues=config.tool_issues if TOOL_ISSUES in requested else None,
    )


def _configured_evidence_streams(
    config: EvidenceStreamsConfig,
    names: Sequence[str],
) -> EvidenceStreamRegistry:
    """Register selected streams with the typed settings loaded for each one."""

    return registered_builtin_streams(
        anomaly_and_patterns=(
            config.anomaly_and_patterns if ANOMALY_AND_PATTERNS in names else None
        ),
        tool_issues=config.tool_issues if TOOL_ISSUES in names else None,
    )


def _write_anomaly_and_patterns(
    output: OutputConfig,
    loader: TraceLoader,
    snapshot: TraceSnapshot,
    evidence: EvidenceStreamResult,
) -> int:
    artifacts = evidence.artifacts
    if not isinstance(artifacts, AnomalyAndPatternsArtifacts):
        raise TypeError("anomaly-and-patterns stream returned unexpected artifacts")
    result = artifacts.result

    if output.directory == Path("-"):
        sys.stdout.write(result.digest)
        return EXIT_OK

    target = output.directory / "ia2"
    target.mkdir(parents=True, exist_ok=True)
    (target / "digest.md").write_text(result.digest, encoding="utf-8")
    write_json(target / "anomalies.json", result.anomalies)
    write_json(target / "trajectory_groups.json", result.trajectory_groups)
    write_json(target / "verdict_groups.json", result.verdict_groups)
    write_json(target / "failure_groups.json", result.failure_groups)
    write_json(target / "cross_tool_failure_groups.json", result.cross_tool_failure_groups)
    write_json(target / "failure_events.json", result.failure_events)
    write_json(target / "features.json", prepared_features(result.prepared))
    write_json(
        target / "problems.json",
        [problem.model_dump(mode="json") for problem in evidence.problems],
    )
    write_json(
        target / "run.json",
        _run_metadata(
            loader,
            {
                "contamination": artifacts.config.contamination,
                "cluster_candidates": artifacts.config.cluster_candidates,
                "minimum_independent_traces": artifacts.config.minimum_independent_traces,
                "input_scaling": artifacts.config.input_scaling,
                "feature_names": (
                    list(artifacts.config.feature_names)
                    if artifacts.config.feature_names
                    else "default"
                ),
            },
        ),
    )

    if not output.quiet:
        flagged = [row for row in result.anomalies if row.is_anomaly]
        groups = result.trajectory_groups
        print(f"Anomaly and pattern evidence over {snapshot.trace_count} traces -> {target}")
        print(f"  unusual traces        : {len(flagged)}")
        print(f"  trajectory clusters   : {len(groups['clusters']) if groups else 'not evaluable'}")
        print(f"  verdict groups        : {len(result.verdict_groups)}")
        print(f"  failure groups        : {len(result.failure_groups)}")
        print(f"  cross-tool groups     : {len(result.cross_tool_failure_groups)}")
        print(f"  digest                : {target / 'digest.md'}")
        contamination = artifacts.config.contamination
        expected = snapshot.trace_count * contamination
        if expected < 1:
            print(
                f"  note: contamination={contamination} on {snapshot.trace_count} traces expects "
                f"{expected:.2f} flags. The default is tuned for corpora in the thousands; "
                "try --contamination 0.15 on a small sample."
            )
    return EXIT_OK


def _write_tool_issues(
    output: OutputConfig,
    loader: TraceLoader,
    snapshot: TraceSnapshot,
    evidence: EvidenceStreamResult,
) -> int:
    artifacts = evidence.artifacts
    if not isinstance(artifacts, ToolIssueEvidenceArtifacts):
        raise TypeError("tool-issue evidence stream returned unexpected artifacts")

    findings = list(artifacts.findings)
    cards = list(artifacts.cards)
    eligible = [card for card in cards if card.eligible_for_analyst]
    include_audit = artifacts.config.include_audit_problems
    rendered = cards if (include_audit or not eligible) else eligible
    target = output.directory / "ia3"
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "findings.json", findings)
    write_json(target / "cards.json", cards)
    write_json(target / "coverage.json", artifacts.catalog_coverage)
    write_json(
        target / "problems.json",
        [problem.model_dump(mode="json") for problem in evidence.problems],
    )
    (target / "cards.md").write_text(_render_cards(rendered, total=len(cards)), encoding="utf-8")
    write_json(
        target / "run.json",
        _run_metadata(
            loader,
            {
                "retry_threshold": artifacts.config.retry_threshold,
                "minimum_independent_cases": artifacts.config.minimum_independent_cases,
            },
        ),
    )

    if not output.quiet:
        print(f"Tool-issue evidence over {snapshot.trace_count} traces -> {target}")
        print(f"  findings              : {len(findings)}")
        print(f"  distinct issue types  : {len({f['issue_type'] for f in findings})}/19")
        print(f"  cards                 : {len(cards)}")
        print(f"  eligible for analyst  : {len(eligible)}")
        print(f"  cards                 : {target / 'cards.md'}")
        if cards and not eligible:
            distinct = loader.describe()["distinct_logical_cases"]
            print(
                f"  note: no card reached {artifacts.config.minimum_independent_cases} "
                "independent logical "
                "cases, i.e. no single issue type recurred across that many distinct cases."
            )
            print(
                f"        The loader reports {distinct} distinct logical cases. Verify the "
                "source's case identity before changing the threshold."
            )
    return EXIT_OK


def _render_cards(cards: Sequence[ToolIssueCard], *, total: int | None = None) -> str:
    lines = ["# Tool-issue evidence cards", ""]
    eligible = [card for card in cards if card.eligible_for_analyst]
    total = len(cards) if total is None else total
    lines.append(
        f"{total} card(s) in total; {len(eligible)} reached the independent-case threshold "
        "and are eligible for the Analyst."
    )
    if total > len(cards):
        lines += [
            "",
            f"Showing the {len(cards)} eligible card(s). The remaining {total - len(cards)} "
            "are in `cards.json`; enable "
            "`evidence_streams.tool_issues.include_audit_problems` to render them here too.",
        ]
    lines += [
        "",
        "A card is not an Insight. It is recurring, attributable evidence that a human or "
        "Analyst LLM still has to interpret. `impact_status` is never established here.",
        "",
    ]
    for card in sorted(cards, key=lambda item: (not item.eligible_for_analyst, item.card_id)):
        mark = "ELIGIBLE" if card.eligible_for_analyst else "audit only"
        lines += [
            f"## {card.card_id}  ({mark})",
            "",
            f"- family: {card.issue_family}",
            f"- mechanism: {card.mechanism_key}",
            f"- findings: {card.finding_count}",
            f"- independent logical cases: {card.independent_case_count}",
            f"- impact: {card.impact_status}",
            "",
        ]
        for item in card.representative_evidence[:3]:
            pointer = json.dumps(item.source_pointer, sort_keys=True)
            lines.append(
                f"  - `{item.trace_id}` call `{item.call_id}` "
                f"({item.tool_name}): {item.observation.strip()}"
            )
            lines.append(f"    source: `{pointer}`")
        lines.append("")
    return "\n".join(lines)


def _run(
    config: RunConfig,
    loader: TraceLoader,
    stream_names: Sequence[str] = BUILTIN_STREAM_NAMES,
) -> int:
    output = config.output
    analyst = config.analyst
    if analyst.enabled:
        missing = _analyst_preflight(analyst)
        if missing:
            print(f"error: {missing}", file=sys.stderr)
            print(
                "       set analyst.enabled to false to run evidence streams only.",
                file=sys.stderr,
            )
            return EXIT_ERROR

    snapshot = loader.load()
    output.directory.mkdir(parents=True, exist_ok=True)
    evidence = _configured_evidence_streams(config.evidence_streams, stream_names).analyze_all(
        snapshot
    )
    writers = {
        ANOMALY_AND_PATTERNS: _write_anomaly_and_patterns,
        TOOL_ISSUES: _write_tool_issues,
    }
    for result in evidence:
        if writers[result.stream_name](output, loader, snapshot, result) != EXIT_OK:
            return EXIT_ERROR
        if not output.quiet:
            print()

    analyst_ran = False
    if analyst.enabled:
        code = _run_insights(analyst, output, loader, snapshot, evidence)
        if code != EXIT_OK:
            return code
        analyst_ran = True
        if not output.quiet:
            print()

    (output.directory / "index.md").write_text(
        _render_index(
            str(loader.describe()["source"]),
            stream_names=stream_names,
            has_analyst=analyst_ran,
        ),
        encoding="utf-8",
    )
    if not output.quiet:
        print(f"index: {output.directory / 'index.md'}")
    return EXIT_OK


def cmd_run(config: RunConfig) -> int:
    stream_names = tuple(
        name
        for name, stream_config in (
            (ANOMALY_AND_PATTERNS, config.evidence_streams.anomaly_and_patterns),
            (TOOL_ISSUES, config.evidence_streams.tool_issues),
        )
        if stream_config is not None
    )
    return _run(config, _configured_trace_loader(config.trace), stream_names)


def _analyst_preflight(config: AnalystGenerationConfig) -> str | None:
    load_dotenv(config.env_file)
    if not (resolve(ENV_API_KEY) or "").strip():
        return "the Analyst needs an API key; set INSIGHT_AGENT_API_KEY"
    if "litellm" in sys.modules:
        litellm_available = sys.modules["litellm"] is not None
    else:
        litellm_available = importlib.util.find_spec("litellm") is not None
    if not litellm_available:
        return "the Analyst needs litellm, which is not installed"
    return None


def _render_index(
    source: str,
    *,
    stream_names: Sequence[str],
    has_analyst: bool,
) -> str:
    lines = [
        "# Insight Agent run",
        "",
        f"Trace source: `{source}`",
        "",
    ]
    if ANOMALY_AND_PATTERNS in stream_names:
        lines += [
            "## Anomalies and patterns — what is unusual, and what recurs",
            "",
            "- [digest.md](ia2/digest.md) — the packet written for the Analyst",
            "- [anomalies.json](ia2/anomalies.json), [features.json](ia2/features.json)",
            "- [trajectory_groups.json](ia2/trajectory_groups.json), "
            "[verdict_groups.json](ia2/verdict_groups.json)",
            "- [failure_groups.json](ia2/failure_groups.json), "
            "[cross_tool_failure_groups.json](ia2/cross_tool_failure_groups.json)",
            "",
        ]
    if TOOL_ISSUES in stream_names:
        lines += [
            "## Tool issues — what is demonstrably wrong with tool use",
            "",
            "- [cards.md](ia3/cards.md) — recurrence-qualified evidence cards",
            "- [cards.json](ia3/cards.json), [findings.json](ia3/findings.json)",
            "- [coverage.json](ia3/coverage.json) — all nineteen finding types",
            "",
        ]
    if has_analyst:
        lines += [
            "## Analyst — authored Insights",
            "",
            "- [insights.json](analyst/insights.json) — the authored Insights",
            "- [prompt.md](analyst/prompt.md) — the exact prompt sent",
            "- [run.json](analyst/run.json) — model, usage, prompt version",
            "",
            "This is the only LLM-authored artifact here, and the only one that is not",
            "reproducible. Everything above it is deterministic.",
            "",
        ]
    lines += [
        "## Reading these outputs",
        "",
        "An anomaly is not an error. A card is not an Insight. Abstention is a designed",
        "outcome, not a failure: when evidence is missing the detectors decline to guess.",
        "",
    ]
    return "\n".join(lines)


def _insights_inputs(args: RunAnalystCommand):
    """Load a snapshot and the evidence consumed by InsightsGeneration.

    Either reads artifacts a previous run already produced, or computes them
    in-process. Explicit `--digest`/`--cards` wins so a paid Analyst call can be
    re-issued against a frozen evidence set without re-running the engines.
    """
    loader = _trace_loader(args)
    snapshot = loader.load()

    if args.digest:
        digest = Path(args.digest).read_text(encoding="utf-8")
        cards = tuple(
            ToolIssueCard.model_validate(card)
            for card in json.loads(Path(args.cards).read_text(encoding="utf-8"))
        )
        anomalies = _read_sibling(args.digest, "anomalies.json") or []
        failure_groups = _read_sibling(args.digest, "failure_groups.json") or []
        cross_tool_groups = _read_sibling(args.digest, "cross_tool_failure_groups.json") or []
        finding_rows = _read_sibling(args.cards, "findings.json") or []
        anomaly_result = AnomalyAndPatternsAnalysis(
            prepared=(),
            failure_events=(),
            anomalies=tuple(anomalies),
            trajectory_groups=None,
            verdict_groups=(),
            failure_groups=tuple(failure_groups),
            cross_tool_failure_groups=tuple(cross_tool_groups),
            digest=digest,
        )
        anomaly_problems = problems_from_analysis(anomaly_result)
        tool_problems = problems_from_cards(
            cards,
            include_audit=args.evidence_streams.tool_issues.include_audit_problems,
        )
        evidence = (
            EvidenceStreamResult(
                stream_name="anomaly-and-patterns",
                problems=anomaly_problems,
                artifacts=AnomalyAndPatternsArtifacts(
                    result=anomaly_result,
                    config=AnomalyAndPatternsConfig(),
                ),
            ),
            EvidenceStreamResult(
                stream_name="tool-issues",
                problems=tool_problems,
                artifacts=ToolIssueEvidenceArtifacts(
                    findings=tuple(finding_rows),
                    cards=cards,
                    catalog_coverage={},
                    config=args.evidence_streams.tool_issues,
                ),
            ),
        )
        return loader, snapshot, evidence

    return (
        loader,
        snapshot,
        _registered_evidence_streams(args.evidence_streams).analyze_all(snapshot),
    )


def _read_sibling(reference: Path, name: str) -> Any:
    candidate = Path(reference).parent / name
    if not candidate.is_file():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _run_insights(
    analyst: AnalystGenerationConfig,
    output: OutputConfig,
    loader: TraceLoader,
    snapshot: TraceSnapshot,
    evidence: Sequence[EvidenceStreamResult],
    *,
    dry_run: bool = False,
) -> int:
    # Credentials are only needed here, so `.env` is only read here. A purely
    # deterministic run never touches the filesystem for configuration.
    loaded = load_dotenv(analyst.env_file)
    model = analyst.model or resolve(ENV_MODEL) or ANALYST_DEFAULT_MODEL
    api_base = analyst.api_base or resolve(ENV_API_BASE)
    api_key = resolve(ENV_API_KEY)

    try:
        generation = InsightsGeneration(
            snapshot=snapshot,
            evidence=evidence,
            corpus=loader.describe(),
            prompt_version=analyst.prompt_version,
        )
    except (InsightsGenerationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    target = output.directory / "analyst"
    target.mkdir(parents=True, exist_ok=True)

    system, user = generation.build_prompt()
    (target / "prompt.md").write_text(
        f"<!-- system -->\n\n{system}\n\n<!-- user -->\n\n{user}\n", encoding="utf-8"
    )

    if dry_run:
        approx = (len(system) + len(user)) // 4
        print("dry run — no API call made")
        print(f"  model                 : {model}")
        print(f"  prompt version        : {analyst.prompt_version}")
        if api_base:
            print(f"  api base              : {api_base}")
        print(f"  credentials           : {'found' if api_key else 'NOT FOUND'}")
        if loaded:
            print(f"  loaded from .env      : {', '.join(sorted(loaded))}")
        print(f"  problems              : {generation.problems_presented}")
        print(f"  prompt                : {target / 'prompt.md'}")
        print(f"  approx input tokens   : {approx:,}")
        return EXIT_OK

    extra: dict[str, Any] = {}
    if analyst.temperature is not None:
        # Unset by default on purpose: temperature is rejected outright by some
        # current models. Only sent when explicitly asked for.
        extra["temperature"] = analyst.temperature

    try:
        result = generation.generate(
            model=model,
            max_tokens=analyst.max_tokens,
            api_base=api_base,
            api_key=api_key,
            max_tool_rounds=analyst.max_tool_rounds,
            **extra,
        )
    except ResponseParseError as exc:
        (target / "analyst_raw.txt").write_text(exc.raw, encoding="utf-8")
        print(f"error: {exc}", file=sys.stderr)
        print(f"       raw response saved to {target / 'analyst_raw.txt'}", file=sys.stderr)
        return EXIT_ERROR
    except InsightsGenerationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    write_json(target / "insights.json", result.insights)
    write_json(
        target / "run.json",
        {
            "model": result.model,
            "usage": result.usage,
            "prompt_version": result.prompt_version,
            "api_base": api_base,
            "tool_calls": result.tool_calls,
            "traces_fetched": list(result.traces_fetched),
            "problems_presented": generation.problems_presented,
            "insight_count": len(result.insights),
            "corpus": loader.describe(),
        },
    )

    if not output.quiet:
        print(f"Analyst over {generation.problems_presented} problem(s) -> {target}")
        print(f"  model                 : {result.model}")
        if result.usage:
            tokens = result.usage.get("total_tokens") or result.usage.get("prompt_tokens")
            if tokens:
                print(f"  tokens                : {tokens:,}")
        if result.tool_calls:
            print(
                f"  trace lookups         : {result.tool_calls} call(s), "
                f"{len(result.traces_fetched)} distinct trace(s)"
            )
        print(f"  insights              : {len(result.insights)}")
        print(f"  insights              : {target / 'insights.json'}")

        if not result.insights:
            print(
                "  note: the Analyst filed zero Insights. That is a valid outcome — it means "
                "the evidence did not support a specific, recurring, actionable problem."
            )

        # Reported, not enforced: there is no validator, but an Insight citing a
        # trace that is not in the evidence is the failure mode most worth seeing.
        # A trace the model opened for itself is legitimate evidence, so the
        # known set has to include what it fetched — otherwise every prompt
        # with trace lookup looks like it is inventing ids.
        known = generation.known_trace_ids | set(result.traces_fetched)
        unknown = sorted(result.cited_trace_ids - known) if known else []
        if unknown:
            # Distinguish "read it but did not show us" from "does not exist".
            available = {trace.id for trace in snapshot}
            fabricated = sorted(t for t in unknown if t not in available)
            print(
                f"  note: {len(unknown)} cited trace id(s) are not in the evidence or the "
                f"traces fetched: {', '.join(unknown[:3])}{'...' if len(unknown) > 3 else ''}"
            )
            if fabricated:
                print(
                    f"  WARNING: {len(fabricated)} of those do not exist in the corpus at all — "
                    "treat this run's citations as unreliable."
                )
    return EXIT_OK


def cmd_run_analyst(args: RunAnalystCommand) -> int:
    """Author Insights from a snapshot and its evidence streams.

    The only command in the package that calls out to a model, and the only
    non-deterministic one. Config-driven executions and `demo` can invoke it;
    disable the Analyst for a purely deterministic run.
    """

    try:
        loader, snapshot, evidence = _insights_inputs(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _run_insights(
        args.analyst,
        args.output,
        loader,
        snapshot,
        evidence,
        dry_run=args.dry_run,
    )


def _bundled_data_dir() -> Path:
    # Resolve from the parent package: `data/` has no __init__.py, so
    # files("insight_agent.data") would return a MultiplexedPath that does not
    # stringify into a usable filesystem path.
    return Path(str(files("insight_agent"))) / "data"


def cmd_demo(args: DemoCommand) -> int:
    """End-to-end on the bundled data: the 'does this work at all' command."""
    source = _bundled_data_dir()
    corpus = source / "sample_corpus.jsonl"

    print(f"Using the bundled sample corpus: {corpus}\n")

    loaded = FSDataLoader(corpus)
    print(f"validate: {loaded.load().trace_count} trace(s), 0 error(s)")

    cov = corpus_coverage(loaded)
    print(
        f"coverage: {cov['rules']['evaluable']}/{cov['rules']['total']} "
        "tool-issue rules evaluable\n"
    )

    config = RunConfig(
        trace=TraceConfig(filesystem=FilesystemConfig(path=corpus)),
        output=args.output,
        evidence_streams=EvidenceStreamsConfig(
            anomaly_and_patterns=args.evidence_streams.anomaly_and_patterns,
            tool_issues=args.evidence_streams.tool_issues,
        ),
        analyst=args.analyst,
    )
    return cmd_run(config)


# -- parser ----------------------------------------------------------------


def _utility_cli_source() -> CliSettingsSource[UtilityCommands]:
    """Build every utility subcommand and option from its Pydantic model."""

    return CliSettingsSource(
        UtilityCommands,
        cli_prog_name="insight-agent",
        cli_shortcuts=UTILITY_CLI_SHORTCUTS,
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the generated utility parser for tests and embedding."""

    return _utility_cli_source().root_parser


def _run_utility(argv: Sequence[str]) -> int:
    source = _utility_cli_source()
    commands = CliApp.run(
        UtilityCommands,
        cli_args=list(argv),
        cli_settings_source=source,
    )
    command = get_subcommand(commands)
    handlers = {
        SchemaCommand: cmd_schema,
        ValidateCommand: cmd_validate,
        CoverageCommand: cmd_coverage,
        ExplainFailuresCommand: cmd_explain_failures,
        RunAnalystCommand: cmd_run_analyst,
        DemoCommand: cmd_demo,
    }
    return handlers[type(command)](command)


def _run_config_from_cli(argv: Sequence[str]) -> RunConfig:
    """Layer generated Pydantic CLI values over one validated YAML config."""

    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    known, _ = bootstrap.parse_known_args(argv)

    parser = argparse.ArgumentParser(
        prog="insight-agent",
        description=CLI_DESCRIPTION,
        epilog=(
            "Utility commands: schema, validate, coverage, explain-failures, run-analyst, demo"
        ),
    )
    parser.add_argument("--version", action="version", version=f"insight-agent {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML run configuration; generated CLI flags override its values",
    )
    cli_source = CliSettingsSource(
        RunConfig,
        root_parser=parser,
    )

    initial: dict[str, Any] = {}
    if known.config is not None:
        initial = load_run_config(known.config).model_dump()
    return CliApp.run(
        RunConfig,
        cli_args=list(argv),
        cli_settings_source=cli_source,
        **initial,
    )


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if raw_argv and raw_argv[0] in UTILITY_COMMAND_NAMES:
            return _run_utility(raw_argv)
        config = _run_config_from_cli(raw_argv)
        return cmd_run(config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except ModuleNotFoundError as exc:
        if exc.name == "mlflow" or (exc.name or "").startswith("mlflow."):
            print("error: MLflow support is not installed.", file=sys.stderr)
            print("       install it with: uv sync --extra mlflow", file=sys.stderr)
            return EXIT_ERROR
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        if isinstance(exc, FSDataLoadError):
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_SCHEMA
        if isinstance(exc, MLflowTraceLoadError):
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
