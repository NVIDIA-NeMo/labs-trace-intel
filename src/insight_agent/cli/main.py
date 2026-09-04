"""Command-line implementations.

Exit codes:
    0  success
    1  usage or runtime error
    2  schema (structural) validation errors
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy
import sklearn
from nooa import build_prompt_data
from nooa.prompts import render_prompt_data
from nooa.unifiedllm import CompletionClient

from insight_agent import __version__
from insight_agent.cli.artifacts import prepared_features, write_json
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
    ToolIssueConfig,
    ToolIssueEvidenceArtifacts,
    detect,
    problems_from_cards,
    strict_failure,
    to_ia3_trace,
)
from insight_agent.evidence_streams.tool_issues.coverage import corpus_coverage, format_coverage
from insight_agent.insights_generation import InsightCompilation
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

DEFAULT_MODEL = "openai/azure/openai/gpt-5.6-luna"
DEFAULT_API_BASE = "https://inference-api.nvidia.com/v1"
DEFAULT_MAX_TOKENS = 300_000
DEFAULT_REASONING_EFFORT = "high"


# -- shared plumbing -------------------------------------------------------


def _add_file_corpus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("traces", type=Path, help="canonical JSONL corpus")


def _add_trace_source_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("traces", type=Path, nargs="?", help="canonical JSONL corpus")
    source.add_argument(
        "--mlflow-experiment",
        metavar="NAME",
        help="load traces directly from this MLflow experiment",
    )
    source.add_argument(
        "--mlflow-export",
        type=Path,
        metavar="PATH",
        help="load a native MLflow trace JSON export",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI or the MLflow default)",
    )
    parser.add_argument(
        "--mlflow-filter",
        default=None,
        help="MLflow trace search filter, for example `trace.status = 'ERROR'`",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=MLFLOW_DEFAULT_MAX_TRACES,
        help=(
            f"maximum MLflow traces to materialize in memory (default: "
            f"{MLFLOW_DEFAULT_MAX_TRACES}); online queries use 500-trace pages per MLflow's "
            "REST limit: "
            "https://mlflow.org/docs/latest/api_reference/rest-api.html#searchtracesv3"
        ),
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("out"), help="output directory (default: ./out)"
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the stdout summary")


def _cluster_candidates(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def _add_anomaly_and_patterns_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--input-scaling", choices=("none", "robust"), default="none")
    parser.add_argument(
        "--cluster-candidates",
        type=_cluster_candidates,
        default=(2, 3, 4, 5, 6, 7, 8),
    )
    parser.add_argument("--min-independent-traces", type=int, default=3)
    parser.add_argument(
        "--feature", action="append", default=None, help="override the feature list (repeatable)"
    )


def _add_tool_issue_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-independent-cases", type=int, default=3)
    parser.add_argument("--retry-threshold", type=int, default=3)
    parser.add_argument("--all-cards", action="store_true")


def _trace_loader(args: argparse.Namespace) -> TraceLoader:
    """Construct the selected trace source without leaking it downstream."""
    experiment = getattr(args, "mlflow_experiment", None)
    export_path = getattr(args, "mlflow_export", None)
    if experiment:
        return MLflowTraceLoader(
            MLflowTraceConfig(
                experiment_name=experiment,
                tracking_uri=getattr(args, "mlflow_tracking_uri", None),
                filter_string=getattr(args, "mlflow_filter", None),
                max_traces=getattr(args, "max_traces", MLFLOW_DEFAULT_MAX_TRACES),
            )
        )

    if export_path:
        mlflow_query_options = []
        if getattr(args, "mlflow_tracking_uri", None) is not None:
            mlflow_query_options.append("--mlflow-tracking-uri")
        if getattr(args, "mlflow_filter", None) is not None:
            mlflow_query_options.append("--mlflow-filter")
        if mlflow_query_options:
            options = ", ".join(mlflow_query_options)
            raise ValueError(f"{options} require --mlflow-experiment")
        return MLflowFileTraceLoader(
            MLflowFileTraceConfig(
                path=export_path,
                max_traces=getattr(args, "max_traces", MLFLOW_DEFAULT_MAX_TRACES),
            )
        )

    mlflow_only = []
    if getattr(args, "mlflow_tracking_uri", None) is not None:
        mlflow_only.append("--mlflow-tracking-uri")
    if getattr(args, "mlflow_filter", None) is not None:
        mlflow_only.append("--mlflow-filter")
    if getattr(args, "max_traces", MLFLOW_DEFAULT_MAX_TRACES) != MLFLOW_DEFAULT_MAX_TRACES:
        mlflow_only.append("--max-traces")
    if mlflow_only:
        options = ", ".join(mlflow_only)
        raise ValueError(f"{options} require --mlflow-experiment")

    return FSDataLoader(args.traces)


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


def cmd_schema(args) -> int:
    text = json.dumps(Trace.model_json_schema(), indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return EXIT_OK


def cmd_validate(args) -> int:
    try:
        snapshot = FSDataLoader(args.traces).load()
    except FSDataLoadError as error:
        payload = {"ok": False, "trace_count": 0, "errors": [str(error)]}
        if args.json:
            sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        else:
            print(f"error: {error}")
        return EXIT_SCHEMA

    payload = {"ok": True, "trace_count": snapshot.trace_count, "errors": []}
    if args.json:
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    else:
        print(f"OK — loaded {snapshot.trace_count} trace(s).")
    return EXIT_OK


def cmd_coverage(args) -> int:
    loader = _trace_loader(args)
    findings = None
    if args.with_findings:
        findings = detect(
            (to_ia3_trace(trace) for trace in loader.load()),
        )

    report = corpus_coverage(loader, findings=findings)
    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    else:
        print(format_coverage(report, verbose=args.verbose))
    return EXIT_OK


def cmd_explain_failures(args) -> int:
    """Show, per call, how each engine decoded the result.

    IA2 and IA3 use different failure decoders. IA2 unwraps text from
    ``content``/``output``/``message``/``error``/``summary`` while IA3 unwraps
    only ``content``; IA2 accepts ``is_error``/``ok``/``exit_code`` while IA3
    accepts ``called``/a non-empty ``error``/shell exit lines. This command
    makes the divergence visible rather than mysterious.
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

    if args.json:
        sys.stdout.write(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        return EXIT_OK

    if not rows:
        print("No disagreements." if args.only_disagreements else "No calls.")
        return EXIT_OK

    print(f"{'trace':<28} {'call':<26} {'tool':<20} {'IA2':<24} {'IA3':<24}")
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
            "Disagreements usually mean result text is wrapped under a key IA3 does not "
            'unwrap. Use {"content": ...} for textual results.'
        )
    return EXIT_OK


def _registered_evidence_streams(
    args: argparse.Namespace,
    *names: str,
) -> EvidenceStreamRegistry:
    requested = names or BUILTIN_STREAM_NAMES
    anomaly_and_patterns: AnomalyAndPatternsConfig | None = None
    if ANOMALY_AND_PATTERNS in requested:
        anomaly_and_patterns = AnomalyAndPatternsConfig(
            contamination=args.contamination,
            input_scaling=args.input_scaling,
            cluster_candidates=args.cluster_candidates,
            minimum_independent_traces=args.min_independent_traces,
            feature_names=tuple(args.feature) if args.feature is not None else None,
        )

    tool_issues: ToolIssueConfig | None = None
    if TOOL_ISSUES in requested:
        tool_issues = ToolIssueConfig(
            minimum_independent_cases=args.min_independent_cases,
            retry_threshold=args.retry_threshold,
            include_audit_problems=args.all_cards,
        )

    return registered_builtin_streams(
        anomaly_and_patterns=anomaly_and_patterns,
        tool_issues=tool_issues,
    )


def _write_anomaly_and_patterns(
    args: argparse.Namespace,
    loader: TraceLoader,
    snapshot: TraceSnapshot,
    evidence: EvidenceStreamResult,
) -> int:
    artifacts = evidence.artifacts
    if not isinstance(artifacts, AnomalyAndPatternsArtifacts):
        raise TypeError("anomaly-and-patterns stream returned unexpected artifacts")
    result = artifacts.result

    if args.out == Path("-"):
        sys.stdout.write(result.digest)
        return EXIT_OK

    target = args.out / "ia2"
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

    if not args.quiet:
        flagged = [row for row in result.anomalies if row.is_anomaly]
        groups = result.trajectory_groups
        print(f"IA2 over {snapshot.trace_count} traces -> {target}")
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


def cmd_run_ia2(args: argparse.Namespace) -> int:
    loader = _trace_loader(args)
    snapshot = loader.load()

    evidence = _registered_evidence_streams(args, ANOMALY_AND_PATTERNS).analyze(
        ANOMALY_AND_PATTERNS, snapshot
    )
    return _write_anomaly_and_patterns(args, loader, snapshot, evidence)


def _write_tool_issues(
    args: argparse.Namespace,
    loader: TraceLoader,
    snapshot: TraceSnapshot,
    evidence: EvidenceStreamResult,
) -> int:
    artifacts = evidence.artifacts
    if not isinstance(artifacts, ToolIssueEvidenceArtifacts):
        raise TypeError("tool-issue evidence stream returned unexpected artifacts")

    findings = list(artifacts.findings)
    cards = list(artifacts.cards)
    eligible = [card for card in cards if card.eligible_for_insight_compilation]
    include_audit = artifacts.config.include_audit_problems
    rendered = cards if (include_audit or not eligible) else eligible
    target = args.out / "ia3"
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

    if not args.quiet:
        print(f"IA3 over {snapshot.trace_count} traces -> {target}")
        print(f"  findings              : {len(findings)}")
        print(f"  distinct issue types  : {len({f['issue_type'] for f in findings})}/19")
        print(f"  cards                 : {len(cards)}")
        print(f"  eligible for compilation: {len(eligible)}")
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


def cmd_run_ia3(args: argparse.Namespace) -> int:
    loader = _trace_loader(args)
    snapshot = loader.load()

    evidence = _registered_evidence_streams(args, TOOL_ISSUES).analyze(TOOL_ISSUES, snapshot)
    return _write_tool_issues(args, loader, snapshot, evidence)


def _render_cards(cards: Sequence[ToolIssueCard], *, total: int | None = None) -> str:
    lines = ["# IA3 evidence cards", ""]
    eligible = [card for card in cards if card.eligible_for_insight_compilation]
    total = len(cards) if total is None else total
    lines.append(
        f"{total} card(s) in total; {len(eligible)} reached the independent-case threshold "
        "and are eligible for Insight compilation."
    )
    if total > len(cards):
        lines += [
            "",
            f"Showing the {len(cards)} eligible card(s). The remaining {total - len(cards)} "
            "are in `cards.json`; re-run with `--all-cards` to render them here too.",
        ]
    lines += [
        "",
        "A card is not an Insight. It is recurring, attributable evidence that a human or "
        "the Insight compilation agent still has to interpret. `impact_status` is never "
        "established here.",
        "",
    ]
    for card in sorted(
        cards, key=lambda item: (not item.eligible_for_insight_compilation, item.card_id)
    ):
        mark = "ELIGIBLE" if card.eligible_for_insight_compilation else "audit only"
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


def _run_all(args: argparse.Namespace, loader: TraceLoader) -> int:
    if not args.no_insights:
        missing = _insight_preflight(args)
        if missing:
            print(f"error: {missing}", file=sys.stderr)
            print("       pass --no-insights to run IA2 and IA3 only.", file=sys.stderr)
            return EXIT_ERROR

    snapshot = loader.load()
    args.out.mkdir(parents=True, exist_ok=True)
    registry = _registered_evidence_streams(args)
    anomaly_and_patterns, tool_issues = registry.analyze_all(snapshot)
    if _write_anomaly_and_patterns(args, loader, snapshot, anomaly_and_patterns) != EXIT_OK:
        return EXIT_ERROR
    if not args.quiet:
        print()

    if _write_tool_issues(args, loader, snapshot, tool_issues) != EXIT_OK:
        return EXIT_ERROR
    if not args.quiet:
        print()

    insights_ran = False
    if not args.no_insights:
        code = _run_insights(args, loader, snapshot, (anomaly_and_patterns, tool_issues))
        if code != EXIT_OK:
            return code
        insights_ran = True
        if not args.quiet:
            print()

    (args.out / "index.md").write_text(
        _render_index(str(loader.describe()["source"]), has_insights=insights_ran),
        encoding="utf-8",
    )
    if not args.quiet:
        print(f"index: {args.out / 'index.md'}")
    return EXIT_OK


def cmd_run_all(args: argparse.Namespace) -> int:
    return _run_all(args, _trace_loader(args))


def _insight_preflight(args: argparse.Namespace) -> str | None:
    _load_inference_environment(getattr(args, "env_file", None))
    if not _inference_api_key():
        return "Insight compilation needs INFERENCE_API_KEY"
    return None


def _render_index(source: str, *, has_insights: bool) -> str:
    lines = [
        "# Insight Agent run",
        "",
        f"Trace source: `{source}`",
        "",
        "## IA2 — what is unusual, and what recurs",
        "",
        "- [digest.md](ia2/digest.md) — the packet written for Insight compilation",
        "- [anomalies.json](ia2/anomalies.json), [features.json](ia2/features.json)",
        "- [trajectory_groups.json](ia2/trajectory_groups.json), "
        "[verdict_groups.json](ia2/verdict_groups.json)",
        "- [failure_groups.json](ia2/failure_groups.json), "
        "[cross_tool_failure_groups.json](ia2/cross_tool_failure_groups.json)",
        "",
        "## IA3 — what is demonstrably wrong with tool use",
        "",
        "- [cards.md](ia3/cards.md) — recurrence-qualified evidence cards",
        "- [cards.json](ia3/cards.json), [findings.json](ia3/findings.json)",
        "- [coverage.json](ia3/coverage.json) — all nineteen finding types",
        "",
    ]
    if has_insights:
        lines += [
            "## Insight compilation — authored Insights",
            "",
            "- [insights.json](insights/insights.json) — the authored Insights",
            "- [prompt.txt](insights/prompt.txt) — the exact prompt sent",
            "- [run.json](insights/run.json) — model and usage",
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


def _insights_inputs(args):
    """Load a snapshot and the evidence consumed by InsightCompilation.

    Either reads artifacts a previous run already produced, or computes them
    in-process. Explicit `--digest`/`--cards` wins so a paid compilation call can be
    re-issued against a frozen evidence set without re-running the engines.
    """
    if bool(args.digest) != bool(args.cards):
        raise ValueError("--digest and --cards must be given together")

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
        tool_problems = problems_from_cards(cards, include_audit=args.all_cards)
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
                    config=ToolIssueConfig(include_audit_problems=args.all_cards),
                ),
            ),
        )
        return loader, snapshot, evidence

    return (
        loader,
        snapshot,
        _registered_evidence_streams(args).analyze_all(snapshot),
    )


def _read_sibling(reference: Path, name: str) -> Any:
    candidate = Path(reference).parent / name
    if not candidate.is_file():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _load_inference_environment(path: Path | None) -> set[str]:
    from dotenv import dotenv_values, load_dotenv

    values = dotenv_values(path) if path else dotenv_values()
    load_dotenv(path, override=False)
    return {key for key, value in values.items() if value}


def _inference_api_key() -> str | None:
    return next(
        (
            os.environ.get(name)
            for name in (
                "INFERENCE_API_KEY",
                "NVIDIA_INFERENCE_API_KEY",
                "INSIGHT_AGENT_API_KEY",
            )
            if os.environ.get(name)
        ),
        None,
    )


def _run_insights(
    args: argparse.Namespace,
    loader: TraceLoader,
    snapshot: TraceSnapshot,
    evidence: Sequence[EvidenceStreamResult],
) -> int:
    loaded = _load_inference_environment(getattr(args, "env_file", None))
    model = (
        args.model
        or os.environ.get("INFERENCE_MODEL")
        or os.environ.get("INSIGHT_AGENT_MODEL")
        or DEFAULT_MODEL
    )
    api_base = (
        getattr(args, "api_base", None)
        or os.environ.get("INFERENCE_API_BASE")
        or os.environ.get("INSIGHT_AGENT_API_BASE")
        or DEFAULT_API_BASE
    )
    api_key = _inference_api_key()
    client = CompletionClient(
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=args.max_tokens,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        allowed_openai_params=["tool_choice", "reasoning_effort"],
        drop_params=True,
    )
    compilation = InsightCompilation(llm=client)
    presented = list(evidence)
    problems_presented = sum(len(result.problems) for result in presented)
    target = args.out / "insights"
    target.mkdir(parents=True, exist_ok=True)

    try:
        prompt_data = asyncio.run(
            build_prompt_data(compilation.compile_insights, presented, snapshot, [])
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: could not build Insight compilation prompt: {exc}", file=sys.stderr)
        return EXIT_ERROR

    prompt = render_prompt_data(prompt_data)
    (target / "prompt.txt").write_text(prompt, encoding="utf-8")
    if args.dry_run:
        print("dry run — no API call made")
        print(f"  model                 : {model}")
        print(f"  api base              : {api_base}")
        print(f"  credentials           : {'found' if api_key else 'NOT FOUND'}")
        if loaded:
            print(f"  loaded from .env      : {', '.join(sorted(loaded))}")
        print(f"  problems              : {problems_presented}")
        print(f"  prompt                : {target / 'prompt.txt'}")
        print(f"  approx input tokens   : {len(prompt) // 4:,}")
        return EXIT_OK

    if not api_key:
        print("error: Insight compilation needs INFERENCE_API_KEY", file=sys.stderr)
        return EXIT_ERROR

    try:
        insights = asyncio.run(compilation.compile_insights(presented, snapshot, []))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: Insight compilation failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    rows = [insight.model_dump(mode="json") for insight in insights]
    write_json(target / "insights.json", rows)
    write_json(
        target / "run.json",
        {
            "model": model,
            "api_base": api_base,
            "problems_presented": problems_presented,
            "insight_count": len(rows),
            "corpus": loader.describe(),
        },
    )

    if not args.quiet:
        print(f"Insight compilation over {problems_presented} problem(s) -> {target}")
        print(f"  model                 : {model}")
        print(f"  insights              : {len(rows)}")
        print(f"  insights              : {target / 'insights.json'}")
        if not rows:
            print(
                "  note: zero Insights is valid when the evidence does not support a "
                "specific, recurring, actionable problem."
            )

        known = {
            trace_id
            for result in presented
            for problem in result.problems
            for trace_id in problem.supporting_trace_ids
        }
        cited = {trace_id for insight in insights for trace_id in insight.trace_refs}
        unknown = sorted(cited - known)
        if unknown:
            available = {trace.id for trace in snapshot}
            fabricated = sorted(trace_id for trace_id in unknown if trace_id not in available)
            print(
                f"  note: {len(unknown)} cited trace id(s) are not in the evidence: "
                f"{', '.join(unknown[:3])}{'...' if len(unknown) > 3 else ''}"
            )
            if fabricated:
                print(f"  WARNING: {len(fabricated)} cited trace id(s) do not exist in the corpus")
    return EXIT_OK


def cmd_run_insights(args: argparse.Namespace) -> int:
    """Author Insights from a snapshot and its evidence streams.

    The only command in the package that calls out to a model, and the only
    non-deterministic one. `run-all` and `demo` invoke it by default; pass
    `--no-insights` to either for a purely deterministic run.
    """

    try:
        loader, snapshot, evidence = _insights_inputs(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _run_insights(args, loader, snapshot, evidence)


def _bundled_data_dir() -> Path:
    # Resolve from the parent package: `data/` has no __init__.py, so
    # files("insight_agent.data") would return a MultiplexedPath that does not
    # stringify into a usable filesystem path.
    return Path(str(files("insight_agent"))) / "data"


def cmd_demo(args) -> int:
    """End-to-end on the bundled data: the 'does this work at all' command."""
    source = _bundled_data_dir()
    corpus = source / "sample_corpus.jsonl"

    print(f"Using the bundled sample corpus: {corpus}\n")

    args.traces = corpus
    loaded = _trace_loader(args)
    print(f"validate: {loaded.load().trace_count} trace(s), 0 error(s)")

    cov = corpus_coverage(loaded)
    print(f"coverage: {cov['rules']['evaluable']}/{cov['rules']['total']} IA3 rules evaluable\n")

    return _run_all(args, loaded)


# -- parser ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="insight-agent",
        description=(
            "Deterministic trace-evidence preprocessing. IA2 finds what is unusual and what "
            "recurs; IA3 finds what is demonstrably wrong with tool use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"insight-agent {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # schema
    p = sub.add_parser("schema", help="print the canonical JSON Schema")
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_schema)

    # validate
    p = sub.add_parser("validate", help="parse a canonical Trace JSONL corpus")
    p.add_argument("traces", type=Path)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_validate)

    # coverage
    p = sub.add_parser("coverage", help="report which IA3 rules this corpus can support")
    _add_file_corpus_arguments(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="list every rule")
    p.add_argument("--with-findings", action="store_true", help="also run IA3 and mark what fired")
    p.set_defaults(func=cmd_coverage)

    # explain-failures
    p = sub.add_parser(
        "explain-failures", help="compare IA2's and IA3's failure decoding call by call"
    )
    _add_trace_source_arguments(p)
    p.add_argument("--only-disagreements", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_explain_failures)

    # run-ia2
    p = sub.add_parser("run-ia2", help="anomalies, recurring patterns, and the Insight digest")
    _add_trace_source_arguments(p)
    _add_output_arguments(p)
    _add_anomaly_and_patterns_arguments(p)
    p.set_defaults(func=cmd_run_ia2)

    # run-ia3
    p = sub.add_parser("run-ia3", help="deterministic tool-issue detection and evidence cards")
    _add_trace_source_arguments(p)
    _add_output_arguments(p)
    _add_tool_issue_arguments(p)
    p.set_defaults(func=cmd_run_ia3)

    # run-all
    p = sub.add_parser("run-all", help="validate, then run IA2, IA3 and Insight compilation")
    _add_trace_source_arguments(p)
    _add_output_arguments(p)
    _add_anomaly_and_patterns_arguments(p)
    _add_tool_issue_arguments(p)
    p.add_argument(
        "--no-insights",
        action="store_true",
        help="skip Insight compilation and stop at IA2/IA3 evidence (no API key needed)",
    )
    p.add_argument("--model", default=None)
    p.add_argument("--api-base", default=None)
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.set_defaults(func=cmd_run_all, digest=None, cards=None, dry_run=False)

    # run-insights
    p = sub.add_parser(
        "run-insights", help="author Insights from the IA2 digest and IA3 cards (calls an LLM)"
    )
    _add_trace_source_arguments(p)
    _add_output_arguments(p)
    p.add_argument(
        "--model",
        default=None,
        help=f"LiteLLM model string (default: $INFERENCE_MODEL, else {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--api-base",
        default=None,
        help=f"OpenAI-compatible endpoint (default: {DEFAULT_API_BASE})",
    )
    p.add_argument(
        "--env-file", type=Path, default=None, help="path to a .env (default: autodetect)"
    )
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--digest", type=Path, default=None, help="use an existing digest.md")
    p.add_argument("--cards", type=Path, default=None, help="use an existing cards.json")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble and write the prompt, make no API call",
    )
    _add_anomaly_and_patterns_arguments(p)
    _add_tool_issue_arguments(p)
    p.set_defaults(func=cmd_run_insights)

    # demo
    p = sub.add_parser("demo", help="run everything on the bundled sample data")
    _add_output_arguments(p)
    _add_anomaly_and_patterns_arguments(p)
    _add_tool_issue_arguments(p)
    p.add_argument(
        "--no-insights",
        action="store_true",
        help="skip Insight compilation and stop at IA2/IA3 evidence (no API key needed)",
    )
    p.add_argument("--model", default=None)
    p.add_argument("--api-base", default=None)
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.set_defaults(func=cmd_demo, digest=None, cards=None, dry_run=False)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
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
