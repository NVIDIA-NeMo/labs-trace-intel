"""Command-line implementations.

Exit codes:
    0  success
    1  usage or runtime error
    2  schema (structural) validation errors
    3  lint errors in strict mode
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings as _warnings
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy
import sklearn

from insight_agent import __version__
from insight_agent.adapters.messages import adapt_file
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
from insight_agent.evidence_streams.contracts import EvidenceStreamResult
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
from insight_agent.insights_generation import DEFAULT_MAX_TOKENS as ANALYST_DEFAULT_MAX_TOKENS
from insight_agent.insights_generation import DEFAULT_MAX_TOOL_ROUNDS as ANALYST_DEFAULT_TOOL_ROUNDS
from insight_agent.insights_generation import DEFAULT_MODEL as ANALYST_DEFAULT_MODEL
from insight_agent.insights_generation import DEFAULT_PROMPT_VERSION as ANALYST_DEFAULT_PROMPT
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
    InsightTraceLoader,
    InsightTraceOptions,
    MLflowFileTraceConfig,
    MLflowFileTraceLoader,
    MLflowTraceConfig,
    MLflowTraceLoader,
    MLflowTraceLoadError,
    TraceLoader,
    TraceLoadError,
    load_tool_catalog,
    trace_schema,
    validate_corpus,
)
from insight_agent.traces import TraceSnapshot

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_SCHEMA = 2
EXIT_LINT = 3

STRUCTURAL_CODES = {"invalid_json", "not_an_object", "schema_violation", "unknown_field"}


# -- shared plumbing -------------------------------------------------------


def _add_corpus_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tool-catalog",
        type=Path,
        default=None,
        help="corpus-wide tool catalog JSON, used for records that carry none of their own",
    )
    parser.add_argument(
        "--allow-metric-shadowing",
        action="store_true",
        help="permit metrics that overwrite a built-in IA2 feature",
    )


def _add_file_corpus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("traces", type=Path, help="canonical JSONL corpus")
    _add_corpus_options(parser)


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
    _add_corpus_options(parser)


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


#: Loader warnings already shown this process.
_REPORTED_WARNINGS: set[str] = set()


def _trace_loader(args: argparse.Namespace) -> TraceLoader:
    """Construct the selected trace source without leaking it downstream."""
    experiment = getattr(args, "mlflow_experiment", None)
    export_path = getattr(args, "mlflow_export", None)
    if experiment or export_path:
        if getattr(args, "tool_catalog", None) is not None:
            raise ValueError("--tool-catalog is only supported with a canonical JSONL corpus")
        if getattr(args, "allow_metric_shadowing", False):
            raise ValueError(
                "--allow-metric-shadowing is only supported with a canonical JSONL corpus"
            )

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

    options = InsightTraceOptions(
        tool_catalog=load_tool_catalog(args.tool_catalog),
        allow_metric_shadowing=getattr(args, "allow_metric_shadowing", False),
        strict=True,
    )

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        loader = InsightTraceLoader.from_path(args.traces, options)

    for warning in caught:
        message = str(warning.message)
        if message not in _REPORTED_WARNINGS:
            _REPORTED_WARNINGS.add(message)
            print(f"note: {message}", file=sys.stderr)
    return loader


def _run_metadata(
    loader: TraceLoader,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Provenance recorded beside every result.

    Records the library versions because scikit-learn minor releases can change
    IsolationForest and KMeans output, and records whether steps were present
    because that changes IA2's feature vector.
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
    text = json.dumps(trace_schema(), indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return EXIT_OK


def cmd_validate(args) -> int:
    report = validate_corpus(args.traces, allow_metric_shadowing=args.allow_metric_shadowing)

    if args.json:
        sys.stdout.write(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n")
    else:
        shown = report.diagnostics[: args.max_errors] if args.max_errors else report.diagnostics
        for diagnostic in shown:
            print(diagnostic.format())
        hidden = len(report.diagnostics) - len(shown)
        if hidden > 0:
            print(f"... and {hidden} more (raise --max-errors to see them)")

        counts = {s: len(report.of(s)) for s in ("error", "warning", "info")}
        print(
            f"\n{report.record_count} record(s): "
            f"{counts['error']} error(s), {counts['warning']} warning(s), {counts['info']} info"
        )
        if report.ok:
            print("OK — this corpus can be loaded.")
            if counts["warning"]:
                print("Warnings do not block a run, but each one names a real downstream effect.")

    if not report.ok:
        structural = any(d.code in STRUCTURAL_CODES for d in report.errors)
        return EXIT_SCHEMA if structural else EXIT_LINT
    return EXIT_OK


def cmd_coverage(args) -> int:
    loader = _trace_loader(args)
    findings = None
    if args.with_findings:
        findings = detect(
            (to_ia3_trace(trace) for trace in loader.load().scan()),
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
    ia2_traces = {trace.id: to_ia2_trace(trace) for trace in snapshot.scan()}
    rows = []

    for trace in (to_ia3_trace(item) for item in snapshot.scan()):
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
    eligible = [card for card in cards if card.eligible_for_analyst]
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
        print(f"  eligible for analyst  : {len(eligible)}")
        print(f"  cards                 : {target / 'cards.md'}")
        if cards and not eligible:
            traces = tuple(snapshot.scan())
            distinct = len({trace.logical_case_id or trace.id for trace in traces})
            populated = any(trace.logical_case_id for trace in traces)
            print(
                f"  note: no card reached {artifacts.config.minimum_independent_cases} "
                "independent logical "
                "cases, i.e. no single issue type recurred across that many distinct cases."
            )
            if not populated:
                print(
                    "        No record sets logical_case_id, so every trace counts as its own "
                    "case. Populating it makes this count meaningful."
                )
            else:
                print(
                    f"        logical_case_id is populated ({distinct} distinct cases), so this "
                    "is most likely correct: nothing recurred often enough yet. Do not merge "
                    "distinct tasks under one case id to clear the threshold."
                )
    return EXIT_OK


def cmd_run_ia3(args: argparse.Namespace) -> int:
    loader = _trace_loader(args)
    snapshot = loader.load()

    evidence = _registered_evidence_streams(args, TOOL_ISSUES).analyze(TOOL_ISSUES, snapshot)
    return _write_tool_issues(args, loader, snapshot, evidence)


def _render_cards(cards: Sequence[ToolIssueCard], *, total: int | None = None) -> str:
    lines = ["# IA3 evidence cards", ""]
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
            "are in `cards.json`; re-run with `--all-cards` to render them here too.",
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


def _run_all(args: argparse.Namespace, loader: TraceLoader) -> int:
    if not args.no_analyst:
        missing = _analyst_preflight(args)
        if missing:
            print(f"error: {missing}", file=sys.stderr)
            print("       pass --no-analyst to run IA2 and IA3 only.", file=sys.stderr)
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

    analyst_ran = False
    if not args.no_analyst:
        if not args.agent:
            args.agent = (
                getattr(args, "mlflow_experiment", None)
                or Path(getattr(args, "mlflow_export", None) or args.traces).stem
            )
        code = _run_insights(args, loader, snapshot, (anomaly_and_patterns, tool_issues))
        if code != EXIT_OK:
            return code
        analyst_ran = True
        if not args.quiet:
            print()

    (args.out / "index.md").write_text(
        _render_index(str(loader.describe()["source"]), has_analyst=analyst_ran),
        encoding="utf-8",
    )
    if not args.quiet:
        print(f"index: {args.out / 'index.md'}")
    return EXIT_OK


def cmd_run_all(args: argparse.Namespace) -> int:
    return _run_all(args, _trace_loader(args))


def _analyst_preflight(args: argparse.Namespace) -> str | None:
    load_dotenv(getattr(args, "env_file", None))
    if not (resolve(ENV_API_KEY) or "").strip():
        return "the Analyst needs an API key; set INSIGHT_AGENT_API_KEY"
    if "litellm" in sys.modules:
        litellm_available = sys.modules["litellm"] is not None
    else:
        litellm_available = importlib.util.find_spec("litellm") is not None
    if not litellm_available:
        return "the Analyst needs litellm, which is not installed"
    return None


def _render_index(source: str, *, has_analyst: bool) -> str:
    lines = [
        "# Insight Agent run",
        "",
        f"Trace source: `{source}`",
        "",
        "## IA2 — what is unusual, and what recurs",
        "",
        "- [digest.md](ia2/digest.md) — the packet written for the Analyst",
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


def _insights_inputs(args):
    """Load a snapshot and the evidence consumed by InsightsGeneration.

    Either reads artifacts a previous run already produced, or computes them
    in-process. Explicit `--digest`/`--cards` wins so a paid Analyst call can be
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


def _run_insights(
    args: argparse.Namespace,
    loader: TraceLoader,
    snapshot: TraceSnapshot,
    evidence: Sequence[EvidenceStreamResult],
) -> int:
    # Credentials are only needed here, so `.env` is only read here. A purely
    # deterministic run never touches the filesystem for configuration.
    loaded = load_dotenv(getattr(args, "env_file", None))
    model = args.model or resolve(ENV_MODEL) or ANALYST_DEFAULT_MODEL
    api_base = getattr(args, "api_base", None) or resolve(ENV_API_BASE)
    api_key = resolve(ENV_API_KEY)

    try:
        generation = InsightsGeneration(
            snapshot=snapshot,
            evidence=evidence,
            agent=args.agent,
            corpus=loader.describe(),
            prompt_version=args.prompt_version,
        )
    except (InsightsGenerationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    target = args.out / "analyst"
    target.mkdir(parents=True, exist_ok=True)

    system, user = generation.build_prompt()
    (target / "prompt.md").write_text(
        f"<!-- system -->\n\n{system}\n\n<!-- user -->\n\n{user}\n", encoding="utf-8"
    )

    if args.dry_run:
        approx = (len(system) + len(user)) // 4
        print("dry run — no API call made")
        print(f"  agent                 : {args.agent}")
        print(f"  model                 : {model}")
        print(f"  prompt version        : {args.prompt_version}")
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
    if args.temperature is not None:
        # Unset by default on purpose: temperature is rejected outright by some
        # current models. Only sent when explicitly asked for.
        extra["temperature"] = args.temperature

    try:
        result = generation.generate(
            model=model,
            max_tokens=args.max_tokens,
            api_base=api_base,
            api_key=api_key,
            max_tool_rounds=args.max_tool_rounds,
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
            "agent": args.agent,
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

    if not args.quiet:
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
            available = {trace.id for trace in snapshot.scan()}
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


def cmd_run_analyst(args: argparse.Namespace) -> int:
    """Author Insights from a snapshot and its evidence streams.

    The only command in the package that calls out to a model, and the only
    non-deterministic one. `run-all` and `demo` invoke it by default; pass
    `--no-analyst` to either for a purely deterministic run.
    """

    try:
        loader, snapshot, evidence = _insights_inputs(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _run_insights(args, loader, snapshot, evidence)


def cmd_adapt_messages(args) -> int:
    records = adapt_file(
        args.input,
        fmt=args.format,
        tool_catalog_path=args.tool_catalog,
        trace_id_prefix=args.trace_id_prefix,
    )
    text = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for r in records
    )
    if str(args.out) == "-":
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {len(records)} trace(s) to {args.out}")
        print(f"next: uv run insight-agent validate {args.out}")
    return EXIT_OK


def _bundled_data_dir() -> Path:
    # Resolve from the parent package: `data/` has no __init__.py, so
    # files("insight_agent.data") would return a MultiplexedPath that does not
    # stringify into a usable filesystem path.
    return Path(str(files("insight_agent"))) / "data"


ADAPTER_TEMPLATE = '''#!/usr/bin/env -S uv run
"""Adapter: {name} -> Insight Agent canonical JSONL (insight-trace/v1).

Verify loop — run these after every change, never batch them:

    {run_path} > traces.jsonl
    uv run insight-agent validate traces.jsonl     # must exit 0
    uv run insight-agent coverage traces.jsonl     # what can actually fire?
    uv run insight-agent run-ia3 traces.jsonl -o out

Then open out/ia3/cards.json and check three findings against the raw source by
hand. A rule that fires on 100% of calls is an adapter bug, not a discovery.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from typing import Any

SCHEMA_VERSION = "insight-trace/v1"


def load_corpus_context(path: str) -> dict[str, Any]:
    """Corpus-level data that individual records need.

    Tool definitions often live once at the top of an export rather than on
    every record, so they are read here and threaded into `to_canonical`.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    tools = data.get("tools") if isinstance(data, dict) else None
    return {{"tool_catalog": {{t["name"]: t.get("parameters") for t in tools}} if tools else None}}


def iter_source_records(path: str) -> Iterator[Any]:
    """Yield one source record per trace. Adjust to your input format."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    yield from (data if isinstance(data, list) else [data])


def to_canonical(source: Any, index: int, context: dict[str, Any]) -> dict[str, Any]:
    """Map one source record to one canonical trace.

    Start with the required fields only, get `validate` to exit 0, then add one
    optional field at a time and re-check `coverage`.
    """
    trace_id = str(source.get("id") or f"trace-{{index:05d}}")

    calls = []
    for position, raw in enumerate(source.get("tool_calls", [])):
        call: dict[str, Any] = {{
            # Required four:
            "call_id": str(raw.get("id") or f"{{trace_id}}#{{position}}"),
            "call_index": position,
            "tool_name": str(raw["name"]),
            "arguments": raw.get("arguments"),   # emit RAW, do not re-parse
        }}
        # Omit "result" entirely when no result was recorded: absence is what
        # fires missing_tool_result. "result": None means the tool returned null.
        # Textual results go under "content" -- IA3 unwraps nothing else.
        if "result" in raw:
            found = raw["result"]
            call["result"] = {{"content": found}} if isinstance(found, str) else found
        calls.append(call)

    record: dict[str, Any] = {{
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "calls": calls,
        # Add next, in this order:
        #   "logical_case_id": "...",       makes card eligibility honest
        #   "source_pointer": {{...}},        makes evidence reopenable
        #   "steps": [...],                 real IA2 trajectory tokens
    }}
    # Highest-value optional field: unlocks six argument-contract rules.
    if context.get("tool_catalog"):
        record["tool_catalog"] = context["tool_catalog"]
    return record


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {{argv[0]}} SOURCE", file=sys.stderr)
        return 2
    context = load_corpus_context(argv[1])
    for index, source in enumerate(iter_source_records(argv[1])):
        record = to_canonical(source, index, context)
        print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
'''


def cmd_init_adapter(args) -> int:
    args.dir.mkdir(parents=True, exist_ok=True)
    path = args.dir / f"{args.name}.py"
    if path.exists() and not args.force:
        print(f"{path} already exists; pass --force to overwrite.")
        return EXIT_ERROR

    run_path = str(path) if path.is_absolute() else f"./{path}"
    path.write_text(
        ADAPTER_TEMPLATE.format(name=args.name, path=path, run_path=run_path), encoding="utf-8"
    )
    path.chmod(path.stat().st_mode | 0o111)
    print(f"wrote {path}\n")
    print("The verify loop — run every iteration, never batch it:")
    print("  1. uv run insight-agent schema                  # read the contract")
    print("  2. inspect 2-3 raw source records; find where the call->result link lives")
    print(f"  3. {run_path} SOURCE > traces.jsonl   # required fields only, first")
    print("  4. uv run insight-agent validate traces.jsonl   # until it exits 0")
    print("  5. uv run insight-agent coverage traces.jsonl   # see which rules abstain")
    print("  6. add ONE optional field, then repeat 3-5")
    print("  7. uv run insight-agent run-ia3 traces.jsonl -o out, then check 3 findings by hand")
    return EXIT_OK


def cmd_demo(args) -> int:
    """End-to-end on the bundled data: the 'does this work at all' command."""
    source = _bundled_data_dir()
    corpus = source / "sample_corpus.jsonl"

    print(f"Using the bundled sample corpus: {corpus}\n")

    args.traces = corpus
    args.tool_catalog = None
    args.allow_metric_shadowing = False

    loaded = _trace_loader(args)
    print(
        f"validate: {loaded.report.record_count} records, "
        f"{len(loaded.report.errors)} error(s), {len(loaded.report.warnings)} warning(s)"
    )

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
    p = sub.add_parser("validate", help="check a corpus against the schema and the lints")
    p.add_argument("traces", type=Path)
    p.add_argument("--allow-metric-shadowing", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--max-errors", type=int, default=50)
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
    p = sub.add_parser("run-ia2", help="anomalies, recurring patterns, and the Analyst digest")
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
    p = sub.add_parser("run-all", help="validate, then run IA2, IA3 and the Analyst")
    _add_trace_source_arguments(p)
    _add_output_arguments(p)
    _add_anomaly_and_patterns_arguments(p)
    _add_tool_issue_arguments(p)
    p.add_argument(
        "--no-analyst",
        action="store_true",
        help="skip the Analyst and stop at IA2/IA3 evidence (no API key needed)",
    )
    p.add_argument(
        "--agent",
        default=None,
        help="name of the agent under test (defaults to the corpus filename)",
    )
    p.add_argument("--model", default=None)
    p.add_argument("--api-base", default=None)
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("--prompt-version", default=ANALYST_DEFAULT_PROMPT)
    p.add_argument("--max-tool-rounds", type=int, default=ANALYST_DEFAULT_TOOL_ROUNDS)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=ANALYST_DEFAULT_MAX_TOKENS)
    p.set_defaults(func=cmd_run_all, digest=None, cards=None, dry_run=False)

    # run-analyst
    p = sub.add_parser(
        "run-analyst", help="author Insights from the IA2 digest and IA3 cards (calls an LLM)"
    )
    _add_trace_source_arguments(p)
    _add_output_arguments(p)
    p.add_argument("--agent", required=True, help="name of the agent under test")
    p.add_argument(
        "--model",
        default=None,
        help=f"any litellm model string (default: $INSIGHT_AGENT_MODEL, else {ANALYST_DEFAULT_MODEL})",
    )
    p.add_argument(
        "--api-base",
        default=None,
        help="OpenAI-compatible endpoint (default: $INSIGHT_AGENT_API_BASE)",
    )
    p.add_argument(
        "--env-file", type=Path, default=None, help="path to a .env (default: autodetect)"
    )
    p.add_argument(
        "--prompt-version",
        default=ANALYST_DEFAULT_PROMPT,
        help=f"packaged prompt to use (default: {ANALYST_DEFAULT_PROMPT})",
    )
    p.add_argument(
        "--max-tool-rounds",
        type=int,
        default=ANALYST_DEFAULT_TOOL_ROUNDS,
        help="cap on trace-lookup rounds, for prompts that use the fetch tool",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="only sent when given; several current models reject it outright",
    )
    p.add_argument("--max-tokens", type=int, default=ANALYST_DEFAULT_MAX_TOKENS)
    p.add_argument("--digest", type=Path, default=None, help="use an existing digest.md")
    p.add_argument("--cards", type=Path, default=None, help="use an existing cards.json")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble and write the prompt, make no API call",
    )
    _add_anomaly_and_patterns_arguments(p)
    _add_tool_issue_arguments(p)
    p.set_defaults(func=cmd_run_analyst)

    # adapt-messages
    p = sub.add_parser(
        "adapt-messages", help="convert Anthropic/OpenAI message lists to canonical JSONL"
    )
    p.add_argument("input", type=Path)
    p.add_argument("--format", choices=("anthropic", "openai", "auto"), default="auto")
    p.add_argument("-o", "--out", type=Path, default=Path("traces.jsonl"))
    p.add_argument("--tool-catalog", type=Path, default=None)
    p.add_argument("--trace-id-prefix", default="trace")
    p.set_defaults(func=cmd_adapt_messages)

    # init-adapter
    p = sub.add_parser("init-adapter", help="scaffold a new adapter and print the verify loop")
    p.add_argument("name")
    p.add_argument("--dir", type=Path, default=Path("adapters"))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init_adapter)

    # demo
    p = sub.add_parser("demo", help="run everything on the bundled sample data")
    _add_output_arguments(p)
    _add_anomaly_and_patterns_arguments(p)
    _add_tool_issue_arguments(p)
    p.add_argument(
        "--no-analyst",
        action="store_true",
        help="skip the Analyst and stop at IA2/IA3 evidence (no API key needed)",
    )
    p.add_argument(
        "--agent",
        default=None,
        help="name of the agent under test (defaults to the corpus filename)",
    )
    p.add_argument("--model", default=None)
    p.add_argument("--api-base", default=None)
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("--prompt-version", default=ANALYST_DEFAULT_PROMPT)
    p.add_argument("--max-tool-rounds", type=int, default=ANALYST_DEFAULT_TOOL_ROUNDS)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=ANALYST_DEFAULT_MAX_TOKENS)
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
        if isinstance(exc, TraceLoadError):
            print(f"error: {exc}", file=sys.stderr)
            for diagnostic in exc.report.errors[:20]:
                print(f"  {diagnostic.format()}", file=sys.stderr)
            return EXIT_SCHEMA
        if isinstance(exc, MLflowTraceLoadError):
            print(f"error: {exc}", file=sys.stderr)
            if isinstance(exc.__cause__, ImportError) or "not installed" in str(exc).lower():
                print("       install it with: uv sync --extra mlflow", file=sys.stderr)
            return EXIT_ERROR
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
