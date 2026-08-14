"""Command-line interface.

Every subcommand imports its engine lazily. ``insight-agent validate`` and
``insight-agent schema`` are the commands an adapter author runs dozens of
times in a row, and they have no reason to pay the ~1s scikit-learn import.

Exit codes:
    0  success
    1  usage or runtime error
    2  schema (structural) validation errors
    3  lint errors in strict mode
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import __version__

# analyst.py is import-light (no litellm at module scope), so naming its
# defaults here costs nothing and keeps `--help` accurate.
from .analyst import DEFAULT_MAX_TOKENS as ANALYST_DEFAULT_MAX_TOKENS
from .analyst import DEFAULT_MAX_TOOL_ROUNDS as ANALYST_DEFAULT_TOOL_ROUNDS
from .analyst import DEFAULT_MODEL as ANALYST_DEFAULT_MODEL
from .analyst import DEFAULT_PROMPT_VERSION as ANALYST_DEFAULT_PROMPT

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_SCHEMA = 2
EXIT_LINT = 3

STRUCTURAL_CODES = {"invalid_json", "not_an_object", "schema_violation", "unknown_field"}


# -- shared plumbing -------------------------------------------------------


def _add_corpus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("traces", type=Path, help="canonical JSONL corpus")
    parser.add_argument(
        "--tool-catalog",
        type=Path,
        default=None,
        help="corpus-wide tool catalog JSON, used for records that carry none of their own",
    )
    parser.add_argument(
        "--profile", type=Path, default=None, help="venue profile JSON (see `insight-agent sample`)"
    )
    parser.add_argument(
        "--allow-metric-shadowing",
        action="store_true",
        help="permit metrics that overwrite a built-in IA2 feature",
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("out"), help="output directory (default: ./out)"
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the stdout summary")


#: Lint messages already shown this process. ``run-all`` loads the corpus once
#: per phase, and repeating the same warning three times with a stack trace
#: attached trains people to ignore it.
_REPORTED_WARNINGS: set[str] = set()


def _load(args: argparse.Namespace):
    """Load a corpus using the standard corpus arguments."""
    import warnings as _warnings

    from .loader import LoadOptions, load_corpus, load_tool_catalog
    from .venue import load_profile

    options = LoadOptions(
        tool_catalog=load_tool_catalog(args.tool_catalog),
        profile=load_profile(args.profile),
        allow_metric_shadowing=getattr(args, "allow_metric_shadowing", False),
        strict=True,
    )

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        corpus = load_corpus(args.traces, options)

    for warning in caught:
        message = str(warning.message)
        if message not in _REPORTED_WARNINGS:
            _REPORTED_WARNINGS.add(message)
            print(f"note: {message}", file=sys.stderr)
    return corpus


def _run_metadata(corpus, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provenance recorded beside every result.

    Records the library versions because scikit-learn minor releases can change
    IsolationForest and KMeans output, and records whether steps were present
    because that changes IA2's feature vector.
    """
    import numpy
    import sklearn

    payload = {
        "insight_agent_version": __version__,
        "numpy_version": numpy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "corpus": corpus.describe(),
        "venue_profile": corpus.options.profile.to_dict(),
    }
    if extra:
        payload["parameters"] = extra
    return payload


# -- commands --------------------------------------------------------------


def cmd_schema(args) -> int:
    from .validate import trace_schema

    text = json.dumps(trace_schema(), indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return EXIT_OK


def cmd_validate(args) -> int:
    from .validate import validate_corpus

    report = validate_corpus(
        args.traces, allow_metric_shadowing=args.allow_metric_shadowing
    )

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
    from .coverage import corpus_coverage, format_coverage

    corpus = _load(args)
    findings = None
    if args.with_findings:
        from .ia3_tid import detect

        findings = detect(corpus.ia3(), profile=corpus.options.profile)

    report = corpus_coverage(corpus, findings=findings)
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
    from .ia2_pipeline import decode_explicit_failure
    from .ia3_tid import strict_failure

    corpus = _load(args)
    profile = corpus.options.profile
    ia2_traces = {t.trace_id: t for t in corpus.ia2()}
    rows = []

    for trace in corpus.ia3():
        ia2_calls = {c.call_id: c for c in ia2_traces[trace.trace_id].calls}
        for call in trace.calls:
            ia3_failed, ia3_marker = strict_failure(call, profile=profile)
            ia2_call = ia2_calls.get(call.call_id)
            if ia2_call is None:
                continue
            ia2_failed, ia2_marker, _ = decode_explicit_failure(
                ia2_call.tool_name, ia2_call.result, profile=profile
            )
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
        print(f"{row['trace_id'][:27]:<28} {row['call_id'][:25]:<26} "
              f"{row['tool_name'][:19]:<20} {ia2[:23]:<24} {ia3[:23]:<24}{flag}")

    disagreements = sum(1 for r in rows if not r["agree"])
    print(f"\n{len(rows)} call(s), {disagreements} disagreement(s).")
    if disagreements:
        print(
            "Disagreements usually mean result text is wrapped under a key IA3 does not "
            "unwrap. Use {\"content\": ...} for textual results."
        )
    return EXIT_OK


def cmd_run_ia2(args) -> int:
    from .ia2_pipeline import run_ia2
    from .serialize import prepared_features, write_json

    corpus = _load(args)
    traces = corpus.ia2()
    cluster_candidates = tuple(int(k) for k in args.cluster_candidates.split(","))
    feature_names = tuple(args.feature) if args.feature else None

    kwargs: dict[str, Any] = {
        "contamination": args.contamination,
        "cluster_candidates": cluster_candidates,
        "minimum_independent_traces": args.min_independent_traces,
        "input_scaling": args.input_scaling,
        "profile": corpus.options.profile,
    }
    if feature_names:
        kwargs["feature_names"] = feature_names

    result = run_ia2(traces, **kwargs)

    if args.out == Path("-"):
        sys.stdout.write(result["digest"])
        return EXIT_OK

    target = args.out / "ia2"
    target.mkdir(parents=True, exist_ok=True)
    (target / "digest.md").write_text(result["digest"], encoding="utf-8")
    write_json(target / "anomalies.json", result["anomalies"])
    write_json(target / "trajectory_groups.json", result["trajectory_groups"])
    write_json(target / "verdict_groups.json", result["verdict_groups"])
    write_json(target / "failure_groups.json", result["failure_groups"])
    write_json(target / "cross_tool_failure_groups.json", result["cross_tool_failure_groups"])
    write_json(target / "failure_events.json", result["failure_events"])
    write_json(target / "features.json", prepared_features(result["prepared"]))
    write_json(target / "run.json", _run_metadata(corpus, kwargs | {
        "feature_names": list(feature_names) if feature_names else "default",
    }))

    if not args.quiet:
        flagged = [a for a in result["anomalies"] if a["is_anomaly"]]
        print(f"IA2 over {len(traces)} traces -> {target}")
        print(f"  unusual traces        : {len(flagged)}")
        print(f"  trajectory clusters   : "
              f"{len(result['trajectory_groups']['clusters']) if result['trajectory_groups'] else 'not evaluable'}")
        print(f"  verdict groups        : {len(result['verdict_groups'])}")
        print(f"  failure groups        : {len(result['failure_groups'])}")
        print(f"  cross-tool groups     : {len(result['cross_tool_failure_groups'])}")
        print(f"  digest                : {target / 'digest.md'}")
        expected = len(traces) * args.contamination
        if expected < 1:
            print(
                f"  note: contamination={args.contamination} on {len(traces)} traces expects "
                f"{expected:.2f} flags. The default is tuned for corpora in the thousands; "
                "try --contamination 0.15 on a small sample."
            )
    return EXIT_OK


def cmd_run_ia3(args) -> int:
    from .ia3_tid import build_cards, catalog_coverage, detect
    from .serialize import write_json

    corpus = _load(args)
    traces = corpus.ia3()
    findings = detect(
        traces, profile=corpus.options.profile, retry_threshold=args.retry_threshold
    )
    cards = build_cards(findings, minimum_independent_cases=args.min_independent_cases)

    # cards.json always holds the complete set — dropping audit-only cards from
    # the machine-readable output would hide near-misses. --all-cards controls
    # only how much of it the human-readable rendering shows.
    eligible = [c for c in cards if c["eligible_for_analyst"]]
    rendered = cards if (args.all_cards or not eligible) else eligible

    target = args.out / "ia3"
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "findings.json", findings)
    write_json(target / "cards.json", cards)
    write_json(target / "coverage.json", catalog_coverage(findings))
    (target / "cards.md").write_text(
        _render_cards(rendered, total=len(cards)), encoding="utf-8"
    )
    write_json(target / "run.json", _run_metadata(corpus, {
        "retry_threshold": args.retry_threshold,
        "minimum_independent_cases": args.min_independent_cases,
    }))

    if not args.quiet:
        eligible = [c for c in cards if c["eligible_for_analyst"]]
        print(f"IA3 over {len(traces)} traces -> {target}")
        print(f"  findings              : {len(findings)}")
        print(f"  distinct issue types  : {len({f['issue_type'] for f in findings})}/19")
        print(f"  cards                 : {len(cards)}")
        print(f"  eligible for analyst  : {len(eligible)}")
        print(f"  cards                 : {target / 'cards.md'}")
        if cards and not eligible:
            distinct = len({r.get("logical_case_id") or r["trace_id"] for r in corpus.records})
            populated = any(r.get("logical_case_id") for r in corpus.records)
            print(
                f"  note: no card reached {args.min_independent_cases} independent logical "
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


def _render_cards(cards: Sequence[dict[str, Any]], *, total: int | None = None) -> str:
    lines = ["# IA3 evidence cards", ""]
    eligible = [c for c in cards if c["eligible_for_analyst"]]
    total = len(cards) if total is None else total
    lines.append(
        f"{total} card(s) in total; {len(eligible)} reached the independent-case threshold "
        "and are eligible for the Analyst."
    )
    if total > len(cards):
        lines.append("")
        lines.append(
            f"Showing the {len(cards)} eligible card(s). The remaining "
            f"{total - len(cards)} are in `cards.json`; re-run with `--all-cards` to render "
            "them here too."
        )
    lines.append("")
    lines.append(
        "A card is not an Insight. It is recurring, attributable evidence that a human or "
        "Analyst LLM still has to interpret. `impact_status` is never established here."
    )
    lines.append("")
    for card in sorted(cards, key=lambda c: (not c["eligible_for_analyst"], c["card_id"])):
        mark = "ELIGIBLE" if card["eligible_for_analyst"] else "audit only"
        lines.append(f"## {card['card_id']}  ({mark})")
        lines.append("")
        lines.append(f"- family: {card['issue_family']}")
        lines.append(f"- mechanism: {card['mechanism_key']}")
        lines.append(f"- findings: {card['finding_count']}")
        lines.append(f"- independent logical cases: {card['independent_case_count']}")
        lines.append(f"- impact: {card['impact_status']}")
        lines.append("")
        for evidence in card.get("representative_evidence", [])[:3]:
            observation = str(evidence.get("observation", "")).strip()
            pointer = json.dumps(evidence.get("source_pointer", {}), sort_keys=True)
            lines.append(
                f"  - `{evidence.get('trace_id')}` call `{evidence.get('call_id')}` "
                f"({evidence.get('tool_name')}): {observation}"
            )
            lines.append(f"    source: `{pointer}`")
        lines.append("")
    return "\n".join(lines)


def _analyst_preflight(args) -> str | None:
    """Why the Analyst cannot run, or None if it can.

    Checked before IA2/IA3 output is consumed so a missing key is reported as
    one actionable line rather than as a traceback from inside litellm.
    """
    from .config import ENV_API_KEY, load_dotenv, resolve

    # Credentials are resolved BEFORE litellm is imported, and the order is
    # load-bearing: importing litellm runs `dotenv.load_dotenv()` as a side
    # effect, which pulls any `.env` in the cwd into os.environ and would
    # silently override an explicit --env-file.
    load_dotenv(getattr(args, "env_file", None))
    if not (resolve(ENV_API_KEY) or "").strip():
        return (
            "the Analyst needs an API key. Copy .env.example to .env and set "
            "INSIGHT_AGENT_API_KEY (see docs/analyst.md)"
        )

    try:
        import litellm  # noqa: F401
    except ImportError:
        return (
            "the Analyst needs litellm, which is not installed. "
            'Install it with: pip install "insight-agent[analyst]"'
        )
    return None


def cmd_run_all(args) -> int:
    from .validate import validate_corpus

    report = validate_corpus(args.traces, allow_metric_shadowing=args.allow_metric_shadowing)
    if not report.ok:
        print(f"{len(report.errors)} validation error(s); run `insight-agent validate` for detail.")
        return EXIT_SCHEMA

    args.out.mkdir(parents=True, exist_ok=True)
    for command in (cmd_run_ia2, cmd_run_ia3):
        code = command(args)
        if code != EXIT_OK:
            return code
        if not args.quiet:
            print()

    # The Analyst runs by default. A run that stops at evidence is a partial
    # run, so an unusable Analyst is a hard failure rather than a silent skip —
    # `--no-analyst` is the way to ask for evidence only.
    analyst_ran = False
    if not getattr(args, "no_analyst", False):
        missing = _analyst_preflight(args)
        if missing:
            print(f"error: {missing}", file=sys.stderr)
            print("       pass --no-analyst to run IA2 and IA3 only.", file=sys.stderr)
            return EXIT_ERROR
        # Reuse the digest and cards just written rather than recomputing them.
        args.digest = args.out / "ia2" / "digest.md"
        args.cards = args.out / "ia3" / "cards.json"
        args.dry_run = getattr(args, "dry_run", False)
        if not getattr(args, "agent", None):
            # The prompt only uses this as a label for the system under test.
            args.agent = Path(args.traces).stem
        code = cmd_run_analyst(args)
        if code != EXIT_OK:
            return code
        analyst_ran = True
        if not args.quiet:
            print()

    index = [
        "# Insight Agent run",
        "",
        f"Corpus: `{args.traces}`",
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
    if analyst_ran:
        index += [
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
    index += [
        "## Reading these outputs",
        "",
        "An anomaly is not an error. A card is not an Insight. Abstention is a designed",
        "outcome, not a failure: when evidence is missing the detectors decline to guess.",
        "",
    ]
    (args.out / "index.md").write_text("\n".join(index), encoding="utf-8")
    if not args.quiet:
        print(f"index: {args.out / 'index.md'}")
    return EXIT_OK


def _analyst_inputs(args) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Produce (digest, cards, corpus_description) for the Analyst.

    Either reads artifacts a previous run already produced, or computes them
    in-process. Explicit `--digest`/`--cards` wins so a paid Analyst call can be
    re-issued against a frozen evidence set without re-running the engines.
    """
    if bool(args.digest) != bool(args.cards):
        raise ValueError("--digest and --cards must be given together")

    corpus = _load(args)

    if args.digest:
        digest = Path(args.digest).read_text(encoding="utf-8")
        cards = json.loads(Path(args.cards).read_text(encoding="utf-8"))
        # `anomalies.json` and `findings.json` sit beside the artifacts a
        # previous run wrote; use them when present so the cross-stream section
        # is identical to the computed path.
        anomalies = _read_sibling(args.digest, "anomalies.json") or []
        finding_rows = _read_sibling(args.cards, "findings.json") or []
        return (digest, cards, corpus.describe(),
                _uncovered(anomalies, finding_rows), _records_by_id(corpus))

    from .ia2_pipeline import run_ia2
    from .ia3_tid import build_cards, detect

    result = run_ia2(
        corpus.ia2(),
        minimum_independent_traces=args.min_independent_traces,
        contamination=args.contamination,
        profile=corpus.options.profile,
    )
    findings = detect(corpus.ia3(), profile=corpus.options.profile)
    cards = build_cards(findings, minimum_independent_cases=args.min_independent_cases)
    return (
        result["digest"],
        cards,
        corpus.describe(),
        _uncovered(result["anomalies"], findings),
        _records_by_id(corpus),
    )


def _records_by_id(corpus) -> dict[str, Any]:
    """Raw canonical records, keyed by trace_id, for the lookup tool."""
    return {r["trace_id"]: r for r in corpus.records}


def _read_sibling(reference: Path, name: str) -> Any:
    candidate = Path(reference).parent / name
    if not candidate.is_file():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _uncovered(
    anomalies: Sequence[Mapping[str, Any]], findings: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Anomaly-flagged traces on which no IA3 rule fired.

    Not derivable from the prompt text: a card exposes only three
    representative traces, so a trace can carry findings and still be absent
    from every card. Computed from the full finding list instead.
    """
    covered = {f["trace_id"] for f in findings}
    return [
        {
            "trace_id": row["trace_id"],
            "anomaly_score": row["anomaly_score"],
            "anomaly_reasons": list(row.get("anomaly_reasons") or []),
        }
        for row in anomalies
        if row.get("is_anomaly") and row["trace_id"] not in covered
    ]


def cmd_run_analyst(args) -> int:
    """Author Insights from the IA2 digest and IA3 cards.

    The only command in the package that calls out to a model, and the only
    non-deterministic one. `run-all` and `demo` invoke it by default; pass
    `--no-analyst` to either for a purely deterministic run.
    """
    from .analyst import (
        AnalystError,
        AnalystRequest,
        ResponseParseError,
        author_insights,
        build_prompt,
        select_cards,
    )
    from .config import ENV_API_BASE, ENV_API_KEY, ENV_MODEL, load_dotenv, resolve
    from .serialize import write_json

    # Credentials are only needed here, so `.env` is only read here. A purely
    # deterministic run never touches the filesystem for configuration.
    loaded = load_dotenv(getattr(args, "env_file", None))
    model = args.model or resolve(ENV_MODEL) or ANALYST_DEFAULT_MODEL
    api_base = getattr(args, "api_base", None) or resolve(ENV_API_BASE)
    api_key = resolve(ENV_API_KEY)

    try:
        digest, all_cards, corpus_description, uncovered, records = _analyst_inputs(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    cards, withheld = select_cards(all_cards, all_cards=args.all_cards)
    request = AnalystRequest(
        agent=args.agent,
        digest=digest,
        cards=cards,
        withheld_card_count=withheld,
        corpus=corpus_description,
        prompt_version=args.prompt_version,
        uncovered_anomalies=uncovered,
    )

    target = args.out / "analyst"
    target.mkdir(parents=True, exist_ok=True)

    system, user = build_prompt(request)
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
        print(f"  cards shown / withheld: {len(cards)} / {withheld}")
        print(f"  uncovered anomalies   : {len(uncovered)}")
        print(f"  prompt                : {target / 'prompt.md'}")
        print(f"  approx input tokens   : {approx:,}")
        return EXIT_OK

    extra: dict[str, Any] = {}
    if args.temperature is not None:
        # Unset by default on purpose: temperature is rejected outright by some
        # current models. Only sent when explicitly asked for.
        extra["temperature"] = args.temperature

    try:
        result = author_insights(
            request,
            model=model,
            max_tokens=args.max_tokens,
            api_base=api_base,
            api_key=api_key,
            trace_records=records,
            max_tool_rounds=args.max_tool_rounds,
            **extra,
        )
    except ResponseParseError as exc:
        (target / "analyst_raw.txt").write_text(exc.raw, encoding="utf-8")
        print(f"error: {exc}", file=sys.stderr)
        print(f"       raw response saved to {target / 'analyst_raw.txt'}", file=sys.stderr)
        return EXIT_ERROR
    except AnalystError as exc:
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
            "cards_shown": len(cards),
            "cards_withheld": withheld,
            "insight_count": len(result.insights),
            "corpus": corpus_description,
        },
    )

    if not args.quiet:
        print(f"Analyst over {len(cards)} card(s) -> {target}")
        print(f"  model                 : {result.model}")
        if result.usage:
            tokens = result.usage.get("total_tokens") or result.usage.get("prompt_tokens")
            if tokens:
                print(f"  tokens                : {tokens:,}")
        if result.tool_calls:
            print(f"  trace lookups         : {result.tool_calls} call(s), "
                  f"{len(result.traces_fetched)} distinct trace(s)")
        print(f"  insights              : {len(result.insights)}")
        print(f"  insights              : {target / 'insights.json'}")

        if withheld:
            print(
                f"  note: {withheld} audit-only card(s) were withheld (they did not recur "
                "across enough independent cases). Pass --all-cards to include them."
            )
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
        known = _known_trace_ids(digest, cards) | set(result.traces_fetched)
        unknown = sorted(result.cited_trace_ids - known) if known else []
        if unknown:
            # Distinguish "read it but did not show us" from "does not exist".
            fabricated = sorted(t for t in unknown if t not in records)
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


def _known_trace_ids(digest: str, cards: Sequence[Mapping[str, Any]]) -> set[str]:
    """Trace IDs the Analyst was actually shown."""
    known = {
        str(e.get("trace_id"))
        for card in cards
        for e in card.get("representative_evidence", [])
        if e.get("trace_id")
    }
    # Digest tables render trace ids inside backticks.
    known |= set(re.findall(r"`([^`\s|]+)`", digest))
    return known


def cmd_adapt_messages(args) -> int:
    from .adapters.messages import adapt_file

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
        print(f"next: insight-agent validate {args.out}")
    return EXIT_OK


def _bundled_data_dir() -> Path:
    # Resolve from the parent package: `data/` has no __init__.py, so
    # files("insight_agent.data") would return a MultiplexedPath that does not
    # stringify into a usable filesystem path.
    from importlib.resources import files

    return Path(str(files("insight_agent"))) / "data"


def cmd_sample(args) -> int:
    source = _bundled_data_dir()
    available = sorted(p.name for p in source.iterdir() if p.suffix in {".json", ".jsonl"})

    if args.list or not args.copy:
        print(f"Bundled sample data ({source}):")
        for name in available:
            print(f"  {name}")
        if not args.copy:
            print("\nCopy it out with: insight-agent sample --copy ./sample")
        return EXIT_OK

    args.copy.mkdir(parents=True, exist_ok=True)
    for name in available:
        shutil.copy2(source / name, args.copy / name)
    print(f"copied {len(available)} file(s) to {args.copy}")
    print(f"next: insight-agent run-all {args.copy / 'sample_corpus.jsonl'} -o out")
    return EXIT_OK


ADAPTER_TEMPLATE = '''"""Adapter: {name} -> Insight Agent canonical JSONL (insight-trace/v1).

Verify loop — run these after every change, never batch them:

    python {path} > traces.jsonl
    insight-agent validate traces.jsonl     # must exit 0
    insight-agent coverage traces.jsonl     # what can actually fire?
    insight-agent run-ia3 traces.jsonl -o out

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
        print(f"usage: python {{argv[0]}} SOURCE", file=sys.stderr)
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

    path.write_text(ADAPTER_TEMPLATE.format(name=args.name, path=path), encoding="utf-8")
    print(f"wrote {path}\n")
    print("The verify loop — run every iteration, never batch it:")
    print("  1. insight-agent schema                  # read the contract")
    print("  2. inspect 2-3 raw source records; find where the call->result link lives")
    print(f"  3. python {path} SOURCE > traces.jsonl   # required fields only, first")
    print("  4. insight-agent validate traces.jsonl   # until it exits 0")
    print("  5. insight-agent coverage traces.jsonl   # see which rules abstain")
    print("  6. add ONE optional field, then repeat 3-5")
    print("  7. insight-agent run-ia3 traces.jsonl -o out, then check 3 findings by hand")
    return EXIT_OK


def cmd_demo(args) -> int:
    """End-to-end on the bundled data: the 'does this work at all' command."""
    source = _bundled_data_dir()
    corpus = source / "sample_corpus.jsonl"

    print(f"Using the bundled sample corpus: {corpus}\n")

    from .validate import validate_corpus

    report = validate_corpus(corpus)
    print(f"validate: {report.record_count} records, {len(report.errors)} error(s), "
          f"{len(report.warnings)} warning(s)")
    if not report.ok:
        return EXIT_SCHEMA

    from .coverage import corpus_coverage

    args.traces = corpus
    args.tool_catalog = None
    args.profile = None
    args.allow_metric_shadowing = False

    cov = corpus_coverage(_load(args))
    print(f"coverage: {cov['rules']['evaluable']}/{cov['rules']['total']} IA3 rules evaluable\n")

    return cmd_run_all(args)


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
    _add_corpus_arguments(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="list every rule")
    p.add_argument("--with-findings", action="store_true", help="also run IA3 and mark what fired")
    p.set_defaults(func=cmd_coverage)

    # explain-failures
    p = sub.add_parser(
        "explain-failures", help="compare IA2's and IA3's failure decoding call by call"
    )
    _add_corpus_arguments(p)
    p.add_argument("--only-disagreements", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_explain_failures)

    # run-ia2
    p = sub.add_parser("run-ia2", help="anomalies, recurring patterns, and the Analyst digest")
    _add_corpus_arguments(p)
    _add_output_arguments(p)
    p.add_argument("--contamination", type=float, default=0.02)
    p.add_argument("--input-scaling", choices=("none", "robust"), default="none")
    p.add_argument("--cluster-candidates", default="2,3,4,5,6,7,8")
    p.add_argument("--min-independent-traces", type=int, default=3)
    p.add_argument("--feature", action="append", default=None,
                   help="override the feature list (repeatable)")
    p.set_defaults(func=cmd_run_ia2)

    # run-ia3
    p = sub.add_parser("run-ia3", help="deterministic tool-issue detection and evidence cards")
    _add_corpus_arguments(p)
    _add_output_arguments(p)
    p.add_argument("--min-independent-cases", type=int, default=3)
    p.add_argument("--retry-threshold", type=int, default=3)
    p.add_argument("--all-cards", action="store_true")
    p.set_defaults(func=cmd_run_ia3)

    # run-all
    p = sub.add_parser("run-all", help="validate, then run IA2, IA3 and the Analyst")
    _add_corpus_arguments(p)
    _add_output_arguments(p)
    p.add_argument("--contamination", type=float, default=0.02)
    p.add_argument("--input-scaling", choices=("none", "robust"), default="none")
    p.add_argument("--cluster-candidates", default="2,3,4,5,6,7,8")
    p.add_argument("--min-independent-traces", type=int, default=3)
    p.add_argument("--feature", action="append", default=None)
    p.add_argument("--min-independent-cases", type=int, default=3)
    p.add_argument("--retry-threshold", type=int, default=3)
    p.add_argument("--all-cards", action="store_true")
    p.add_argument(
        "--no-analyst",
        action="store_true",
        help="skip the Analyst and stop at IA2/IA3 evidence (no API key needed)",
    )
    p.add_argument("--agent", default=None,
                   help="name of the agent under test (defaults to the corpus filename)")
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
    _add_corpus_arguments(p)
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
    p.add_argument("--env-file", type=Path, default=None, help="path to a .env (default: autodetect)")
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
    p.add_argument(
        "--all-cards",
        action="store_true",
        help="also show audit-only cards that did not clear the recurrence gate",
    )
    p.add_argument("--digest", type=Path, default=None, help="use an existing digest.md")
    p.add_argument("--cards", type=Path, default=None, help="use an existing cards.json")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble and write the prompt, make no API call",
    )
    p.add_argument("--contamination", type=float, default=0.02)
    p.add_argument("--min-independent-traces", type=int, default=3)
    p.add_argument("--min-independent-cases", type=int, default=3)
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

    # sample
    p = sub.add_parser("sample", help="list or copy the bundled sample data")
    p.add_argument("--list", action="store_true")
    p.add_argument("--copy", type=Path, default=None)
    p.set_defaults(func=cmd_sample)

    # init-adapter
    p = sub.add_parser("init-adapter", help="scaffold a new adapter and print the verify loop")
    p.add_argument("name")
    p.add_argument("--dir", type=Path, default=Path("adapters"))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init_adapter)

    # demo
    p = sub.add_parser("demo", help="run everything on the bundled sample data")
    _add_output_arguments(p)
    p.add_argument("--contamination", type=float, default=0.02)
    p.add_argument("--input-scaling", choices=("none", "robust"), default="none")
    p.add_argument("--cluster-candidates", default="2,3,4,5,6,7,8")
    p.add_argument("--min-independent-traces", type=int, default=3)
    p.add_argument("--feature", action="append", default=None)
    p.add_argument("--min-independent-cases", type=int, default=3)
    p.add_argument("--retry-threshold", type=int, default=3)
    p.add_argument("--all-cards", action="store_true")
    p.add_argument(
        "--no-analyst",
        action="store_true",
        help="skip the Analyst and stop at IA2/IA3 evidence (no API key needed)",
    )
    p.add_argument("--agent", default=None,
                   help="name of the agent under test (defaults to the corpus filename)")
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
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        from .loader import CorpusError

        if isinstance(exc, CorpusError):
            print(f"error: {exc}", file=sys.stderr)
            for diagnostic in exc.report.errors[:20]:
                print(f"  {diagnostic.format()}", file=sys.stderr)
            return EXIT_SCHEMA
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
