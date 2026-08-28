"""Report which tool-issue rules a corpus can support.

This is the feedback loop for anyone writing an adapter. Validation answers
"is my JSON well formed"; coverage answers the far more useful question "given
what I populated, which of the nineteen IA3 rules can fire at all, and what
would I have to add to unlock the rest?"

Without it the failure mode is silent and demoralising: the adapter validates,
the run completes, no findings appear, and there is nothing to tell you that
six rules abstained because ``tool_catalog`` was missing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from insight_agent.evidence_streams.tool_issues import FINDING_TYPES
from insight_agent.trace_loaders import InsightTraceV1Loader

__all__ = ["RULE_REQUIREMENTS", "corpus_coverage", "format_coverage"]


@dataclass(frozen=True)
class RuleRequirement:
    """What a finding type needs before it can possibly fire."""

    issue_type: str
    needs: str
    gate: str
    detail: str


#: Every IA3 finding type, with the canonical field that gates it.
RULE_REQUIREMENTS: tuple[RuleRequirement, ...] = (
    RuleRequirement(
        "unknown_tool",
        "tool_catalog",
        "evidence_streams/tool_issues.py:157",
        "The catalog is what makes a tool name 'unknown'.",
    ),
    RuleRequirement(
        "missing_required_argument",
        "tool_catalog",
        "evidence_streams/tool_issues.py:167",
        "Needs the per-tool argument schema.",
    ),
    RuleRequirement(
        "unknown_argument",
        "tool_catalog",
        "evidence_streams/tool_issues.py:168",
        "Needs a schema with additionalProperties:false.",
    ),
    RuleRequirement(
        "argument_type_mismatch",
        "tool_catalog",
        "evidence_streams/tool_issues.py:169",
        "Needs typed properties in the schema.",
    ),
    RuleRequirement(
        "argument_enum_violation",
        "tool_catalog",
        "evidence_streams/tool_issues.py:170",
        "Needs an enum constraint in the schema.",
    ),
    RuleRequirement(
        "json_schema_violation",
        "tool_catalog",
        "evidence_streams/tool_issues.py:171",
        "Any schema keyword other than required/additionalProperties/type/enum.",
    ),
    RuleRequirement(
        "malformed_tool_call",
        "always",
        "evidence_streams/tool_issues.py:155",
        "Fires on non-object arguments with or without a catalog.",
    ),
    RuleRequirement(
        "duplicate_call_id",
        "always",
        "evidence_streams/tool_issues.py:299",
        "Needs call_id, which is required.",
    ),
    RuleRequirement(
        "missing_tool_result",
        "result-key discipline",
        "evidence_streams/tool_issues.py:304",
        "Fires only when the 'result' key is absent (or result_missing / "
        "result_count:0). An adapter that always emits a result, even null, "
        "silences this rule.",
    ),
    RuleRequirement(
        "duplicate_tool_result",
        "result_count",
        "evidence_streams/tool_issues.py:308",
        "Needs result_count > 1.",
    ),
    RuleRequirement(
        "call_result_id_mismatch",
        "result_id",
        "evidence_streams/tool_issues.py:319",
        "Only populate result_id from a genuine result-to-call reference.",
    ),
    RuleRequirement(
        "orphan_tool_result",
        "orphan_results",
        "evidence_streams/tool_issues.py:417",
        "Needs the adapter to capture results with no matching call.",
    ),
    RuleRequirement(
        "mapped_instrumentation_alias",
        "instrumentation_alias_of",
        "evidence_streams/tool_issues.py:330",
        "Needs the adapter to know the real tool behind an alias.",
    ),
    RuleRequirement(
        "explicit_tool_failure",
        "result",
        "evidence_streams/tool_issues.py:342",
        "Needs a decodable result. explicit_error:false disables it entirely.",
    ),
    RuleRequirement(
        "explicit_prerequisite_or_state_failure",
        "result",
        "evidence_streams/tool_issues.py:399",
        "Needs result text matching the venue's state patterns.",
    ),
    RuleRequirement(
        "unresolved_placeholder_argument",
        "result",
        "evidence_streams/tool_issues.py:366",
        "Needs a failed or rejected call whose argument is exactly a placeholder.",
    ),
    RuleRequirement(
        "explicitly_rejected_ungrounded_identifier",
        "complete_provenance_context",
        "evidence_streams/tool_issues.py:378",
        "Set true only if every user message and prior result was captured.",
    ),
    RuleRequirement(
        "repeated_identical_failed_call",
        "result",
        "evidence_streams/tool_issues.py:430",
        "Needs three failing calls with byte-identical arguments.",
    ),
    RuleRequirement(
        "modified_retry_same_failure",
        "result",
        "evidence_streams/tool_issues.py:442",
        "Needs three failing calls sharing a failure class across >=2 argument sets.",
    ),
)


def _field_presence(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count how many records/calls populate each optional field."""

    counts = {
        "tool_catalog": 0,
        "steps": 0,
        "logical_case_id": 0,
        "source_pointer": 0,
        "observed_verdict": 0,
        "cost": 0,
        "metrics": 0,
        "complete_provenance_context": 0,
        "task_text": 0,
        "orphan_results": 0,
        "result_present": 0,
        "result_absent": 0,
        # A result that is present but empty is indistinguishable from a healthy
        # corpus by rule-evaluability alone, yet nothing can ever be decoded
        # from it. Counted separately so the notes can say so.
        "result_empty": 0,
        "arguments_empty": 0,
        "result_id": 0,
        "result_count_gt_1": 0,
        "duration_ms": 0,
        "explicit_error_true": 0,
        "explicit_error_false": 0,
        "instrumentation_alias_of": 0,
        "prior_user_text": 0,
        "returned_data": 0,
        "call_source_pointer": 0,
        "total_calls": 0,
        "total_steps": 0,
    }

    for record in records:
        for key in (
            "tool_catalog",
            "steps",
            "logical_case_id",
            "source_pointer",
            "observed_verdict",
            "cost",
            "metrics",
            "task_text",
            "orphan_results",
        ):
            if record.get(key):
                counts[key] += 1
        if record.get("complete_provenance_context") is True:
            counts["complete_provenance_context"] += 1

        steps = record.get("steps") or []
        counts["total_steps"] += len(steps)

        for call in record.get("calls", []):
            counts["total_calls"] += 1
            missing = (
                "result" not in call
                or call.get("result_missing") is True
                or call.get("result_count") == 0
            )
            counts["result_absent" if missing else "result_present"] += 1
            if not missing:
                result = call.get("result")
                if result is None or result == "" or result == {} or result == []:
                    counts["result_empty"] += 1

            arguments = call.get("arguments")
            # `{}` carries nothing, and a lone key such as `_unparsed_input`
            # is an adapter's placeholder for "the source had no arguments".
            if arguments in ({}, None, ""):
                counts["arguments_empty"] += 1
            elif isinstance(arguments, Mapping) and len(arguments) == 1:
                only = next(iter(arguments))
                if str(only).startswith("_") and arguments[only] in (None, "", {}, []):
                    counts["arguments_empty"] += 1

            if call.get("result_id") is not None:
                counts["result_id"] += 1
            if isinstance(call.get("result_count"), int) and call["result_count"] > 1:
                counts["result_count_gt_1"] += 1
            if call.get("duration_ms") is not None:
                counts["duration_ms"] += 1
            if call.get("explicit_error") is True:
                counts["explicit_error_true"] += 1
            if call.get("explicit_error") is False:
                counts["explicit_error_false"] += 1
            if call.get("instrumentation_alias_of"):
                counts["instrumentation_alias_of"] += 1
            if call.get("prior_user_text"):
                counts["prior_user_text"] += 1
            if "returned_data" in call:
                counts["returned_data"] += 1
            if call.get("source_pointer"):
                counts["call_source_pointer"] += 1

    return counts


def _rule_status(
    requirement: RuleRequirement,
    presence: Mapping[str, int],
    fired: Iterable[str],
    has_corpus_catalog: bool,
) -> dict[str, Any]:
    fired = set(fired)
    evaluable = True
    reason = ""

    if requirement.needs == "tool_catalog":
        evaluable = bool(presence["tool_catalog"]) or has_corpus_catalog
        if not evaluable:
            reason = "no tool_catalog on any record and none supplied with --tool-catalog"
    elif requirement.needs == "result_id":
        evaluable = presence["result_id"] > 0
        if not evaluable:
            reason = "no call sets result_id"
    elif requirement.needs == "result_count":
        evaluable = presence["result_count_gt_1"] > 0
        if not evaluable:
            reason = "no call sets result_count > 1"
    elif requirement.needs == "orphan_results":
        evaluable = presence["orphan_results"] > 0
        if not evaluable:
            reason = "no record captures orphan_results"
    elif requirement.needs == "instrumentation_alias_of":
        evaluable = presence["instrumentation_alias_of"] > 0
        if not evaluable:
            reason = "no call sets instrumentation_alias_of"
    elif requirement.needs == "complete_provenance_context":
        evaluable = presence["complete_provenance_context"] > 0
        if not evaluable:
            reason = "no record asserts complete_provenance_context"
    elif requirement.needs == "result-key discipline":
        evaluable = presence["result_absent"] > 0
        if not evaluable:
            reason = (
                "every call carries a result key, so nothing can be reported as missing. "
                "If the source really does drop results, omit the key (or set result_missing)"
            )
    elif requirement.needs == "result":
        evaluable = presence["result_present"] > 0
        if not evaluable:
            reason = "no call carries a result to decode"

    return {
        "issue_type": requirement.issue_type,
        "evaluable": evaluable,
        "fired": requirement.issue_type in fired,
        "requires": requirement.needs,
        "reason": reason,
        "gate": requirement.gate,
        "detail": requirement.detail,
    }


def corpus_coverage(
    loader: InsightTraceV1Loader,
    *,
    findings: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the full capability report for a loaded corpus."""

    records = list(loader.records)
    presence = _field_presence(records)
    has_corpus_catalog = loader.options.tool_catalog is not None
    fired = {f["issue_type"] for f in (findings or [])}

    rules = [_rule_status(r, presence, fired, has_corpus_catalog) for r in RULE_REQUIREMENTS]
    evaluable = [r for r in rules if r["evaluable"]]

    trace_count = len(records)
    distinct_cases = len({r.get("logical_case_id") or r["trace_id"] for r in records})

    notes: list[str] = []

    if trace_count < 3:
        notes.append(
            f"Only {trace_count} trace(s): IA2 trajectory clustering needs at least three and "
            "will abstain."
        )
    if distinct_cases < 3:
        notes.append(
            f"Only {distinct_cases} distinct logical case(s): IA3 promotes a card at three "
            "independent cases, so nothing can become eligible_for_analyst."
        )
    if trace_count and distinct_cases == trace_count and presence["logical_case_id"] == 0:
        notes.append(
            "No record sets logical_case_id, so every trace counts as its own independent "
            "case. If several traces are retries of the same task, card eligibility is "
            "currently overstated."
        )
    # These two are the "corpus looks fine, detects nothing" cases. Rule
    # evaluability cannot see them: the fields are present, they are just empty.
    if presence["result_present"] and presence["result_empty"] >= presence["result_present"] * 0.95:
        notes.append(
            f"{presence['result_empty']}/{presence['result_present']} present results are empty "
            '(null, "", {} or []). Nothing can be decoded from them, so every result-based '
            "rule will stay silent no matter how many calls you feed in. This usually means "
            "the export carries span metadata but not tool payloads."
        )
    if presence["total_calls"] and presence["arguments_empty"] >= presence["total_calls"] * 0.95:
        notes.append(
            f"{presence['arguments_empty']}/{presence['total_calls']} calls have empty or "
            "placeholder arguments, so no argument-contract rule can find anything even with a "
            "tool_catalog supplied. Check that the adapter is reading the right field."
        )

    if presence["total_calls"] and presence["explicit_error_false"] > presence["total_calls"] * 0.9:
        notes.append(
            "explicit_error is false on almost every call, which disables all text-based "
            "failure decoding."
        )
    if presence["result_absent"] == 0 and presence["total_calls"]:
        notes.append(
            "Every call carries a result key. If the source genuinely loses results "
            "sometimes, missing_tool_result cannot report it."
        )
    if presence["steps"] == 0:
        notes.append("No record has steps: IA2 trajectory tokens fall back to one token per call.")
    elif presence["steps"] < trace_count:
        notes.append(
            f"Only {presence['steps']}/{trace_count} records have steps. Mixed corpora are not "
            "internally comparable, because trajectory features differ between the two halves."
        )

    return {
        "source": loader.source,
        "trace_count": trace_count,
        "call_count": presence["total_calls"],
        "step_count": presence["total_steps"],
        "distinct_logical_cases": distinct_cases,
        # Without the field the count is derived from trace_id, which looks
        # identical to a populated corpus unless we say so.
        "logical_case_id_populated": presence["logical_case_id"] > 0,
        "tool_catalog": {
            "records_with_catalog": presence["tool_catalog"],
            "corpus_wide_catalog_supplied": has_corpus_catalog,
        },
        "rules": {
            "total": len(FINDING_TYPES),
            "evaluable": len(evaluable),
            "abstaining": [r["issue_type"] for r in rules if not r["evaluable"]],
            "detail": rules,
        },
        "field_presence": presence,
        "notes": notes,
    }


def format_coverage(report: Mapping[str, Any], *, verbose: bool = False) -> str:
    """Render the coverage report for a terminal."""

    lines: list[str] = []
    lines.append(f"Corpus: {report['source']}")
    derived = "" if report.get("logical_case_id_populated", True) else " (defaulted from trace_id)"
    lines.append(
        f"  {report['trace_count']} traces, {report['call_count']} calls, "
        f"{report['step_count']} steps, "
        f"{report['distinct_logical_cases']} distinct logical cases{derived}"
    )

    rules = report["rules"]
    lines.append("")
    lines.append(f"IA3 rules evaluable: {rules['evaluable']}/{rules['total']}")

    abstaining = rules["abstaining"]
    if abstaining:
        lines.append("")
        lines.append("Abstaining, and why:")
        by_reason: dict[str, list[str]] = {}
        for rule in rules["detail"]:
            if not rule["evaluable"]:
                by_reason.setdefault(rule["reason"], []).append(rule["issue_type"])
        for reason, types in by_reason.items():
            lines.append(f"  - {reason}")
            for issue_type in sorted(types):
                lines.append(f"      {issue_type}")
    else:
        lines.append("  (every rule has the evidence it needs)")

    if verbose:
        lines.append("")
        lines.append("Per-rule detail:")
        for rule in rules["detail"]:
            mark = "ok  " if rule["evaluable"] else "SKIP"
            fired = " (fired)" if rule["fired"] else ""
            lines.append(f"  [{mark}] {rule['issue_type']}{fired} — needs {rule['requires']}")

    if report["notes"]:
        lines.append("")
        lines.append("Notes:")
        for note in report["notes"]:
            lines.append(f"  - {note}")

    return "\n".join(lines)
