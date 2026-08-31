"""End-to-end CLI behaviour, including exit codes and abstention reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from insight_agent.cli import EXIT_OK, EXIT_SCHEMA, main
from insight_agent.evidence_streams.tool_issues import FINDING_TYPES

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "insight_agent" / "data"
TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS = DATA_DIR / "sample_corpus.jsonl"


@pytest.fixture
def corpus_without(tmp_path):
    """A copy of the sample corpus with chosen top-level fields removed."""

    def _make(*fields: str) -> Path:
        path = tmp_path / "stripped.jsonl"
        lines = []
        for line in CORPUS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for field in fields:
                record.pop(field, None)
            lines.append(json.dumps(record))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    return _make


# -- basics ----------------------------------------------------------------


def test_schema_command_emits_the_canonical_schema(capsys):
    assert main(["schema"]) == EXIT_OK
    schema = json.loads(capsys.readouterr().out)
    assert schema["$id"].endswith("insight-trace/v1")
    assert set(schema["required"]) == {"schema_version", "trace_id", "calls"}


def test_validate_exits_zero_on_the_sample_corpus(capsys):
    assert main(["validate", str(CORPUS)]) == EXIT_OK
    assert "OK" in capsys.readouterr().out


def test_validate_exits_two_on_a_corrupt_corpus(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"schema_version": "insight-trace/v1"}\n{not json\n', encoding="utf-8")
    assert main(["validate", str(bad)]) == EXIT_SCHEMA
    out = capsys.readouterr().out
    assert "invalid_json" in out or "ERROR" in out


def test_validate_json_output_is_machine_readable(capsys):
    assert main(["validate", str(CORPUS), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["record_count"] == 18


# -- coverage --------------------------------------------------------------


def test_coverage_reports_full_capability_on_the_sample(capsys):
    assert main(["coverage", str(CORPUS), "--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["rules"]["evaluable"] == len(FINDING_TYPES) == 19
    assert report["rules"]["abstaining"] == []


def test_coverage_names_the_rules_that_abstain_without_a_catalog(corpus_without, capsys):
    assert main(["coverage", str(corpus_without("tool_catalog")), "--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)

    abstaining = set(report["rules"]["abstaining"])
    assert abstaining == {
        "unknown_tool",
        "missing_required_argument",
        "unknown_argument",
        "argument_type_mismatch",
        "argument_enum_violation",
        "json_schema_violation",
    }
    # malformed_tool_call still works without a catalog.
    assert "malformed_tool_call" not in abstaining


def test_coverage_human_output_explains_why(corpus_without, capsys):
    main(["coverage", str(corpus_without("tool_catalog"))])
    out = capsys.readouterr().out
    assert "13/19" in out
    assert "no tool_catalog" in out


def test_coverage_warns_when_logical_case_id_is_absent(corpus_without, capsys):
    main(["coverage", str(corpus_without("logical_case_id")), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert any("logical_case_id" in note for note in report["notes"])


# -- runs ------------------------------------------------------------------


def test_run_ia2_writes_a_digest_and_run_metadata(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["run-ia2", str(CORPUS), "-o", str(out)]) == EXIT_OK

    digest = (out / "ia2" / "digest.md").read_text(encoding="utf-8")
    assert "## Unusual traces" in digest
    assert "Recurring strict tool-failure" in digest

    run = json.loads((out / "ia2" / "run.json").read_text(encoding="utf-8"))
    # Version capture matters: sklearn minor releases change IsolationForest output.
    assert run["scikit_learn_version"]
    assert run["numpy_version"]
    assert run["corpus"]["steps_present"] is True


def test_run_ia2_digest_is_deterministic(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    main(["run-ia2", str(CORPUS), "-o", str(first)])
    main(["run-ia2", str(CORPUS), "-o", str(second)])
    assert (first / "ia2" / "digest.md").read_bytes() == (second / "ia2" / "digest.md").read_bytes()


def test_run_ia2_to_stdout(capsys):
    assert main(["run-ia2", str(CORPUS), "-o", "-"]) == EXIT_OK
    assert "IA2 cited evidence digest" in capsys.readouterr().out


def test_run_ia3_fires_all_nineteen_types_and_promotes_cards(tmp_path):
    out = tmp_path / "out"
    assert main(["run-ia3", str(CORPUS), "-o", str(out)]) == EXIT_OK

    findings = json.loads((out / "ia3" / "findings.json").read_text(encoding="utf-8"))
    cards = json.loads((out / "ia3" / "cards.json").read_text(encoding="utf-8"))

    assert {f["issue_type"] for f in findings} == set(FINDING_TYPES)
    assert any(c["eligible_for_analyst"] for c in cards)
    # Eligibility must track the independent-case count, not the finding count.
    for card in cards:
        assert card["eligible_for_analyst"] == (card["independent_case_count"] >= 3)


def test_run_ia3_cards_markdown_cites_evidence(tmp_path):
    out = tmp_path / "out"
    main(["run-ia3", str(CORPUS), "-o", str(out)])
    text = (out / "ia3" / "cards.md").read_text(encoding="utf-8")

    assert "ELIGIBLE" in text
    assert "A card is not an Insight" in text
    assert "source:" in text  # representative evidence carries pointers


def test_all_cards_widens_the_rendering_but_not_the_json(tmp_path):
    """cards.json is always complete; --all-cards only affects cards.md."""
    default, everything = tmp_path / "a", tmp_path / "b"
    main(["run-ia3", str(CORPUS), "-o", str(default), "--quiet"])
    main(["run-ia3", str(CORPUS), "-o", str(everything), "--all-cards", "--quiet"])

    def sections(path):
        return [
            line
            for line in (path / "ia3" / "cards.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("## ")
        ]

    assert len(sections(default)) < len(sections(everything))
    assert json.loads((default / "ia3" / "cards.json").read_text(encoding="utf-8")) == json.loads(
        (everything / "ia3" / "cards.json").read_text(encoding="utf-8")
    )
    assert "--all-cards" in (default / "ia3" / "cards.md").read_text(encoding="utf-8")


def test_run_ia3_degrades_without_a_catalog_instead_of_crashing(corpus_without, tmp_path):
    out = tmp_path / "out"
    assert main(["run-ia3", str(corpus_without("tool_catalog")), "-o", str(out)]) == EXIT_OK

    findings = json.loads((out / "ia3" / "findings.json").read_text(encoding="utf-8"))
    fired = {f["issue_type"] for f in findings}
    assert "unknown_tool" not in fired
    assert "explicit_tool_failure" in fired  # non-contract rules keep working


def test_run_all_writes_an_index(tmp_path):
    out = tmp_path / "out"
    assert main(["run-all", str(CORPUS), "-o", str(out), "--quiet", "--no-analyst"]) == EXIT_OK
    index = (out / "index.md").read_text(encoding="utf-8")
    assert "An anomaly is not an error" in index
    for expected in ("ia2/digest.md", "ia3/cards.md"):
        assert expected in index


def test_run_all_refuses_an_invalid_corpus(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"schema_version": "insight-trace/v1", "trace_id": "t"}\n', encoding="utf-8")
    assert main(["run-all", str(bad), "-o", str(tmp_path / "out")]) == EXIT_SCHEMA


def test_demo_runs_end_to_end(tmp_path, capsys):
    assert main(["demo", "-o", str(tmp_path / "out"), "--quiet", "--no-analyst"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "19/19 IA3 rules evaluable" in out
    assert (tmp_path / "out" / "ia2" / "digest.md").exists()


# -- explain-failures ------------------------------------------------------


def test_explain_failures_lists_every_call(capsys):
    assert main(["explain-failures", str(CORPUS), "--json"]) == EXIT_OK
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 79
    assert {"ia2_failed", "ia3_failed", "agree"} <= set(rows[0])


def test_explain_failures_surfaces_the_content_vs_output_trap(tmp_path, capsys):
    """A result under 'output' fails in IA2 and is invisible to IA3."""
    path = tmp_path / "trap.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "insight-trace/v1",
                "trace_id": "t1",
                "calls": [
                    {
                        "call_id": "c0",
                        "call_index": 0,
                        "tool_name": "Search",
                        "arguments": {},
                        "result": {"output": "Error: boom"},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["explain-failures", str(path), "--only-disagreements", "--json"]) == EXIT_OK
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["ia2_failed"] is True
    assert rows[0]["ia3_failed"] is False


# -- scaffolding -----------------------------------------------------------


def test_init_adapter_scaffolds_a_runnable_template(tmp_path, capsys):
    assert main(["init-adapter", "mytool", "--dir", str(tmp_path)]) == EXIT_OK
    adapter = tmp_path / "mytool.py"
    source = adapter.read_text(encoding="utf-8")

    compile(source, "mytool.py", "exec")  # must be valid Python
    assert source.startswith("#!/usr/bin/env -S uv run\n")
    assert adapter.stat().st_mode & 0o111
    assert "insight-agent validate" in source
    assert "missing_tool_result" in source  # the result-absence trap is documented
    output = capsys.readouterr().out
    assert "verify loop" in output.lower()
    assert f"3. {adapter} SOURCE" in output


def test_init_adapter_refuses_to_clobber(tmp_path, capsys):
    main(["init-adapter", "mytool", "--dir", str(tmp_path)])
    assert main(["init-adapter", "mytool", "--dir", str(tmp_path)]) != EXIT_OK
