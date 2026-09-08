"""End-to-end CLI behaviour, including exit codes and abstention reporting."""

from __future__ import annotations

import importlib
import json
import warnings
from pathlib import Path

import pytest

from insight_agent.cli import EXIT_ERROR, EXIT_OK, EXIT_SCHEMA, build_parser, main
from insight_agent.evidence_streams.tool_issues import FINDING_TYPES
from insight_agent.trace_loaders import (
    FSDataLoader,
    MLflowFileTraceConfig,
    MLflowTraceConfig,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "insight_agent" / "data"
TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS = DATA_DIR / "sample_corpus.jsonl"


@pytest.fixture
def corpus_without(tmp_path):
    """A copy of the sample corpus with chosen trace attributes removed."""

    def _make(*fields: str) -> Path:
        path = tmp_path / "stripped.jsonl"
        lines = []
        for line in CORPUS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for field in fields:
                record["attributes"].pop(field, None)
            lines.append(json.dumps(record))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    return _make


# -- basics ----------------------------------------------------------------


def test_schema_command_emits_the_canonical_schema(capsys):
    assert main(["schema"]) == EXIT_OK
    schema = json.loads(capsys.readouterr().out)
    assert schema["title"] == "Trace"
    assert set(schema["required"]) == {"id", "root_spans", "aggregate"}


def test_validate_exits_zero_on_the_sample_corpus(capsys):
    assert main(["validate", str(CORPUS)]) == EXIT_OK
    assert "OK" in capsys.readouterr().out


def test_validate_exits_two_on_a_corrupt_corpus(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "incomplete"}\n{not json\n', encoding="utf-8")
    assert main(["validate", str(bad)]) == EXIT_SCHEMA
    out = capsys.readouterr().out
    assert "invalid trace" in out


def test_validate_json_output_is_machine_readable(capsys):
    assert main(["validate", str(CORPUS), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["trace_count"] == 18


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
    assert run["corpus"]["trace_count"] == 18


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
    assert any(c["eligible_for_insight_compilation"] for c in cards)
    # Eligibility must track the independent-case count, not the finding count.
    for card in cards:
        assert card["eligible_for_insight_compilation"] == (card["independent_case_count"] >= 3)


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
    assert main(["run-all", str(CORPUS), "-o", str(out), "--quiet", "--no-insights"]) == EXIT_OK
    index = (out / "index.md").read_text(encoding="utf-8")
    assert "An anomaly is not an error" in index
    for expected in ("ia2/digest.md", "ia3/cards.md"):
        assert expected in index


def test_run_all_refuses_an_invalid_corpus(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "t"}\n', encoding="utf-8")
    assert main(["run-all", str(bad), "--no-insights", "-o", str(tmp_path / "out")]) == EXIT_SCHEMA


def test_run_all_help_cites_the_mlflow_search_page_limit(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["run-all", "--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "500-trace pages" in help_text
    compact_help = "".join(help_text.split())
    assert (
        "https://mlflow.org/docs/latest/api_reference/rest-api.html#searchtracesv3" in compact_help
    )


def test_run_all_accepts_mlflow_as_an_alternative_to_the_positional_file(monkeypatch, tmp_path):
    cli_main = importlib.import_module("insight_agent.cli.main")
    seen = {}

    class FakeMLflowTraceLoader:
        def __init__(self, config):
            seen["config"] = config
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.delegate = FSDataLoader(CORPUS)

        def load(self):
            return self.delegate.load()

        def describe(self):
            return {
                **self.delegate.describe(),
                "source": "mlflow://http://mlflow.example/experiments/customer-support",
            }

    monkeypatch.setattr(cli_main, "MLflowTraceLoader", FakeMLflowTraceLoader)

    out = tmp_path / "out"
    assert (
        main(
            [
                "run-all",
                "--mlflow-experiment",
                "customer-support",
                "--mlflow-tracking-uri",
                "http://mlflow.example",
                "--mlflow-filter",
                "trace.status = 'ERROR'",
                "--max-traces",
                "25",
                "--min-independent-cases",
                "100",
                "--no-insights",
                "-o",
                str(out),
            ]
        )
        == EXIT_OK
    )
    assert seen == {
        "config": MLflowTraceConfig(
            experiment_name="customer-support",
            tracking_uri="http://mlflow.example",
            filter_string="trace.status = 'ERROR'",
            max_traces=25,
        )
    }
    assert "mlflow://http://mlflow.example" in (out / "index.md").read_text(encoding="utf-8")


def test_run_all_accepts_a_native_mlflow_export_without_conversion(monkeypatch, tmp_path):
    cli_main = importlib.import_module("insight_agent.cli.main")
    export = tmp_path / "mlflow-traces.json"
    export.write_text("{}", encoding="utf-8")
    seen = {}

    class FakeMLflowFileTraceLoader:
        def __init__(self, config):
            seen["config"] = config
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.delegate = FSDataLoader(CORPUS)

        def load(self):
            return self.delegate.load()

        def describe(self):
            return {
                **self.delegate.describe(),
                "source": f"mlflow-export:{export}",
            }

    monkeypatch.setattr(cli_main, "MLflowFileTraceLoader", FakeMLflowFileTraceLoader)

    out = tmp_path / "out"
    assert (
        main(
            [
                "run-all",
                "--mlflow-export",
                str(export),
                "--max-traces",
                "25",
                "--no-insights",
                "--quiet",
                "-o",
                str(out),
            ]
        )
        == EXIT_OK
    )
    assert seen == {"config": MLflowFileTraceConfig(path=export, max_traces=25)}
    assert f"mlflow-export:{export}" in (out / "index.md").read_text(encoding="utf-8")


def test_file_and_mlflow_sources_are_mutually_exclusive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["run-all", str(CORPUS), "--mlflow-experiment", "customer-support", "--no-insights"]
        )


def test_live_mlflow_and_export_sources_are_mutually_exclusive(tmp_path):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run-all",
                "--mlflow-experiment",
                "customer-support",
                "--mlflow-export",
                str(tmp_path / "traces.json"),
                "--no-insights",
            ]
        )


def test_mlflow_query_options_require_the_mlflow_source(tmp_path, capsys):
    assert (
        main(
            [
                "run-all",
                str(CORPUS),
                "--mlflow-filter",
                "trace.status = 'ERROR'",
                "--no-insights",
                "-o",
                str(tmp_path / "out"),
            ]
        )
        == EXIT_ERROR
    )
    assert "--mlflow-filter require --mlflow-experiment" in capsys.readouterr().err


def test_mlflow_query_options_are_not_applied_to_an_export(tmp_path, capsys):
    export = tmp_path / "traces.json"
    export.write_text("{}", encoding="utf-8")

    assert (
        main(
            [
                "run-all",
                "--mlflow-export",
                str(export),
                "--mlflow-filter",
                "trace.status = 'ERROR'",
                "--no-insights",
                "-o",
                str(tmp_path / "out"),
            ]
        )
        == EXIT_ERROR
    )
    assert "--mlflow-filter require --mlflow-experiment" in capsys.readouterr().err


def test_missing_mlflow_extra_has_an_actionable_error(monkeypatch, tmp_path, capsys):
    cli_main = importlib.import_module("insight_agent.cli.main")

    class MissingMLflowLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            raise ModuleNotFoundError("No module named 'mlflow'", name="mlflow")

    monkeypatch.setattr(cli_main, "MLflowTraceLoader", MissingMLflowLoader)

    assert (
        main(
            [
                "run-all",
                "--mlflow-experiment",
                "customer-support",
                "--no-insights",
                "-o",
                str(tmp_path / "out"),
            ]
        )
        == EXIT_ERROR
    )
    error = capsys.readouterr().err
    assert "MLflow support is not installed" in error
    assert "uv sync --extra mlflow" in error


def test_demo_runs_end_to_end(tmp_path, capsys):
    assert main(["demo", "-o", str(tmp_path / "out"), "--quiet", "--no-insights"]) == EXIT_OK
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
                "id": "t1",
                "root_spans": [
                    {
                        "id": "c0",
                        "kind": "TOOL",
                        "tool_name": "Search",
                        "input": {},
                        "output": {"output": "Error: boom"},
                    }
                ],
                "aggregate": {},
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
