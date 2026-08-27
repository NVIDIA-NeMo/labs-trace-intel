"""The committed sample data must always match its generator.

The corpus is generated rather than hand-written because hitting all nineteen
finding types by hand is where mistakes hide. Committing the output keeps the
package usable without running a build step; regenerating here keeps the two
from drifting apart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "src" / "insight_agent" / "data"
TOOLS_DIR = REPO_ROOT / "tools"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

GENERATED_FILES = ("sample_corpus.jsonl", "sample_tool_catalog.json", "venue_profile_example.json")


@pytest.mark.parametrize("filename", GENERATED_FILES)
def test_committed_sample_data_matches_the_generator(tmp_path, filename):
    import make_sample_corpus

    make_sample_corpus.write_outputs(tmp_path)
    regenerated = (tmp_path / filename).read_bytes()
    committed = (DATA_DIR / filename).read_bytes()

    assert regenerated == committed, (
        f"{filename} differs from what tools/make_sample_corpus.py produces. "
        "Re-run `python tools/make_sample_corpus.py` and commit the result."
    )


def test_sample_corpus_validates_in_strict_mode():
    from insight_agent.trace_loaders import validate_corpus

    report = validate_corpus(DATA_DIR / "sample_corpus.jsonl")
    assert report.ok, [d.format() for d in report.errors]


def test_sample_corpus_warnings_are_deliberate():
    """The sample plants exactly one lintable defect, to demonstrate the lint."""
    from insight_agent.trace_loaders import validate_corpus

    report = validate_corpus(DATA_DIR / "sample_corpus.jsonl")
    assert [d.code for d in report.warnings] == ["duplicate_call_id"]


def test_sample_corpus_shape():
    records = [
        json.loads(line)
        for line in (DATA_DIR / "sample_corpus.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) >= 12, "clustering and recurrence need a corpus of real size"

    trace_ids = {r["trace_id"] for r in records}
    case_ids = {r["logical_case_id"] for r in records}
    assert len(trace_ids) == len(records), "trace ids must be unique"
    # Several traces must share a logical case, otherwise the sample cannot
    # demonstrate that card eligibility counts cases and not traces.
    assert len(case_ids) < len(trace_ids)


def test_venue_profile_example_loads():
    from insight_agent.evidence_streams.venue import load_profile

    profile = load_profile(DATA_DIR / "venue_profile_example.json")
    assert profile.name == "docops-renamed"
    assert "PythonSandbox" in profile.code_execution_tools


def test_standalone_catalog_matches_the_embedded_one():
    catalog = json.loads((DATA_DIR / "sample_tool_catalog.json").read_text(encoding="utf-8"))
    first = json.loads(
        (DATA_DIR / "sample_corpus.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first["tool_catalog"] == catalog
