"""The committed sample data must always match its generator.

The corpus is generated rather than hand-written because hitting all nineteen
finding types by hand is where mistakes hide. Committing the output keeps the
package usable without running a build step; regenerating here keeps the two
from drifting apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

from insight_agent.evidence_streams._trace import walk_spans
from insight_agent.trace_loaders import FSDataLoader
from insight_agent.traces import SpanKind

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "src" / "insight_agent" / "data"
TOOLS_DIR = REPO_ROOT / "tools"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import generate_demo_corpus  # noqa: E402 - tools path must be registered first


def test_committed_sample_data_matches_the_generator(tmp_path):
    generate_demo_corpus.write_outputs(tmp_path)
    filename = "sample_corpus.jsonl"
    regenerated = (tmp_path / filename).read_bytes()
    committed = (DATA_DIR / filename).read_bytes()

    assert regenerated == committed, (
        f"{filename} differs from what tools/generate_demo_corpus.py produces. "
        "Re-run `./tools/generate_demo_corpus.py` and commit the result."
    )


def test_sample_corpus_validates_in_strict_mode():
    assert FSDataLoader(DATA_DIR / "sample_corpus.jsonl").load().trace_count == 18


def test_sample_corpus_duplicate_call_id_is_deliberate():
    """The sample plants duplicate source call IDs to exercise that detector."""
    snapshot = FSDataLoader(DATA_DIR / "sample_corpus.jsonl").load()
    trace = snapshot.get_trace_by_id("docops-instrumentation")
    call_ids = [
        visit.span.tool_call.call_id
        for visit in walk_spans(trace)
        if visit.span.kind is SpanKind.TOOL and visit.span.tool_call is not None
    ]
    assert call_ids.count("docops-instrumentation#dup") == 2


def test_sample_corpus_shape():
    traces = list(FSDataLoader(DATA_DIR / "sample_corpus.jsonl").load())
    assert len(traces) >= 12, "clustering and recurrence need a corpus of real size"

    trace_ids = {trace.id for trace in traces}
    case_ids = {trace.attributes["logical_case_id"] for trace in traces}
    assert len(trace_ids) == len(traces), "trace ids must be unique"
    # Several traces must share a logical case, otherwise the sample cannot
    # demonstrate that card eligibility counts cases and not traces.
    assert len(case_ids) < len(trace_ids)
