# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from nooa.unifiedllm import FakeLLMClient
from pydantic import ValidationError
from rich.console import Console
from trace_ingest.source_links import http_source_url

from insight_agent.cli.output import RunOutput, RunResult
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult, Problem
from insight_agent.insight import Insight, load_insights, resolve_trace_links
from insight_agent.trace_loaders.fs import FSDataLoader
from insight_agent.traces import Trace, TraceSnapshot


def trace(id, url=None):
    return Trace(id=id, root_spans=[], aggregate={}, source_url=url)


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "https://user:secret@host/trace",
        "https://host/\x1b]8;;evil",
        "file://remote-host/trace",
        "relative/path",
    ],
)
def test_trace_links_reject_unusable_urls(url):
    with pytest.raises(ValidationError):
        trace("a", url)
    assert http_source_url(url) is None


def test_jsonl_links_use_source_file_and_count_physical_lines(tmp_path):
    path = tmp_path / "traces #1.jsonl"
    provider_url = "https://provider.test/traces/a"
    path.write_text(
        "\n" + trace("a", provider_url).model_dump_json() + "\n\n" + trace("b").model_dump_json()
    )
    snapshot = FSDataLoader(path).load()
    assert snapshot.get_trace_by_id("a").source_url == path.as_uri() + "#L2"
    assert snapshot.get_trace_by_id("b").source_url == path.as_uri() + "#L4"


def test_links_are_resolved_from_sources_and_existing_artifacts_only(tmp_path):
    old = Insight(
        name="Issue",
        description="Details",
        trace_refs=["old", "current"],
        trace_links={"old": "https://old.test/trace", "current": "https://stale.test/trace"},
    )
    generated = old.model_copy(
        update={
            "trace_refs": ["old", "current", "missing", "no-link"],
            "trace_links": {"missing": "https://invented.test/trace"},
        }
    )
    snapshot = TraceSnapshot([trace("current", "https://current.test/trace"), trace("no-link")])
    [resolved] = resolve_trace_links([generated], snapshot, [old])
    assert resolved.trace_refs == generated.trace_refs
    assert resolved.trace_links == {
        "old": "https://old.test/trace",
        "current": "https://current.test/trace",
    }
    path = tmp_path / "insights.yml"
    path.write_text(yaml.safe_dump([resolved.model_dump()]))
    assert load_insights(path) == [resolved]
    path.write_text(
        yaml.safe_dump([{"name": "Legacy", "description": "Details", "trace_refs": ["a", "b"]}])
    )
    assert "trace_links" not in load_insights(path)[0].model_dump()


@pytest.mark.parametrize("terminal", [False, True])
def test_terminal_links_and_plain_logs(terminal, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = StringIO()
    insight = Insight(
        name="Issue [literal]",
        description="Details",
        trace_refs=["a", "b"],
        trace_links={"a": "https://provider.test/trace/a"},
    )
    RunOutput(Console(file=stream, force_terminal=terminal, width=120)).report(
        RunResult(2, [], [insight]), Path("-")
    )
    rendered = stream.getvalue()
    assert "https://provider.test/trace/a" in rendered
    assert "Issue [literal]" in rendered and "  b" in rendered
    assert ("\x1b]8;" in rendered) == terminal
    if not terminal:
        assert "\x1b" not in rendered


def test_cli_saves_and_prints_loader_links_after_compilation(
    tmp_path, monkeypatch, capsys, select_streams
):
    import insight_agent.cli.main as cli

    path = tmp_path / "traces.jsonl"
    path.write_text(trace("a").model_dump_json() + "\n" + trace("b").model_dump_json())
    insight = Insight(
        name="Repeated failure",
        description="Details",
        trace_refs=["a", "b"],
        trace_links={"a": "https://invented.test/trace"},
    )
    monkeypatch.setattr(cli, "_check_environment", lambda config: "test-key")
    monkeypatch.setattr(cli, "_build_llm", lambda *args: FakeLLMClient())
    monkeypatch.setattr(
        cli,
        "_run_evidence_streams",
        AsyncMock(
            return_value=[
                EvidenceStreamResult(
                    stream_name="tool-issues",
                    problems=(Problem(description="Failure", supporting_trace_ids=("a", "b")),),
                )
            ]
        ),
    )
    monkeypatch.setattr(
        cli,
        "InsightCompilation",
        lambda **kwargs: SimpleNamespace(compile_insights=AsyncMock(return_value=[insight])),
    )
    output = tmp_path / "insights.yml"
    assert (
        cli.main(
            [
                "--trace.filesystem.path",
                str(path),
                "--output-path",
                str(output),
                "--evidence-streams",
                json.dumps(select_streams(tool_issues={})),
            ]
        )
        == cli.EXIT_OK
    )
    captured = capsys.readouterr()
    [saved] = load_insights(output)
    assert saved.trace_links == {"a": path.as_uri() + "#L1", "b": path.as_uri() + "#L2"}
    assert yaml.safe_load(captured.out) == [saved.model_dump()]
    assert "invented.test" not in captured.out + captured.err
    assert saved.trace_links["a"] in captured.err
