# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from pydantic import ValidationError

from insight_agent.cli.main import _run_evidence_streams, get_config
from insight_agent.evidence_streams.ethos_divergence import ethos_divergence_detector as ethos
from insight_agent.evidence_streams.evidence_streams import Problem
from insight_agent.traces import TraceSnapshot


def test_ethos_cli_runs_registered_detector(tmp_path, monkeypatch):
    path = tmp_path / "ethos.md"
    path.write_text("Never issue refunds.", encoding="utf-8")
    config = get_config(
        [
            "--trace.filesystem.path",
            "unused.jsonl",
            "--evidence-streams.ethos-divergence.ethos-path",
            str(path),
        ]
    )
    snapshot = TraceSnapshot([])
    llm = object()
    problem = Problem(description="Issued a refund", supporting_trace_ids=("trace-1",))

    class FakeDetector:
        def __init__(self, *, llm):
            self.llm = llm

        async def detect_issues(self, traces, description, **extra):
            assert self.llm is llm
            assert traces is snapshot
            assert description == ethos.ETHOS_DIVERGENCE
            assert extra == {"ethos": "Never issue refunds."}
            return [problem]

    monkeypatch.setattr(ethos, "IssueDetector", FakeDetector)
    results = asyncio.run(_run_evidence_streams(config.evidence_streams, snapshot, llm))
    assert len(results) == 1
    assert results[0].stream_name == "ethos-divergence"
    assert results[0].problems == (problem,)


def test_ethos_rejects_missing_and_empty_documents(tmp_path):
    path = tmp_path / "ethos.md"
    with pytest.raises(ValidationError):
        ethos.EthosDivergenceConfig(ethos_path=path)
    path.write_text(" \n", encoding="utf-8")
    stream = ethos.EthosDivergenceEvidenceStream(
        config=ethos.EthosDivergenceConfig(ethos_path=path), llm=object()
    )
    with pytest.raises(ValueError, match="non-empty"):
        stream.validate_configuration()
