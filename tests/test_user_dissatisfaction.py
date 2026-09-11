# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nooa.unifiedllm import UnifiedLLM

from insight_agent import complaints
from insight_agent.cli.main import _run_evidence_streams, get_config
from insight_agent.evidence_streams import user_dissatisfaction as stream
from insight_agent.evidence_streams.builtins import registered_builtin_streams
from insight_agent.evidence_streams.evidence_streams import Problem
from insight_agent.traces import Span, SpanKind, Trace, TraceSnapshot


def snapshot(*ids):
    return TraceSnapshot(
        Trace(
            id=trace_id,
            root_spans=[Span(id="span", kind=SpanKind.LLM, output="Assistant text is not input")],
            aggregate={},
        )
        for trace_id in ids
    )


class Classifier(complaints.ComplaintClassifier):
    def __init__(self, token_counts, labels=None):
        super().__init__()
        self.token_counts = token_counts
        self.labels = labels or {}
        self.inputs = []

    def token_count(self, text):
        return self.token_counts[text]

    def score(self, text):
        if self.token_counts[text] > 8192:
            raise ValueError("encoder error")
        self.inputs.append(text)
        return 0.9 if self.labels.get(text, "complaint") == "complaint" else 0.1


def test_screening_preserves_turns_and_ignores_encoder_errors():
    messages = {
        "boundary": ["Please use a table.", "Please use a table."],
        "overflow": ["long conversation", "You ignored my request again."],
        "missing": [],
    }
    classifier = Classifier(
        {
            "Please use a table.": 8192,
            "long conversation": 8193,
            "You ignored my request again.": 20,
        }
    )

    results = stream.screen_user_messages(messages, classifier)

    assert classifier.inputs == [
        "Please use a table.",
        "Please use a table.",
        "You ignored my request again.",
    ]
    assert results["boundary"].flagged
    assert [r.token_count for r in results["boundary"].message_results] == [8192, 8192]
    assert results["boundary"].message_results[0].scores["complaint"] == 0.9
    assert results["overflow"].message_results[0].scores == {}
    assert not results["missing"].flagged
    assert not results["missing"].message_results


@pytest.mark.parametrize("messages", [["complaint", "thanks"], ["thanks", "complaint"]])
def test_any_complaint_flags_trace_regardless_of_other_turns(messages):
    classifier = Classifier(
        {"complaint": 5000, "thanks": 5000, "oversized": 8193},
        {"complaint": "complaint", "thanks": "no_complaint"},
    )
    results = stream.screen_user_messages(
        {"trace": messages, "neutral": ["thanks", "thanks"], "unscored": ["oversized", "thanks"]},
        classifier,
    )
    assert results["trace"].flagged
    assert results["trace"].model_dump()["flagged"] is True
    assert not results["neutral"].flagged
    assert not results["unscored"].flagged
    assert classifier.inputs == messages + ["thanks", "thanks", "thanks"]


def test_registered_stream_passes_candidate_context_and_retains_artifacts(monkeypatch):
    config = get_config(
        [
            "--trace.filesystem.path",
            "unused.jsonl",
            "--evidence-streams.user-dissatisfaction.device",
            "cpu",
        ]
    )
    traces = snapshot("negative", "overflow", "neutral", "missing")
    messages = {
        "negative": ["Please use a table.", "Please use a table."],
        "overflow": ["Very long user text", "You ignored my request again."],
        "neutral": ["Thank you!"],
        "missing": [],
    }
    ratings = {
        trace_id: stream.TraceScreeningResult(
            message_results=[stream.ScreeningResult.model_validate({"label": label})]
        )
        for trace_id, label in zip(
            traces.traces_by_id,
            ["complaint", "complaint", "no_complaint", "no_user_messages"],
            strict=True,
        )
    }
    problem = Problem(
        description="Ignored table request", supporting_trace_ids=("negative", "overflow")
    )
    llm = MagicMock(spec=UnifiedLLM)

    async def extract(received):
        assert received is traces
        return stream.UserMessageExtraction(extract=lambda snapshot: messages)

    async def detect(received, description, **extra):
        assert set(received.traces_by_id) == {"negative", "overflow"}
        assert received.get_trace_by_id("overflow") == traces.get_trace_by_id("overflow")
        assert description == stream.USER_DISSATISFACTION
        assert extra["user_messages"] == {key: messages[key] for key in ("negative", "overflow")}
        return [problem]

    monkeypatch.setattr(complaints, "find_spec", lambda name: True)
    monkeypatch.setattr(
        stream,
        "UserMessageExtractor",
        lambda **kwargs: SimpleNamespace(build_extractor=extract),
    )
    monkeypatch.setattr(
        stream, "IssueDetector", lambda **kwargs: SimpleNamespace(detect_issues=detect)
    )
    monkeypatch.setattr(
        stream.UserDissatisfactionEvidenceStream,
        "classifier",
        SimpleNamespace(projection=SimpleNamespace(metadata={"model": "test"})),
    )
    monkeypatch.setattr(stream, "screen_user_messages", lambda messages, classifier: ratings)

    (result,) = asyncio.run(_run_evidence_streams(config.evidence_streams, traces, lambda: llm))

    assert result.problems == (problem,)
    assert result.artifacts["user_messages"] == messages
    assert result.artifacts["screening"]["negative"]["flagged"]
    assert result.artifacts["coverage"] == {
        "total_traces": 4,
        "traces_with_user_messages": 3,
        "candidate_traces": 2,
    }


@pytest.mark.parametrize("messages", [{}, {"missing": []}, {"actual": [], "extra": []}])
def test_extraction_requires_exact_trace_coverage(monkeypatch, messages):
    async def extract(snapshot):
        return stream.UserMessageExtraction(extract=lambda snapshot: messages)

    monkeypatch.setattr(
        stream,
        "UserMessageExtractor",
        lambda **kwargs: SimpleNamespace(build_extractor=extract),
    )
    detector = stream.UserDissatisfactionEvidenceStream(
        stream.UserDissatisfactionConfig(), MagicMock(spec=UnifiedLLM)
    )
    with pytest.raises(ValueError, match="exactly the supplied trace IDs"):
        detector.analyze(snapshot("actual"))


def test_missing_messages_do_not_load_models_or_invoke_issue_detector(monkeypatch):
    async def extract(snapshot):
        return stream.UserMessageExtraction(extract=lambda snapshot: {"missing": []})

    monkeypatch.setattr(
        stream,
        "UserMessageExtractor",
        lambda **kwargs: SimpleNamespace(build_extractor=extract),
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Missing user messages should not invoke classification or issue detection")

    monkeypatch.setattr(
        stream.UserDissatisfactionEvidenceStream, "classifier", property(unexpected)
    )
    monkeypatch.setattr(stream, "IssueDetector", unexpected)
    detector = stream.UserDissatisfactionEvidenceStream(
        stream.UserDissatisfactionConfig(), MagicMock(spec=UnifiedLLM)
    )

    result = detector.analyze(snapshot("missing"))

    assert not result.problems
    assert result.artifacts["coverage"]["traces_with_user_messages"] == 0
    assert detector.analyze(TraceSnapshot([])).artifacts["coverage"]["total_traces"] == 0


def test_configuration_reports_missing_optional_dependencies_and_llm(monkeypatch):
    config = stream.UserDissatisfactionConfig()
    with pytest.raises(ValueError, match="requires an LLM client"):
        registered_builtin_streams(user_dissatisfaction=config)
    monkeypatch.setattr(complaints, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="uv sync --extra dissatisfaction"):
        registered_builtin_streams(
            user_dissatisfaction=config, llm_factory=lambda: MagicMock(spec=UnifiedLLM)
        )


def test_portable_projection_rejects_incompatible_vectors():
    import numpy as np

    projection = complaints.ComplaintProjection()
    with pytest.raises(ValueError, match="4096-dimensional"):
        projection.compress(np.ones(256))
    with pytest.raises(ValueError, match="L2-normalized"):
        projection.compress(np.zeros(4096))
    with pytest.raises(ValueError, match="132-byte"):
        projection.score(bytes(128))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        projection.score(bytes(128) + np.asarray(float("nan"), dtype="<f4").tobytes())


def test_empty_or_oversized_messages_never_load_encoder():
    classifier = Classifier({"oversized": 8193})
    assert classifier.classify("  ").label == "no_user_messages"
    assert classifier.classify("oversized").label == "no_complaint"
    assert not classifier.inputs
    assert "model" not in classifier.__dict__


def test_classification_truncates_before_encoding():
    prefix = "x" * 20000
    classifier = Classifier({prefix: 5000})
    assert classifier.classify(prefix + "ignored").label == "complaint"
    assert classifier.inputs == [prefix]
