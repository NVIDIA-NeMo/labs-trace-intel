# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json as jsonlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import httpx
import pytest
from nooa.unifiedllm import UnifiedLLM

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


class Classifier:
    def __init__(self, token_counts, labels=None, finish_reason="stop", extra=None):
        self.token_counts = token_counts
        self.labels = labels or {}
        self.finish_reason = finish_reason
        self.extra = extra or {}
        self.inputs = []

    def post(self, path, *, json):
        assert [m["role"] for m in json["messages"]] == ["system", "user"]
        assert json["messages"][0]["content"] == stream.CLASSIFICATION_PROMPT
        assert path == "/v1/chat/completions"
        text = jsonlib.loads(json["messages"][1]["content"])["user_message"]
        self.inputs.append(text)
        assert json["max_tokens"] == stream.OUTPUT_TOKENS
        assert json["response_format"]["json_schema"]["schema"] == stream.RESPONSE_SCHEMA
        data = {
            "choices": [
                {
                    "finish_reason": self.finish_reason,
                    "message": {
                        "content": jsonlib.dumps(
                            {"label": self.labels.get(text, "complaint"), **self.extra}
                        ),
                    },
                }
            ],
            "usage": {"prompt_tokens": self.token_counts[text]},
        }
        return httpx.Response(
            200, json=data, request=httpx.Request("POST", "http://localhost" + path)
        )


def test_screening_preserves_turns_and_truncates_long_messages():
    budget = stream.MAX_TOKENS - stream.OUTPUT_TOKENS
    prefix = "x" * 20000
    classifier = Classifier({"repeat": budget, prefix: 5000})
    results = stream.screen_user_messages(
        {
            "boundary": ["repeat", "repeat"],
            "missing": [],
        },
        cast(httpx.Client, classifier),
    )
    assert classifier.inputs == ["repeat", "repeat"]
    assert results["boundary"].flagged
    assert [r.token_count for r in results["boundary"].message_results] == [budget, budget]
    assert (
        stream.screen_message(prefix + "ignored suffix", cast(httpx.Client, classifier)).label
        == "complaint"
    )
    assert classifier.inputs == ["repeat", "repeat", prefix]
    assert not results["missing"].message_results
    assert stream.screen_message("  ", cast(httpx.Client, classifier)).label == "no_user_messages"


@pytest.mark.parametrize("messages", [["complaint", "thanks"], ["thanks", "complaint"]])
def test_any_complaint_flags_trace_regardless_of_other_turns(messages):
    classifier = Classifier({"complaint": 5000, "thanks": 5000}, {"thanks": "no_complaint"})
    results = stream.screen_user_messages(
        {"trace": messages, "neutral": ["thanks", "thanks"]}, cast(httpx.Client, classifier)
    )
    assert results["trace"].flagged
    assert not results["neutral"].flagged
    assert classifier.inputs == messages + ["thanks", "thanks"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"labels": {"text": "uncertain"}},
        {"extra": {"reason": "An explanation"}},
        {"finish_reason": "length"},
    ],
)
def test_invalid_labels_explanations_and_incomplete_outputs_do_not_flag(kwargs):
    result = stream.screen_message("text", cast(httpx.Client, Classifier({"text": 20}, **kwargs)))
    assert result.label == "no_complaint"


@pytest.mark.parametrize("status", [400, 500])
def test_classifier_errors_do_not_abort_later_messages(status):
    def respond(request):
        text = jsonlib.loads(jsonlib.loads(request.content)["messages"][1]["content"])[
            "user_message"
        ]
        if text == "failed":
            return httpx.Response(status, json={"error": "classification failed"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '{"label":"complaint"}'}}
                ],
                "usage": {"prompt_tokens": 20},
            },
        )

    with httpx.Client(
        base_url="http://classifier", transport=httpx.MockTransport(respond)
    ) as client:
        results = stream.screen_user_messages({"trace": ["failed", "complaint"]}, client)
    assert [r.label for r in results["trace"].message_results] == ["no_complaint", "complaint"]
    assert results["trace"].flagged


def test_registered_stream_passes_candidate_context_and_retains_artifacts(monkeypatch):
    config = get_config(
        [
            "--trace.filesystem.path",
            "unused.jsonl",
            "--evidence-streams.user-dissatisfaction.model-path",
            "model.gguf",
        ]
    )
    traces = snapshot("complaint", "second-complaint", "neutral", "missing")
    messages = {
        "complaint": ["repeat", "repeat"],
        "second-complaint": ["long"],
        "neutral": ["thanks"],
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
        description="Ignored request", supporting_trace_ids=("complaint", "second-complaint")
    )

    async def extract(received):
        assert received is traces
        return stream.UserMessageExtraction(extract=lambda snapshot: messages)

    async def detect(received, description, **extra):
        assert set(received.traces_by_id) == {"complaint", "second-complaint"}
        assert received.get_trace_by_id("second-complaint") is traces.get_trace_by_id(
            "second-complaint"
        )
        assert description == stream.USER_DISSATISFACTION
        assert extra["user_messages"] == {
            key: messages[key] for key in ("complaint", "second-complaint")
        }
        return [problem]

    monkeypatch.setattr(stream, "validate_classifier_configuration", lambda config: None)
    monkeypatch.setattr(
        stream,
        "UserMessageExtractor",
        lambda **kwargs: SimpleNamespace(build_extractor=extract),
    )
    monkeypatch.setattr(
        stream, "IssueDetector", lambda **kwargs: SimpleNamespace(detect_issues=detect)
    )
    monkeypatch.setattr(stream, "load_classifier", lambda config: nullcontext(object()))
    monkeypatch.setattr(stream, "screen_user_messages", lambda messages, classifier: ratings)
    (result,) = asyncio.run(
        _run_evidence_streams(config.evidence_streams, traces, lambda: MagicMock(spec=UnifiedLLM))
    )
    assert result.problems == (problem,)
    assert result.artifacts["user_messages"] == messages
    assert result.artifacts["screening"]["complaint"]["flagged"]
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
        stream.UserDissatisfactionConfig(model_path=Path("unused")), MagicMock(spec=UnifiedLLM)
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

    monkeypatch.setattr(stream, "load_classifier", unexpected)
    monkeypatch.setattr(stream, "IssueDetector", unexpected)
    detector = stream.UserDissatisfactionEvidenceStream(
        stream.UserDissatisfactionConfig(model_path=Path("unused")), MagicMock(spec=UnifiedLLM)
    )
    assert not detector.analyze(snapshot("missing")).problems
    assert detector.analyze(TraceSnapshot([])).artifacts["coverage"]["total_traces"] == 0


def test_configuration_reports_missing_model_executable_and_llm(monkeypatch, tmp_path):
    config = stream.UserDissatisfactionConfig(model_path=tmp_path / "model.gguf")
    with pytest.raises(ValueError, match="requires an LLM client"):
        registered_builtin_streams(user_dissatisfaction=config)
    with pytest.raises(ValueError, match="GGUF model not found"):
        stream.validate_classifier_configuration(config)
    config.model_path.touch()
    monkeypatch.setattr(stream.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="executable not found"):
        stream.validate_classifier_configuration(config)
