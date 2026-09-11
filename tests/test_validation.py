# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from insight_agent.evidence_streams.evidence_streams import Problem
from insight_agent.insights_generation import validation
from insight_agent.insights_generation.validation import ProblemValidation
from insight_agent.traces import Trace, TraceAggregate


def _response(content: str, tool_calls: list[ToolCall]) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        assistant_message={"role": "assistant", "content": content},
        reasoning=None,
        usage=None,
    )


def test_problem_validation_uses_code_before_accepting_problem(tmp_path) -> None:
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 1\n", encoding="utf-8")
    tool_call = ToolCall(
        id="read-1",
        name="read_file",
        arguments=json.dumps({"path": "agent.py"}),
    )
    llm = FakeLLMClient(
        scripted_responses=[
            _response("", [tool_call]),
            _response('{"supported": true}', []),
        ]
    )
    validator = ProblemValidation(tmp_path, llm)
    problem = Problem(description="Requests time out too quickly", supporting_trace_ids=("t1",))
    trace = Trace(id="t1", root_spans=[], aggregate=TraceAggregate())

    supported = asyncio.run(validator.is_supported(problem, (trace,)))

    assert supported is True
    assert llm.call_count == 2


def test_problem_validation_cannot_decide_without_reading_code(tmp_path) -> None:
    llm = FakeLLMClient(scripted_responses=[_response('{"supported": true}', [])])
    validator = ProblemValidation(tmp_path, llm)
    problem = Problem(description="Requests time out too quickly", supporting_trace_ids=("t1",))
    trace = Trace(id="t1", root_spans=[], aggregate=TraceAggregate())

    supported = asyncio.run(validator.is_supported(problem, (trace,)))

    assert supported is None


def test_problem_validation_can_return_not_applicable(tmp_path) -> None:
    llm = FakeLLMClient(scripted_responses=[_response('{"supported": null}', [])])
    validator = ProblemValidation(tmp_path, llm)
    problem = Problem(
        description="Deployment credentials are missing",
        supporting_trace_ids=("t1",),
    )
    trace = Trace(id="t1", root_spans=[], aggregate=TraceAggregate())

    supported = asyncio.run(validator.is_supported(problem, (trace,)))

    assert supported is None


def test_problem_validation_forces_decision_after_tool_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(validation, "_MAX_TOOL_ROUNDS", 1)
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 1\n", encoding="utf-8")
    tool_call = ToolCall(
        id="read-1",
        name="read_file",
        arguments=json.dumps({"path": "agent.py"}),
    )
    llm = FakeLLMClient(
        scripted_responses=[
            _response("", [tool_call]),
            _response('{"supported": false}', []),
        ]
    )
    validator = ProblemValidation(tmp_path, llm)
    problem = Problem(description="Requests time out too quickly", supporting_trace_ids=("t1",))
    trace = Trace(id="t1", root_spans=[], aggregate=TraceAggregate())

    supported = asyncio.run(validator.is_supported(problem, (trace,)))

    assert supported is False
    assert llm.call_count == 2
