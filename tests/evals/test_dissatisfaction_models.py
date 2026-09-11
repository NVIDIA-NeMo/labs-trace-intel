# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-model regression: ten complaints, two reasons, three neutral controls.

INSIGHT_AGENT_EVAL_CLASSIFIER=1 \\
uv run --locked pytest tests/evals/test_dissatisfaction_models.py -s

Hosted analysis uses the project's inference configuration and credentials.
"""

import asyncio
import json
import os
import re
from pathlib import Path

import pytest

from insight_agent.cli.main import _build_llm
from insight_agent.config import RunConfig
from insight_agent.evidence_streams.issue_detector import IssueDetector
from insight_agent.evidence_streams.user_dissatisfaction.stream import (
    USER_DISSATISFACTION,
    UserDissatisfactionConfig,
    load_classifier,
    screen_user_messages,
)
from insight_agent.insights_generation.config import ENV_API_KEY, load_dotenv, resolve
from insight_agent.traces import Span, SpanKind, Trace, TraceSnapshot

COMPLAINTS = {
    "missing-pr": [
        "You said the task was done, but you never opened the pull request I asked for. This is not finished.",
        "I'm frustrated that you gave me a patch instead of opening the requested PR. Please finish the actual task.",
        "Again, there is no pull request URL. You keep claiming completion without publishing the PR.",
        "This is the third time I've asked you to open the pull request. Your local changes are not the deliverable I requested.",
        "You ignored the main requirement: publish a pull request. Stop calling this complete when there is no PR.",
    ],
    "wrong-format": [
        "You ignored my JSON-only instruction and returned prose again. Your response breaks my parser.",
        "I'm fed up with the markdown fences you keep adding. I explicitly asked for raw JSON, and your output isn't usable.",
        "Your answer is in the wrong format. I requested JSON only, but you added an explanation before it.",
        "Again you returned a paragraph instead of the JSON object I requested. Please follow the output format.",
        "This is frustrating: I told you the response must be valid JSON, and you sent a markdown table instead.",
    ],
}
CONTROLS = [
    "Please open a pull request for this change.",
    "I changed my mind; please use JSON instead of a table.",
    "The airline cancelled my flight and I'm angry. Help me request a refund.",
]


@pytest.mark.skipif(
    not os.environ.get("INSIGHT_AGENT_EVAL_CLASSIFIER"), reason="Opt-in real-model evaluation"
)
def test_ten_complaints_are_classified_and_grouped():
    load_dotenv()
    api_key = resolve(ENV_API_KEY) or os.environ.get("INFERENCE_API_KEY")
    assert api_key, "Hosted grouping requires an inference API key"
    config = UserDissatisfactionConfig(device=os.environ.get("INSIGHT_AGENT_EVAL_DEVICE"))
    examples = [
        (theme, message)
        for pair in zip(*COMPLAINTS.values(), strict=True)
        for theme, message in zip(COMPLAINTS, pair, strict=True)
    ]
    messages = {
        f"case-{i:02d}": [message]
        for i, (_, message) in enumerate(examples + [(None, text) for text in CONTROLS])
    }
    expected = {
        theme: {f"case-{i:02d}" for i, (label, _) in enumerate(examples) if label == theme}
        for theme in COMPLAINTS
    }
    classifier = load_classifier(config.device)
    screening = screen_user_messages(messages, classifier)
    candidates = {key: text for key, text in messages.items() if screening[key].flagged}
    traces = TraceSnapshot(
        Trace(
            id=key,
            aggregate={},
            root_spans=[
                Span(
                    id="conversation",
                    kind=SpanKind.LLM,
                    input={
                        "messages": [
                            {
                                "role": "user",
                                "content": "Open a PR."
                                if key in expected["missing-pr"]
                                else "Return raw JSON only.",
                            },
                            {
                                "role": "assistant",
                                "content": "Done. Here is a local patch; no PR has been opened."
                                if key in expected["missing-pr"]
                                else "Here is my explanation in a paragraph.",
                            },
                            {"role": "user", "content": turns[0]},
                        ]
                    },
                )
            ],
        )
        for key, turns in candidates.items()
    )
    run_config = RunConfig(
        trace={"filesystem": {"path": "unused.jsonl"}},
        evidence_streams={"user_dissatisfaction": config},
        model=os.environ.get("INSIGHT_AGENT_EVAL_MODEL", "openai/azure/openai/gpt-5.6-luna"),
        api_base="https://inference-api.nvidia.com/v1",
    )
    problems = asyncio.run(
        IssueDetector(llm=_build_llm(run_config, api_key)).detect_issues(
            traces,
            USER_DISSATISFACTION,
            user_messages=candidates,
            screening={key: screening[key] for key in candidates},
        )
    )
    output = Path("tmp/user-dissatisfaction-mini-eval.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "messages": messages,
                "screening": {key: value.model_dump() for key, value in screening.items()},
                "problems": [problem.model_dump(mode="json") for problem in problems],
            },
            indent=2,
        )
    )
    assert set(candidates) == set.union(*expected.values()), (
        "Expected ten complaints and no controls"
    )
    assert {frozenset(p.supporting_trace_ids) for p in problems} == {
        frozenset(ids) for ids in expected.values()
    }, "Expected one complete group per complaint reason"
    assert all(
        any(
            len(quote) >= 20 and quote in text[0]
            for quote in re.findall(r"(?<!\w)[‘“\"'](.+?)[’”\"'](?!\w)", p.description)
            for text in candidates.values()
        )
        for p in problems
    ), "Groups must include verbatim user quotations"


@pytest.mark.skipif(
    not os.environ.get("INSIGHT_AGENT_EVAL_CLASSIFIER"), reason="Opt-in real-model evaluation"
)
def test_extraction_keeps_later_turns_and_original_event_bodies():
    from insight_agent.evidence_streams.user_dissatisfaction.stream import UserMessageExtractor

    load_dotenv()
    api_key = resolve(ENV_API_KEY) or os.environ.get("INFERENCE_API_KEY")
    assert api_key, "Extraction requires an inference API key"
    event = {
        "provider": "slack",
        "message": {"external_message_id": "3", "author": "human", "body": "Please finish the PR."},
        "context_messages": [
            {"external_message_id": "1", "author": "human", "body": "No PR was opened."},
            {"external_message_id": "2", "author": "human", "body": "No PR was opened."},
            {
                "external_message_id": "reply",
                "author": "assistant",
                "body": "I am disappointed in you.",
            },
        ],
    }
    context = {
        "messages": [
            {"direction": "outbound", "author": "assistant", "body": "I am disappointed in you."}
        ]
    }
    wrapped = {
        "messages": [
            {
                "role": "user",
                "content": "Inbound normalized event:\n"
                + json.dumps(event)
                + "\n\nExisting GLAMR work context:\n"
                + json.dumps(context),
            }
        ]
    }
    snapshot = TraceSnapshot(
        [
            Trace(
                id="actors",
                aggregate={},
                root_spans=[
                    Span(
                        id="seed",
                        kind=SpanKind.AGENT,
                        attributes={"subtype": "user"},
                        output="No PR was opened.",
                    ),
                    *[
                        Span(
                            id=str(i),
                            kind=SpanKind.LLM,
                            attributes={
                                "subtype": "agent",
                                "raw_attributes": {"tau2.actor": "user"},
                            },
                            output=text,
                        )
                        for i, text in enumerate(
                            ["No PR was opened.", "No PR was opened.", "Thanks.\n###STOP###"]
                        )
                    ],
                    Span(
                        id="tool",
                        kind=SpanKind.TOOL,
                        tool_name="lookup",
                        output="You ignored my request.",
                    ),
                ],
            ),
            Trace(
                id="events",
                aggregate={},
                root_spans=[
                    Span(
                        id=str(i),
                        kind=SpanKind.LLM,
                        output=json.dumps(wrapped),
                        attributes={"subtype": "user"},
                    )
                    for i in range(2)
                ],
            ),
            Trace(
                id="generated",
                aggregate={},
                root_spans=[
                    Span(
                        id="document",
                        kind=SpanKind.LLM,
                        input={
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "Automatically summarize the retrieved topic cluster; no human conversation is recorded.",
                                },
                                {
                                    "role": "user",
                                    "content": "Category: Agents\nNumber of topics in cluster: 1\n--- TOPIC ---\nUsers complained about a broken service.",
                                },
                            ]
                        },
                    )
                ],
            ),
        ]
    )
    config = RunConfig(
        trace={"filesystem": {"path": "unused.jsonl"}},
        evidence_streams={"anomaly_and_patterns": {}},
        model=os.environ.get("INSIGHT_AGENT_EVAL_MODEL", "openai/azure/openai/gpt-5.6-luna"),
        api_base="https://inference-api.nvidia.com/v1",
    )
    extraction = asyncio.run(
        UserMessageExtractor(llm=_build_llm(config, api_key)).build_extractor(snapshot)
    )
    extracted = extraction.extract(snapshot)
    assert extracted == {
        "actors": ["No PR was opened.", "No PR was opened.", "Thanks.\n"],
        "events": ["No PR was opened.", "No PR was opened.", "Please finish the PR."],
        "generated": [],
    }

    # The returned function must read its argument, not retain the inspection data.
    replay = TraceSnapshot(
        [
            Trace.model_validate_json(
                snapshot.get_trace_by_id("actors")
                .model_dump_json()
                .replace('"actors"', '"replay"')
                .replace("No PR was opened.", "You changed the wrong file.")
            )
        ]
    )
    assert extraction.extract(replay) == {
        "replay": ["You changed the wrong file.", "You changed the wrong file.", "Thanks.\n"]
    }
