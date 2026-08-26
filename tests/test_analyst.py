"""The Analyst stage.

litellm is mocked everywhere — a test that made a real call would be slow,
costly, and flaky, and would assert on model behaviour rather than on this
package's behaviour. What is worth pinning here is the contract around the
call: what goes into the prompt, what comes back out of a messy response, and
which parameters must never be sent.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from insight_agent.cli import EXIT_ERROR, EXIT_OK, main
from insight_agent.insights_generation import (
    DEFAULT_MODEL,
    InsightsGenerationError,
    ResponseParseError,
)
from insight_agent.insights_generation.llm import (
    _AnalystRequest,
    _author_insights,
    _build_prompt,
    _fetch_traces,
    _parse_insights,
    _prompt_template,
    _select_cards,
)
from insight_agent.traces import Span, SpanKind, ToolCall, Trace, TraceSnapshot

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "insight_agent" / "data"
CORPUS = DATA_DIR / "sample_corpus.jsonl"
EMPTY_SNAPSHOT = TraceSnapshot.from_traces(())


def generate(request: _AnalystRequest, *, snapshot: TraceSnapshot = EMPTY_SNAPSHOT, **kwargs):
    return _author_insights(request, snapshot=snapshot, **kwargs)


def card(card_id: str, *, eligible: bool = True, trace: str = "t1") -> dict:
    return {
        "card_id": card_id,
        "issue_type": "unknown_tool",
        "issue_family": "tool_contract_and_arguments",
        "mechanism_key": "unknown_tool",
        "finding_count": 5,
        "independent_case_count": 3 if eligible else 1,
        "eligible_for_analyst": eligible,
        "representative_evidence": [
            {
                "trace_id": trace,
                "call_id": f"{trace}#0",
                "call_index": 0,
                "tool_name": "GhostTool",
                "observation": "Tool is absent from the active catalog.",
            }
        ],
        "impact_status": "not_established",
        "impact_boundary": "No impact is claimed beyond the directly observed tool-use issue.",
    }


@pytest.fixture
def request_obj():
    return _AnalystRequest(
        agent="DocOps agent",
        digest="# IA2 cited evidence digest\n\n| Trace |\n| `docops-outlier` |",
        cards=[card("tid:unknown_tool:unknown_tool")],
        withheld_card_count=4,
        corpus={"trace_count": 18, "call_count": 79, "distinct_logical_cases": 16},
    )


def fake_response(content: str, *, tokens: int = 1234):
    """A litellm response in the OpenAI shape it normalises every provider to."""
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": tokens, "completion_tokens": 99, "total_tokens": tokens + 99},
    }


@pytest.fixture
def mock_litellm(monkeypatch):
    """Install a fake `litellm` module and record the kwargs it receives."""
    calls = []

    def _install(content='[{"name":"N","description":"D","trace_ids":["t1"]}]'):
        module = types.ModuleType("litellm")

        def completion(**kwargs):
            calls.append(kwargs)
            return fake_response(content)

        module.completion = completion
        monkeypatch.setitem(sys.modules, "litellm", module)
        return calls

    return _install


# -- prompt assembly -------------------------------------------------------


def test_prompt_carries_the_agent_name(request_obj):
    system, _ = _build_prompt(request_obj)
    assert "DocOps agent" in system
    assert "{agent}" not in system


def test_prompt_includes_the_digest_verbatim(request_obj):
    system, _ = _build_prompt(request_obj)
    assert request_obj.digest.strip() in system


def test_prompt_includes_every_shown_card(request_obj):
    system, _ = _build_prompt(request_obj)
    assert "tid:unknown_tool:unknown_tool" in system
    assert "independent_case_count" in system


def test_prompt_states_the_withheld_card_count(request_obj):
    """Silently dropping audit-only cards would misrepresent the evidence set."""
    system, _ = _build_prompt(request_obj)
    assert "4 card(s) were withheld" in system
    assert "must not be cited" in system


def test_prompt_declares_the_lookup_tool(request_obj):
    """Naming the tool is what enables the loop — a prompt that promises tools
    without naming one would be offered no tools at all."""
    system, _ = _build_prompt(request_obj)
    assert "fetch_traces" in system
    assert "max_chars_per_result" in system


def test_prompt_states_the_output_contract_and_permits_zero_insights(request_obj):
    """Without explicit permission a model will manufacture something."""
    system, _ = _build_prompt(request_obj)
    assert '"trace_ids"' in system
    assert "`[]`" in system


def test_output_contract_is_not_duplicated(request_obj):
    """The shipped prompt states the contract; a second copy could conflict."""
    system, user = _build_prompt(request_obj)
    assert '"trace_ids"' in system
    assert '"trace_ids"' not in user


def test_output_contract_is_supplied_when_the_template_omits_it(tmp_path, monkeypatch):
    """A hand-written prompt should still produce a parseable shape."""
    from insight_agent.insights_generation import llm

    monkeypatch.setattr(
        llm, "_prompt_template", lambda v=None: "Find problems in {agent}.\n\n{evidence}"
    )
    system, user = llm._build_prompt(llm._AnalystRequest(agent="A", digest="d"))
    assert '"trace_ids"' in user


def test_a_template_without_the_evidence_marker_is_an_error(monkeypatch):
    """Silently appending the evidence would hide a typo in a custom prompt and
    send a paid request whose evidence landed somewhere the prompt never refers to."""
    from insight_agent.insights_generation import llm

    monkeypatch.setattr(llm, "_prompt_template", lambda v=None: "Analyse {agent}.")
    with pytest.raises(InsightsGenerationError, match="{evidence}"):
        llm._build_prompt(llm._AnalystRequest(agent="A", digest="d"))


def test_prompt_explains_a_missing_card_stream():
    request = _AnalystRequest(agent="A", digest="d", cards=[])
    system, _ = _build_prompt(request)
    assert "No cards reached the Analyst" in system
    assert "do not compensate by lowering" in system


def test_prompt_template_is_versioned_and_packaged():
    assert "You are the Analyst agent" in _prompt_template()


# -- card selection --------------------------------------------------------


def test_only_eligible_cards_are_shown_by_default():
    cards = [card("a"), card("b", eligible=False), card("c", eligible=False)]
    shown, withheld = _select_cards(cards)
    assert [c["card_id"] for c in shown] == ["a"]
    assert withheld == 2


def test_all_cards_shows_everything_and_withholds_nothing():
    cards = [card("a"), card("b", eligible=False)]
    shown, withheld = _select_cards(cards, all_cards=True)
    assert len(shown) == 2
    assert withheld == 0


# -- response parsing ------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '[{"name":"N","description":"D","trace_ids":["t1"]}]',
        '{"insights":[{"name":"N","description":"D","trace_ids":["t1"]}]}',
        '```json\n[{"name":"N","description":"D","trace_ids":["t1"]}]\n```',
        'Sure, here you go:\n[{"name":"N","description":"D","trace_ids":["t1"]}]\nLet me know!',
        '{"name":"N","description":"D","trace_ids":["t1"]}',
    ],
    ids=["bare", "wrapped", "fenced", "prose", "single-object"],
)
def test_parse_tolerates_the_shapes_a_model_actually_returns(raw):
    assert _parse_insights(raw) == [{"name": "N", "description": "D", "trace_ids": ["t1"]}]


def test_parse_accepts_an_empty_array():
    assert _parse_insights("[]") == []


def test_parse_coerces_a_bare_string_trace_id():
    assert _parse_insights('[{"name":"N","description":"D","trace_ids":"t1"}]')[0]["trace_ids"] == [
        "t1"
    ]


def test_parse_drops_entries_without_a_name():
    parsed = _parse_insights('[{"description":"no name"},{"name":"keep","description":"d"}]')
    assert [i["name"] for i in parsed] == ["keep"]


def test_parse_deduplicates_trace_ids():
    parsed = _parse_insights('[{"name":"N","description":"D","trace_ids":["t1","t1","t2"]}]')
    assert parsed[0]["trace_ids"] == ["t1", "t2"]


@pytest.mark.parametrize("raw", ["", "   ", "I could not complete this request."])
def test_parse_failure_carries_the_raw_text(raw):
    """The response was already paid for; losing it is the wrong failure mode."""
    with pytest.raises(ResponseParseError) as excinfo:
        _parse_insights(raw)
    assert excinfo.value.raw == raw


# -- the call --------------------------------------------------------------


def test_no_sampling_parameters_are_ever_sent(mock_litellm, request_obj):
    """temperature/top_p/top_k are REMOVED on Claude Opus 5 and return a 400.

    This is a test rather than a comment because passing one "to be safe" is
    the easiest way to break the call, and it would only ever fail at runtime
    against a real API.
    """
    calls = mock_litellm()
    generate(request_obj)

    assert len(calls) == 1
    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in calls[0], forbidden


def test_call_uses_the_documented_defaults(mock_litellm, request_obj):
    calls = mock_litellm()
    result = generate(request_obj)

    assert calls[0]["model"] == DEFAULT_MODEL == "anthropic/claude-opus-5"
    assert calls[0]["max_tokens"] >= 4096
    assert [m["role"] for m in calls[0]["messages"]] == ["system", "user"]
    assert result.usage["total_tokens"] == 1333


def test_result_exposes_the_trace_ids_it_cited(mock_litellm, request_obj):
    mock_litellm('[{"name":"N","description":"D","trace_ids":["t1","t2"]}]')
    assert generate(request_obj).cited_trace_ids == {"t1", "t2"}


def test_api_base_and_key_are_forwarded_when_given(mock_litellm, request_obj):
    """This is how any OpenAI-compatible gateway is reached."""
    calls = mock_litellm()
    generate(
        request_obj,
        model="openai/openai/gpt-5.2",
        api_base="https://gateway.example.com/v1",
        api_key="secret",
    )
    assert calls[0]["api_base"] == "https://gateway.example.com/v1"
    assert calls[0]["api_key"] == "secret"
    assert calls[0]["model"] == "openai/openai/gpt-5.2"


def test_unset_api_base_is_not_forwarded(mock_litellm, request_obj):
    """Passing api_base=None would stop litellm resolving the provider default."""
    calls = mock_litellm()
    generate(request_obj)
    assert "api_base" not in calls[0]
    assert "api_key" not in calls[0]


def test_missing_litellm_names_the_extra_to_install(monkeypatch, request_obj):
    """The package must import and run without litellm; only this call needs it."""
    monkeypatch.setitem(sys.modules, "litellm", None)
    monkeypatch.delitem(sys.modules, "litellm")

    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def blocked(name, *args, **kwargs):
        if name == "litellm":
            raise ModuleNotFoundError("No module named 'litellm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    with pytest.raises(InsightsGenerationError, match=r"\[analyst\]"):
        generate(request_obj)


# -- trace lookup ----------------------------------------------------------


SNAPSHOT = TraceSnapshot.from_traces(
    (
        Trace(
            id="t1",
            logical_case_id="case-a",
            input="do the thing",
            spans=(
                Span(
                    span_id="c0",
                    kind=SpanKind.TOOL,
                    tool_name="Search",
                    input={"q": "x"},
                    output={"content": "y" * 5000},
                    tool_call=ToolCall(call_id="c0", index=0),
                ),
                Span(
                    span_id="c1",
                    kind=SpanKind.TOOL,
                    tool_name="Read",
                    input={},
                    tool_call=ToolCall(call_id="c1", index=1),
                ),
            ),
        ),
        Trace(id="t2", spans=()),
    )
)
TRACE_INDEX = {trace.id: trace for trace in SNAPSHOT.scan()}


def test_fetch_returns_the_requested_traces():
    out = _fetch_traces(TRACE_INDEX, ["t1", "t2"])
    assert [trace["id"] for trace in out["traces"]] == ["t1", "t2"]
    assert "not_found" not in out


def test_fetch_truncates_large_results_but_says_so():
    """The traces worth opening are the ones flagged for huge outputs."""
    result = _fetch_traces(TRACE_INDEX, ["t1"], max_chars_per_result=100)["traces"][0]["spans"][0][
        "output"
    ]
    assert "truncated" in result
    assert "5,0" in result  # reports the true size


def test_fetch_honours_a_raised_cap():
    result = _fetch_traces(TRACE_INDEX, ["t1"], max_chars_per_result=50_000)["traces"][0]["spans"][
        0
    ]["output"]
    assert result == {"content": "y" * 5000}


def test_fetch_preserves_the_missing_result_distinction():
    """`result` absent must stay absent — it is what missing_tool_result means."""
    spans = _fetch_traces(TRACE_INDEX, ["t1"])["traces"][0]["spans"]
    assert "output" in spans[0]
    assert "output" not in spans[1]


def test_fetch_reports_unknown_ids_rather_than_dropping_them():
    """An empty result would read as 'this trace has no calls'."""
    out = _fetch_traces(TRACE_INDEX, ["t1", "nope"])
    assert out["not_found"] == ["nope"]
    assert "do not reconstruct" in out["note"]


def tool_call(call_id, ids, **extra):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "fetch_traces", "arguments": json.dumps({"trace_ids": ids, **extra})},
    }


@pytest.fixture
def scripted_litellm(monkeypatch):
    """A litellm whose responses are queued in advance."""
    calls = []

    def _install(*responses):
        queue = list(responses)
        module = types.ModuleType("litellm")

        def completion(**kwargs):
            calls.append(kwargs)
            payload = queue.pop(0)
            message = payload if isinstance(payload, dict) else {"content": payload}
            return {
                "choices": [{"message": {"role": "assistant", **message}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

        module.completion = completion
        monkeypatch.setitem(sys.modules, "litellm", module)
        return calls

    return _install


def tool_request(*ids):
    return {"content": None, "tool_calls": [tool_call("call_1", list(ids))]}


ANSWER = '[{"name":"N","description":"D","trace_ids":["t1"]}]'


@pytest.fixture
def tool_prompt(monkeypatch):
    from insight_agent.insights_generation import llm

    monkeypatch.setattr(
        llm,
        "_prompt_template",
        lambda v=None: "Analyse {agent}. Use fetch_traces to read raw traces.\n\n{evidence}",
    )


def test_tools_are_offered_when_the_prompt_asks(scripted_litellm, tool_prompt, request_obj):
    calls = scripted_litellm(ANSWER)

    generate(request_obj, snapshot=SNAPSHOT)
    assert "tools" in calls[0]


def test_a_prompt_that_never_names_the_tool_gets_no_tools(
    scripted_litellm, monkeypatch, request_obj
):
    """A hand-written prompt that never mentions the tool would have no way to
    explain it to the model, so offering it anyway just invites confusion."""
    from insight_agent.insights_generation import llm

    monkeypatch.setattr(llm, "_prompt_template", lambda v=None: "Analyse {agent}.\n\n{evidence}")
    calls = scripted_litellm(ANSWER)
    generate(request_obj, snapshot=SNAPSHOT)
    assert "tools" not in calls[0]


def test_tool_loop_executes_the_fetch_and_feeds_it_back(scripted_litellm, tool_prompt, request_obj):
    calls = scripted_litellm(tool_request("t1", "t2"), ANSWER)
    result = generate(request_obj, snapshot=SNAPSHOT)

    assert result.tool_calls == 1
    assert result.traces_fetched == ("t1", "t2")
    assert result.insights[0]["name"] == "N"

    # Second request carries the assistant turn plus a tool result.
    roles = [m["role"] for m in calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    payload = json.loads(calls[1]["messages"][-1]["content"])
    assert [trace["id"] for trace in payload["traces"]] == ["t1", "t2"]


def test_usage_accumulates_across_tool_rounds(scripted_litellm, tool_prompt, request_obj):
    scripted_litellm(tool_request("t1"), tool_request("t2"), ANSWER)
    result = generate(request_obj, snapshot=SNAPSHOT)

    assert result.tool_calls == 2
    assert result.usage["total_tokens"] == 45  # three round trips


def test_exhausting_the_round_budget_still_produces_an_answer(
    scripted_litellm, tool_prompt, request_obj
):
    """A model that keeps fetching must not end the turn with no Insights."""
    scripted_litellm(tool_request("t1"), tool_request("t1"), ANSWER)
    result = generate(request_obj, snapshot=SNAPSHOT, max_tool_rounds=2)

    assert result.insights[0]["name"] == "N"
    assert result.tool_calls == 2


def test_final_retry_withdraws_the_tools(scripted_litellm, tool_prompt, request_obj):
    calls = scripted_litellm(tool_request("t1"), tool_request("t1"), ANSWER)
    generate(request_obj, snapshot=SNAPSHOT, max_tool_rounds=2)

    assert "tools" in calls[0]
    assert "tools" not in calls[-1], "final call must not offer tools again"
    assert "lookups available" in calls[-1]["messages"][-1]["content"]


def test_evidence_is_inlined_when_the_template_asks_for_it(tool_prompt, request_obj):
    system, user = _build_prompt(request_obj)
    assert "# Evidence for DocOps agent" in system, "evidence substituted at {evidence}"
    assert "{evidence}" not in system
    assert "# Evidence for" not in user, "evidence goes in the system prompt, not both"


def test_a_self_contained_template_gets_only_a_kickoff(monkeypatch, request_obj):
    """A template with its own contract owns the whole prompt."""
    from insight_agent.insights_generation import llm

    monkeypatch.setattr(
        llm,
        "_prompt_template",
        lambda v=None: 'Analyse {agent}. Emit "trace_ids". Use fetch_traces.\n\n{evidence}',
    )
    system, user = llm._build_prompt(request_obj)
    assert user.strip() == "Author Insights for this corpus."


# -- CLI -------------------------------------------------------------------


def test_dry_run_writes_the_prompt_and_makes_no_call(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", None)  # any use would explode
    out = tmp_path / "out"

    assert (
        main(["run-analyst", str(CORPUS), "--agent", "DocOps", "-o", str(out), "--dry-run"])
        == EXIT_OK
    )

    prompt = (out / "analyst" / "prompt.md").read_text(encoding="utf-8")
    assert "DocOps" in prompt
    assert not (out / "analyst" / "insights.json").exists()
    assert "no API call" in capsys.readouterr().out


def test_full_run_writes_insights_and_provenance(tmp_path, mock_litellm):
    mock_litellm(
        '[{"name":"Invented tools","description":"D","trace_ids":["docops-search-alpha"]}]'
    )
    out = tmp_path / "out"

    assert (
        main(["run-analyst", str(CORPUS), "--agent", "DocOps", "-o", str(out), "--quiet"])
        == EXIT_OK
    )

    insights = json.loads((out / "analyst" / "insights.json").read_text(encoding="utf-8"))
    assert [i["name"] for i in insights] == ["Invented tools"]
    assert set(insights[0]) == {"name", "description", "trace_ids"}

    run = json.loads((out / "analyst" / "run.json").read_text(encoding="utf-8"))
    assert run["agent"] == "DocOps"
    assert run["prompt_version"] == "analyst_v3"
    assert run["insight_count"] == 1
    assert run["cards_withheld"] > 0


def test_unparseable_response_is_saved_not_lost(tmp_path, mock_litellm, capsys):
    mock_litellm("I'm afraid I can't do that.")
    out = tmp_path / "out"

    assert main(["run-analyst", str(CORPUS), "--agent", "DocOps", "-o", str(out)]) == EXIT_ERROR
    assert (
        (out / "analyst" / "analyst_raw.txt").read_text(encoding="utf-8").startswith("I'm afraid")
    )
    assert "analyst_raw.txt" in capsys.readouterr().err


def test_zero_insights_is_reported_as_a_valid_outcome(tmp_path, mock_litellm, capsys):
    mock_litellm("[]")
    out = tmp_path / "out"

    assert main(["run-analyst", str(CORPUS), "--agent", "DocOps", "-o", str(out)]) == EXIT_OK
    assert json.loads((out / "analyst" / "insights.json").read_text(encoding="utf-8")) == []
    assert "valid outcome" in capsys.readouterr().out


def test_a_fabricated_trace_id_is_reported_loudly(tmp_path, mock_litellm, capsys):
    """No validator gates this, but the failure must at least be visible."""
    mock_litellm('[{"name":"N","description":"D","trace_ids":["totally-invented-id"]}]')
    out = tmp_path / "out"

    main(["run-analyst", str(CORPUS), "--agent", "DocOps", "-o", str(out)])
    printed = capsys.readouterr().out
    assert "not in the evidence" in printed
    assert "do not exist in the corpus at all" in printed


def test_a_fetched_trace_id_is_not_reported_as_unknown(tmp_path, mock_litellm, capsys):
    """With trace lookup, a trace the model opened itself is legitimate."""
    real = json.loads(CORPUS.read_text(encoding="utf-8").splitlines()[0])["trace_id"]
    mock_litellm(f'[{{"name":"N","description":"D","trace_ids":["{real}"]}}]')

    main(["run-analyst", str(CORPUS), "--agent", "DocOps", "-o", str(tmp_path / "out")])
    assert "not in the evidence" not in capsys.readouterr().out


def test_cli_reads_model_and_endpoint_from_a_dotenv(tmp_path, mock_litellm, monkeypatch):
    """The `.env` path is how an OpenAI-compatible gateway gets configured."""
    for name in ("INSIGHT_AGENT_MODEL", "INSIGHT_AGENT_API_BASE", "INSIGHT_AGENT_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "INSIGHT_AGENT_API_BASE=https://gateway.example.com/v1\n"
        "INSIGHT_AGENT_MODEL=openai/openai/gpt-5.2\n"
        "INSIGHT_AGENT_API_KEY=nv-secret\n",
        encoding="utf-8",
    )
    calls = mock_litellm()

    code = main(
        [
            "run-analyst",
            str(CORPUS),
            "--agent",
            "DocOps",
            "-o",
            str(tmp_path / "out"),
            "--env-file",
            str(env_file),
            "--quiet",
        ]
    )

    assert code == EXIT_OK
    assert calls[0]["model"] == "openai/openai/gpt-5.2"
    assert calls[0]["api_base"] == "https://gateway.example.com/v1"
    assert calls[0]["api_key"] == "nv-secret"


def test_cli_flags_beat_the_dotenv(tmp_path, mock_litellm, monkeypatch):
    monkeypatch.delenv("INSIGHT_AGENT_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("INSIGHT_AGENT_MODEL=from-file\n", encoding="utf-8")
    calls = mock_litellm()

    main(
        [
            "run-analyst",
            str(CORPUS),
            "--agent",
            "DocOps",
            "-o",
            str(tmp_path / "out"),
            "--env-file",
            str(env_file),
            "--model",
            "from-flag",
            "--quiet",
        ]
    )
    assert calls[0]["model"] == "from-flag"


def test_temperature_is_only_sent_when_asked_for(tmp_path, mock_litellm):
    """Several current models reject it outright, so it must stay opt-in."""
    calls = mock_litellm()
    out = tmp_path / "out"

    main(["run-analyst", str(CORPUS), "--agent", "A", "-o", str(out), "--quiet"])
    assert "temperature" not in calls[0]

    main(
        [
            "run-analyst",
            str(CORPUS),
            "--agent",
            "A",
            "-o",
            str(out),
            "--temperature",
            "0.7",
            "--quiet",
        ]
    )
    assert calls[1]["temperature"] == 0.7


def test_dry_run_reports_whether_credentials_were_found(tmp_path, capsys, monkeypatch):
    for name in ("INSIGHT_AGENT_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    empty = tmp_path / ".env"
    empty.write_text("INSIGHT_AGENT_API_KEY=\n", encoding="utf-8")

    main(
        [
            "run-analyst",
            str(CORPUS),
            "--agent",
            "A",
            "-o",
            str(tmp_path / "out"),
            "--env-file",
            str(empty),
            "--dry-run",
        ]
    )
    assert "credentials           : NOT FOUND" in capsys.readouterr().out


def test_digest_and_cards_must_be_given_together(tmp_path, capsys):
    code = main(
        ["run-analyst", str(CORPUS), "--agent", "A", "-o", str(tmp_path), "--digest", str(CORPUS)]
    )
    assert code == EXIT_ERROR
    assert "must be given together" in capsys.readouterr().err


def test_no_analyst_keeps_run_all_free_of_any_api_dependency(tmp_path, monkeypatch):
    """--no-analyst is the escape hatch for a purely deterministic run."""
    monkeypatch.setitem(sys.modules, "litellm", None)  # any use would explode
    out = tmp_path / "out"

    assert main(["run-all", str(CORPUS), "-o", str(out), "--quiet", "--no-analyst"]) == EXIT_OK
    assert not (out / "analyst").exists()
    # The index still *mentions* the Analyst as the digest's reader; what must
    # be absent is the section linking to authored output.
    index = (out / "index.md").read_text(encoding="utf-8")
    assert "## Analyst" not in index
    assert "analyst/insights.json" not in index


def test_run_all_runs_the_analyst_by_default_reusing_the_computed_evidence(
    tmp_path, mock_litellm, fake_credentials
):
    mock_litellm()
    out = tmp_path / "out"

    code = main(["run-all", str(CORPUS), "-o", str(out), "--quiet", "--agent", "DocOps"])
    assert code == EXIT_OK
    assert (out / "analyst" / "insights.json").exists()
    assert "analyst/insights.json" in (out / "index.md").read_text(encoding="utf-8")


def test_agent_name_defaults_to_the_corpus_filename(tmp_path, mock_litellm, fake_credentials):
    """--agent is a label, not a gate; requiring it would block the default path."""
    calls = mock_litellm()
    out = tmp_path / "out"

    assert main(["run-all", str(CORPUS), "-o", str(out), "--quiet"]) == EXIT_OK
    assert CORPUS.stem in calls[0]["messages"][0]["content"]


def test_run_all_fails_loudly_when_the_analyst_has_no_credentials(tmp_path, monkeypatch, capsys):
    """A run that silently stopped at evidence would look complete but not be."""
    for var in ("INSIGHT_AGENT_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # An empty env file, so a developer's real .env cannot rescue the run.
    empty = tmp_path / "empty.env"
    empty.write_text("", encoding="utf-8")

    code = main(
        ["run-all", str(CORPUS), "-o", str(tmp_path / "out"), "--quiet", "--env-file", str(empty)]
    )
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "API key" in err
    assert "--no-analyst" in err


def test_run_all_fails_loudly_when_litellm_is_absent(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "litellm", None)
    code = main(["run-all", str(CORPUS), "-o", str(tmp_path / "out"), "--quiet"])
    assert code == EXIT_ERROR
    assert "--no-analyst" in capsys.readouterr().err


# -- the digest inventory fix ---------------------------------------------


def test_digest_reports_the_true_flag_count_above_the_render_cap():
    """The inventory used to report the truncated row count as the total."""
    from insight_agent.evidence_streams.anomaly_and_patterns import (
        NormalizedTrace,
        PreparedTrace,
        TraceFeatures,
        build_evidence_digest,
    )

    prepared, anomalies = [], []
    for i in range(120):
        trace = NormalizedTrace(trace_id=f"t{i:03d}", calls=())
        prepared.append(
            PreparedTrace(
                trace=trace,
                features=TraceFeatures(trace_id=trace.trace_id, numeric={}, sequence_tokens=()),
                tool_names=(),
                failure_events=(),
                last_agent_excerpt="",
            )
        )
        anomalies.append(
            {
                "trace_id": trace.trace_id,
                "anomaly_score": 1.0 - i / 1000,
                "is_anomaly": True,
                "anomaly_reasons": ["high output"],
                "pca_x": 0.0,
                "pca_y": 0.0,
                "source_pointer": {},
            }
        )

    digest = build_evidence_digest(prepared, anomalies, None, [], [], [])

    assert "- Isolation Forest flags: 120" in digest
    assert "Showing the 50 highest-scoring of 120 flagged traces" in digest
    assert sum(1 for line in digest.splitlines() if line.startswith("| `t")) == 50
