from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from insight_agent.trace_loaders import (
    MLflowFileTraceConfig,
    MLflowFileTraceLoader,
    MLflowTraceConfig,
    MLflowTraceLoader,
    MLflowTraceLoadError,
)
from insight_agent.traces import UNSET, SpanKind, SpanStatus

MLFLOW_TRACE_FIXTURE = Path(__file__).parent / "data" / "mlflow_trace_v3.json"


@dataclass
class FakePage:
    items: list[object]
    token: str | None = None

    def __iter__(self):
        return iter(self.items)


class FakeClient:
    tracking_uri = "http://mlflow.test"

    def __init__(self, *, pages=(), experiment_id="17"):
        self.pages = list(pages)
        self.experiment_id = experiment_id
        self.search_calls = []

    def get_experiment_by_name(self, name):
        return (
            None
            if self.experiment_id is None
            else SimpleNamespace(experiment_id=self.experiment_id)
        )

    def search_traces(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.pages.pop(0)


def status(code, description=""):
    return SimpleNamespace(status_code=SimpleNamespace(value=code), description=description)


def span(
    span_id,
    *,
    parent_id=None,
    name=None,
    span_type="UNKNOWN",
    start_time_ns=1_000_000_000,
    end_time_ns=2_000_000_000,
    status_code="UNSET",
    status_description="",
    inputs=UNSET,
    outputs=UNSET,
    attributes=None,
):
    attrs = dict(attributes or {})
    if inputs is not UNSET:
        attrs["mlflow.spanInputs"] = inputs
    if outputs is not UNSET:
        attrs["mlflow.spanOutputs"] = outputs
    attrs.setdefault("mlflow.spanType", span_type)
    return SimpleNamespace(
        span_id=span_id,
        parent_id=parent_id,
        name=name or span_id,
        span_type=span_type,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        status=status(status_code, status_description),
        inputs=None if inputs is UNSET else inputs,
        outputs=None if outputs is UNSET else outputs,
        attributes=attrs,
    )


def trace(trace_id, spans=(), *, request_time=0, metadata=None):
    return SimpleNamespace(
        info=SimpleNamespace(
            trace_id=trace_id,
            request_time=request_time,
            trace_metadata=dict(metadata or {}),
        ),
        data=SimpleNamespace(spans=list(spans)),
    )


def native_trace_dict(trace_id="tr-9df8a4c934051e916458d472ac87ee2a"):
    # Emitted by MLflow 3.15.2's Trace.to_json(pretty=True). Keep this as a real
    # provider serialization; the public contract and serializer coverage live at:
    # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Trace.to_json
    # https://github.com/mlflow/mlflow/blob/v3.15.2/tests/entities/test_trace.py#L65-L166
    record = json.loads(MLFLOW_TRACE_FIXTURE.read_text(encoding="utf-8"))
    record["info"]["trace_id"] = trace_id
    for span_record in record["data"]["spans"]:
        span_record["attributes"]["mlflow.traceRequestId"] = json.dumps(trace_id)
    return record


def test_file_loader_reads_native_mlflow_search_json_without_conversion(tmp_path):
    path = tmp_path / "traces.json"
    path.write_text(
        json.dumps({"traces": [native_trace_dict()], "next_page_token": "more"}),
        encoding="utf-8",
    )

    loader = MLflowFileTraceLoader(MLflowFileTraceConfig(path=path, max_traces=10))
    normalized = next(loader.load().scan())

    assert normalized.id == "tr-9df8a4c934051e916458d472ac87ee2a"
    assert normalized.logical_case_id is None
    assert normalized.spans[0].kind is SpanKind.TOOL
    assert normalized.spans[0].input == {"query": "why did the agent retry?"}
    assert normalized.spans[0].output == {"answer": "The upstream timed out."}
    assert normalized.source_pointer == {
        "provider": "mlflow",
        "export_path": str(path.resolve()),
        "trace_id": "tr-9df8a4c934051e916458d472ac87ee2a",
    }
    assert loader.describe()["continuation_token_present"] is True


def test_file_loader_applies_the_trace_bound_and_reports_truncation(tmp_path):
    path = tmp_path / "traces.json"
    path.write_text(
        json.dumps(
            [
                native_trace_dict("tr-11111111111111111111111111111111"),
                native_trace_dict("tr-22222222222222222222222222222222"),
                native_trace_dict("tr-33333333333333333333333333333333"),
            ]
        ),
        encoding="utf-8",
    )

    loader = MLflowFileTraceLoader(MLflowFileTraceConfig(path=path, max_traces=2))
    snapshot = loader.load()

    assert [trace.id for trace in snapshot.scan()] == [
        "tr-11111111111111111111111111111111",
        "tr-22222222222222222222222222222222",
    ]
    assert loader.describe()["export_trace_count"] == 3
    assert loader.describe()["truncated"] is True


@pytest.mark.parametrize("shape", ["bare", "array", "jsonl"])
def test_file_loader_accepts_native_mlflow_trace_serializations(tmp_path, shape):
    first = native_trace_dict("tr-11111111111111111111111111111111")
    second = native_trace_dict("tr-22222222222222222222222222222222")
    if shape == "bare":
        contents = json.dumps(first)
        expected = [first["info"]["trace_id"]]
    elif shape == "array":
        contents = json.dumps([first, second])
        expected = [first["info"]["trace_id"], second["info"]["trace_id"]]
    else:
        contents = f"{json.dumps(first)}\n{json.dumps(second)}\n"
        expected = [first["info"]["trace_id"], second["info"]["trace_id"]]
    path = tmp_path / f"traces-{shape}.json"
    path.write_text(contents, encoding="utf-8")

    snapshot = MLflowFileTraceLoader(MLflowFileTraceConfig(path=path)).load()

    assert [trace.id for trace in snapshot.scan()] == expected


def test_file_loader_rejects_exports_without_span_payloads(tmp_path):
    record = native_trace_dict()
    record["data"]["spans"] = []
    path = tmp_path / "trace-info-only.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(MLflowTraceLoadError, match="has no spans.*--no-include-spans"):
        MLflowFileTraceLoader(MLflowFileTraceConfig(path=path)).load()


def test_file_loader_rejects_duplicate_trace_ids(tmp_path):
    record = native_trace_dict()
    path = tmp_path / "duplicates.json"
    path.write_text(json.dumps([record, record]), encoding="utf-8")

    with pytest.raises(MLflowTraceLoadError, match="duplicate trace id"):
        MLflowFileTraceLoader(MLflowFileTraceConfig(path=path)).load()


def test_loader_resolves_experiment_and_pages_complete_traces_through_search():
    client = FakeClient(
        pages=[
            FakePage(
                [trace("tr-1", request_time=20), trace("tr-0", request_time=10)], token="next"
            ),
            FakePage([trace("tr-2", request_time=30)]),
        ],
    )

    loader = MLflowTraceLoader(
        MLflowTraceConfig(
            experiment_name="agent-traces",
            filter_string="trace.status = 'OK'",
            max_traces=3,
        ),
        client=client,
    )
    snapshot = loader.load()

    assert [item.id for item in snapshot.scan()] == ["tr-0", "tr-1", "tr-2"]
    assert client.search_calls == [
        {
            "locations": ["17"],
            "filter_string": "trace.status = 'OK'",
            "max_results": 3,
            "order_by": ["timestamp_ms DESC", "request_id ASC"],
            "page_token": None,
            "include_spans": True,
        },
        {
            "locations": ["17"],
            "filter_string": "trace.status = 'OK'",
            "max_results": 1,
            "order_by": ["timestamp_ms DESC", "request_id ASC"],
            "page_token": "next",
            "include_spans": True,
        },
    ]


def test_loader_normalizes_mlflow_spans_without_flattening_payloads():
    root = span(
        "root",
        name="agent",
        span_type="AGENT",
        start_time_ns=1_000_000_000,
        end_time_ns=5_000_000_000,
        status_code="OK",
        inputs={"messages": [{"role": "user", "content": "find it"}]},
        outputs={"answer": "done"},
    )
    llm = span(
        "llm",
        parent_id="root",
        span_type="CHAT_MODEL",
        start_time_ns=2_000_000_000,
        end_time_ns=3_000_000_000,
        status_code="ERROR",
        status_description="rate_limit",
        inputs={"prompt": ["one", "two"]},
        outputs=None,
    )
    tool = span(
        "tool",
        parent_id="root",
        name="search",
        span_type="TOOL",
        start_time_ns=3_000_000_000,
        end_time_ns=4_500_000_000,
        inputs={"query": "x", "limit": 3},
    )
    full = trace(
        "tr-1",
        # Deliberately shuffled: the normalized contract requires canonical order.
        [tool, root, llm],
        metadata={"mlflow.trace.session": "session-7"},
    )
    client = FakeClient(pages=[FakePage([full])])

    normalized = next(
        MLflowTraceLoader(MLflowTraceConfig(experiment_name="agent-traces"), client=client)
        .load()
        .scan()
    )

    assert normalized.input == {"messages": [{"role": "user", "content": "find it"}]}
    assert normalized.logical_case_id == "session-7"
    assert [item.span_id for item in normalized.spans] == ["root", "llm", "tool"]
    by_id = {item.span_id: item for item in normalized.spans}
    assert by_id["root"].kind is SpanKind.AGENT
    assert by_id["root"].status is SpanStatus.SUCCESS
    assert by_id["root"].started_at == datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    assert by_id["root"].duration_ms == 4000.0
    assert by_id["llm"].kind is SpanKind.LLM
    assert by_id["llm"].subtype == "CHAT_MODEL"
    assert by_id["llm"].status is SpanStatus.ERROR
    assert by_id["llm"].error_type == "rate_limit"
    assert by_id["llm"].input == {"prompt": ["one", "two"]}
    assert by_id["llm"].output is None
    assert by_id["tool"].kind is SpanKind.TOOL
    assert by_id["tool"].tool_name == "search"
    assert by_id["tool"].input == {"query": "x", "limit": 3}
    assert by_id["tool"].output is UNSET


def test_otlp_json_null_is_distinct_from_an_absent_output():
    explicit_null = span(
        "null",
        span_type="TOOL",
        outputs="null",
        attributes={"output.mime_type": "application/json"},
    )
    missing = span(
        "missing",
        span_type="TOOL",
        start_time_ns=3_000_000_000,
        end_time_ns=4_000_000_000,
    )
    full = trace("tr-null", [missing, explicit_null])
    client = FakeClient(pages=[FakePage([full])])

    normalized = next(
        MLflowTraceLoader(MLflowTraceConfig(experiment_name="agent-traces"), client=client)
        .load()
        .scan()
    )
    by_id = {item.span_id: item for item in normalized.spans}

    # MLflow 3.15.2 exposes an OTLP JSON null as this string through
    # ``Span.outputs``. Attribute presence is what distinguishes it from missing.
    assert by_id["null"].output is None
    assert by_id["missing"].output is UNSET


def test_literal_null_text_is_not_reinterpreted_as_json_null():
    literal = span("literal", span_type="TOOL", outputs="null")
    full = trace("tr-literal", [literal])
    client = FakeClient(pages=[FakePage([full])])

    normalized = next(
        MLflowTraceLoader(MLflowTraceConfig(experiment_name="agent-traces"), client=client)
        .load()
        .scan()
    )

    assert normalized.spans[0].output == "null"


def test_default_complete_trace_search_loads_ten_thousand_in_api_sized_pages():
    full = [trace(f"tr-{index:05d}", request_time=index) for index in range(10_000)]
    client = FakeClient(
        pages=[
            FakePage(
                full[start : start + 500],
                token=str(start + 500) if start + 500 < len(full) else None,
            )
            for start in range(0, len(full), 500)
        ],
    )

    snapshot = MLflowTraceLoader(
        MLflowTraceConfig(experiment_name="agent-traces"), client=client
    ).load()

    assert snapshot.trace_count == 10_000
    assert [call["max_results"] for call in client.search_calls] == [500] * 20
    assert all(call["include_spans"] is True for call in client.search_calls)


def test_unresolved_parents_are_detached_but_remain_visible_in_provenance_and_description():
    child = span("child", parent_id="outside", span_type="TOOL", inputs={}, outputs={"ok": True})
    full = trace("tr-partial", [child], metadata={"mlflow.trace.session": "session-1"})
    client = FakeClient(pages=[FakePage([full])])
    loader = MLflowTraceLoader(MLflowTraceConfig(experiment_name="partial-traces"), client=client)

    normalized = next(loader.load().scan())

    assert normalized.spans[0].parent_span_id is None
    assert normalized.spans[0].source_pointer["unresolved_parent_span_id"] == "outside"
    assert normalized.source_pointer["tracking_uri"] == "http://mlflow.test"
    assert normalized.spans[0].source_pointer["span_id"] == "child"
    assert loader.describe() == {
        "source": "mlflow:http://mlflow.test#experiment/17",
        "trace_count": 1,
        "call_count": 1,
        "steps_present": True,
        "steps_partially_present": False,
        "distinct_logical_cases": 1,
        "span_count": 1,
        "unresolved_parent_count": 1,
        "experiment_name": "partial-traces",
        "experiment_id": "17",
        "tracking_uri": "http://mlflow.test",
        "filter": None,
        "max_traces": 10_000,
    }


def test_unknown_provider_type_is_not_promoted_to_a_known_kind():
    custom = span("memory", span_type="MEMORY")
    full = trace("tr-custom", [custom])
    client = FakeClient(pages=[FakePage([full])])

    normalized = next(
        MLflowTraceLoader(MLflowTraceConfig(experiment_name="agent-traces"), client=client)
        .load()
        .scan()
    )

    assert normalized.spans[0].kind is SpanKind.UNKNOWN
    assert normalized.spans[0].subtype == "MEMORY"


def test_missing_experiment_has_an_actionable_error():
    client = FakeClient(experiment_id=None)

    with pytest.raises(MLflowTraceLoadError, match="MLflow experiment 'missing' was not found"):
        MLflowTraceLoader(MLflowTraceConfig(experiment_name="missing"), client=client).load()


@pytest.mark.parametrize("max_traces", [0, -1])
def test_live_config_max_traces_must_be_positive(max_traces):
    with pytest.raises(ValueError, match="max_traces must be at least 1"):
        MLflowTraceConfig(experiment_name="experiment", max_traces=max_traces)


@pytest.mark.parametrize("max_traces", [0, -1])
def test_file_config_max_traces_must_be_positive(tmp_path, max_traces):
    with pytest.raises(ValueError, match="max_traces must be at least 1"):
        MLflowFileTraceConfig(path=tmp_path / "traces.json", max_traces=max_traces)


@pytest.mark.parametrize("experiment_name", ["", "   "])
def test_live_config_experiment_name_must_not_be_blank(experiment_name):
    with pytest.raises(ValueError, match="experiment_name must not be empty"):
        MLflowTraceConfig(experiment_name=experiment_name)


def test_duplicate_trace_ids_from_search_are_rejected():
    client = FakeClient(
        pages=[FakePage([trace("tr-1", request_time=1), trace("tr-1", request_time=2)])],
    )

    with pytest.raises(MLflowTraceLoadError, match="duplicate trace id 'tr-1'"):
        MLflowTraceLoader(MLflowTraceConfig(experiment_name="experiment"), client=client).load()


def test_empty_search_page_with_a_continuation_token_is_rejected():
    client = FakeClient(pages=[FakePage([], token="unexpected")])

    with pytest.raises(MLflowTraceLoadError, match="empty page with a continuation token"):
        MLflowTraceLoader(MLflowTraceConfig(experiment_name="experiment"), client=client).load()


def test_provider_errors_are_reported_with_experiment_context():
    client = FakeClient()

    def fail_search(**kwargs):
        raise PermissionError("403 forbidden")

    client.search_traces = fail_search

    with pytest.raises(
        MLflowTraceLoadError,
        match="Failed to load MLflow experiment 'private': 403 forbidden",
    ) as raised:
        MLflowTraceLoader(MLflowTraceConfig(experiment_name="private"), client=client).load()

    assert isinstance(raised.value.__cause__, PermissionError)
