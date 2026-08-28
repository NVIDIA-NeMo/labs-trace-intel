# Analyst architecture

The Analyst has one required product: **Insights**. Its top-level architecture is deliberately
small:

```text
TraceLoader -> TraceSnapshot -> EvidenceStream(s) -> InsightsGeneration -> Insights
```

```mermaid
flowchart TB
    source["Structured traces<br/>S3, filesystem, or provider export"]
    loader["TraceLoader<br/>read, validate, normalize"]
    snapshot["TraceSnapshot<br/>normalized Trace + Span records"]

    stream1["EvidenceStream<br/>anomaly and pattern analysis"]
    stream2["EvidenceStream<br/>tool-issue detection"]
    streamN["EvidenceStream N<br/>another discovery technique"]

    context["Optional agent context<br/>ETHOS.md, goals, constraints"]
    generation["InsightsGeneration<br/>investigate, synthesize, and consolidate"]
    insights["Insights<br/>name, description, matching trace IDs"]

    source --> loader --> snapshot
    snapshot --> stream1 --> generation
    snapshot --> stream2 --> generation
    snapshot --> streamN --> generation
    snapshot --> generation
    context --> generation
    generation --> insights
```

`TraceLoader` is the input boundary, not a generic pipeline phase. Each loader owns the I/O,
validation, and normalization for one source and returns the local normalized contract. The
current `InsightTraceV1Loader` reads the repository's `insight-trace/v1` JSONL format. A future
LangSmith, OpenTelemetry, ATIF, or NeMo Platform loader would emit the same `Trace` values
directly; it would not translate through `insight-trace/v1` first.

## 1. TraceLoader and TraceSnapshot

```mermaid
flowchart TB
    input["Input<br/>provider-native, OTel, ATIF,<br/>NeMo Intake, or insight-trace/v1"]
    normalize["Concrete example<br/>InsightTraceV1Loader"]
    output["Output<br/>validated TraceSnapshot<br/>containing Trace + Span records"]
    input --> normalize --> output
```

The loader contract is intentionally one method:

```python
class TraceLoader(Protocol):
    def load(self) -> TraceSnapshot: ...
```

Source-specific configuration and diagnostics stay on the concrete loader. Evidence streams
never receive provider records or loader objects.

A `TraceSnapshot` is the frozen, run-scoped view of the normalized corpus. It gives every
evidence stream and `InsightsGeneration` the same traces, span trees, source pointers, tool
definitions, and corpus identity.

### Normalized Trace

The normalized model is a purpose-built input to the current evidence streams and Analyst. It
does not mirror a trace-store API. A trace contains one canonically ordered, parent-linked
array of spans plus the corpus-level fields IA2 and IA3 consume. Each span records one unit of
agent work, such as an LLM call, tool call, agent turn, chain, retriever, or evaluator.

Unlike NeMo Platform's storage/API models, this contract keeps `input` and `output` as JSON
values. An LLM span can therefore preserve complete input and output messages instead of
flattening them into text.

The contract is implemented as frozen Pydantic v2 models. Its essential shape is:

```python
class SpanKind(StrEnum):
    LLM = "LLM"
    TOOL = "TOOL"
    AGENT = "AGENT"
    CHAIN = "CHAIN"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    RERANKER = "RERANKER"
    EVALUATOR = "EVALUATOR"
    GUARDRAIL = "GUARDRAIL"
    UNKNOWN = "UNKNOWN"


class SpanStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class ToolCall(ContractModel):
    call_id: str | None = None
    index: int | None = None
    result_id: str | None = None
    result_count: int = 1
    instrumentation_alias_of: str | None = None
    prior_user_text: str | None = None
    returned_data: bool | UNSET = UNSET


class Span(ContractModel):
    span_id: str
    kind: SpanKind
    parent_span_id: str | None = None
    name: str | None = None
    subtype: str | None = None
    summary: str | None = None
    status: SpanStatus = SpanStatus.UNKNOWN
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_ms: float | None = None
    input: JsonValue | UNSET = UNSET
    output: JsonValue | UNSET = UNSET
    tool_name: str | None = None
    error_type: str | None = None
    tool_call: ToolCall | None = None
    source_pointer: dict[str, JsonValue] = Field(default_factory=dict)


class Trace(ContractModel):
    id: str
    spans: tuple[Span, ...]  # canonical topological-temporal order
    input: JsonValue | UNSET = UNSET
    cost_usd: float | None = None
    tool_catalog: dict[str, JsonValue | None] | None = None
    logical_case_id: str | None = None
    observed_verdict: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    complete_provenance_context: bool = False
    orphan_results: tuple[dict[str, JsonValue], ...] = ()
    source_pointer: dict[str, JsonValue] = Field(default_factory=dict)
```

`UNSET` is Pydantic's missing sentinel and is distinct from JSON `null`. For example, an absent tool-span `output` means no result
was observed, while `"output": null` means the tool returned `null`. IA3 depends on that
distinction.

Normalizers should populate span timestamps when the source records them. They are nullable
because a source may preserve authoritative order without wall-clock time; normalization does
not fabricate timestamps merely to satisfy the model.

`Trace.spans` represents both sequence and nesting:

- Array order is authoritative for evidence streams. Normalization produces one stable
  topological-temporal order: a parent always precedes its descendants; otherwise spans are
  ordered by `started_at`, with stable source order and then `span_id` breaking ties. Evidence
  streams do not independently re-sort spans.
- `parent_span_id` represents topology. A flat array can describe arbitrarily deep nesting,
  such as `AGENT -> CHAIN -> AGENT -> LLM`, without recursively embedding spans. Parent and
  child spans need not be adjacent in the array.
- Span IDs must be unique, every parent must resolve inside the same trace, and parent links
  must be acyclic. Root spans are the spans without a parent; a separate root ID would be
  redundant and would not represent traces with more than one top-level span.

Consumers that need a recursive view can derive `children(span_id)` or `span_tree()` from the
same array. That view is a convenience and is not a second serialized representation.

A normalized record can preserve LLM messages and tool activity in one span tree:

```json
{
  "id": "trace-123",
  "input": [{"role": "user", "content": "Find the latest report"}],
  "spans": [
    {
      "span_id": "agent-1",
      "parent_span_id": null,
      "kind": "AGENT",
      "status": "success",
      "name": "research-agent",
      "started_at": "2026-08-26T18:00:00Z",
      "ended_at": "2026-08-26T18:00:02Z"
    },
    {
      "span_id": "llm-1",
      "parent_span_id": "agent-1",
      "kind": "LLM",
      "status": "success",
      "name": "reason-and-select-tool",
      "started_at": "2026-08-26T18:00:00Z",
      "ended_at": "2026-08-26T18:00:01Z",
      "input": {
        "messages": [
          {"role": "system", "content": "Use tools when needed."},
          {"role": "user", "content": "Find the latest report"}
        ]
      },
      "output": {
        "messages": [
          {
            "role": "assistant",
            "content": "",
            "tool_calls": [
              {"id": "call-1", "name": "search", "arguments": {"query": "latest report"}}
            ]
          }
        ]
      }
    },
    {
      "span_id": "call-1",
      "parent_span_id": "agent-1",
      "kind": "TOOL",
      "status": "success",
      "name": "search",
      "tool_name": "search",
      "started_at": "2026-08-26T18:00:01Z",
      "ended_at": "2026-08-26T18:00:02Z",
      "input": {"query": "latest report"},
      "output": {"documents": ["report-2026-08"]}
    }
  ]
}
```

The contract deliberately contains only fields consumed by the current system:

- IA2 derives its ordered steps, tool calls, durations, cost, verdicts, and numeric features
  from spans and trace-level fields.
- IA3 derives `CallRecord` values from `TOOL` spans and uses the trace's tool catalog and
  provenance fields.
- Insights Generation can fetch the same structured trace and inspect its message, tool, and
  result payloads.

Provider, model, token, session, agent, and trace-summary fields are omitted until an evidence
stream actually consumes them. Future analysis should extend this contract from a concrete
requirement rather than copying a provider's storage model wholesale.

Concrete example: `InsightTraceV1Loader` validates `insight-trace/v1` records, maps their
steps and calls into normalized spans, and returns a snapshot. A future loader for NeMo
Platform would map its API objects into these local models; it would not return or import
NeMo Platform classes.

### Snapshot handoff

`TraceSnapshot` is a logical handle to one stable corpus, not a requirement to hold every
trace in memory. The first implementation exposes one access pattern: a re-iterable scan.

Conceptually:

```python
class TraceSnapshot(Protocol):
    source: str
    trace_count: int

    def scan(self) -> Iterator[Trace]:
        """Return a new iterator over the same canonical traces each time."""
```

Each call to `scan()` must start from the beginning and yield the same normalized traces in the
same order. It must not return a shared one-shot generator: multiple evidence streams need to
scan the same snapshot independently, and may eventually do so concurrently.

The first backing implementation is in memory. A JSONL or S3-backed implementation can be
added later if measurements show it is needed, without changing the evidence-stream interface.

Batch iteration and indexed `get_many(trace_ids)` lookup are deliberately deferred. They can be
added when scale measurements or Insights generation require them without changing the meaning
of `TraceSnapshot` or `scan()`.

## 2. EvidenceStream(s)

```mermaid
flowchart TB
    input["Input<br/>TraceSnapshot"]
    stream["Concrete example<br/>AnomalyAndPatternsEvidenceStream"]
    output["Output<br/>EvidenceStreamResult<br/>Problems + native artifacts"]
    input --> stream --> output
```

Evidence streams independently surface evidence that may support an Insight. Every stream:

- reads the same `TraceSnapshot`;
- can be enabled, disabled, and evaluated independently;
- does not read prior Insights or another stream's result;
- returns candidate `Problem` values with supporting trace IDs;
- may retain native artifacts for focused inspection and evaluation.

A stream may contain complex internal steps. IA2, for example, performs feature extraction,
anomaly detection, clustering, and recurring-pattern analysis. Those actions belong to IA2;
they do not become top-level Analyst stages.

```python
class EvidenceStream(Protocol):
    name: str

    def validate_configuration(self) -> None: ...

    def analyze(
        self,
        snapshot: TraceSnapshot,
    ) -> EvidenceStreamResult: ...


class Problem(ContractModel):
    description: str
    supporting_trace_ids: tuple[str, ...]


class EvidenceStreamResult(ContractModel):
    stream_name: str
    problems: tuple[Problem, ...]
    artifacts: Any = None
```

`Problem` is the common handoff to Insights generation. It says what may be wrong and which
normalized traces support investigating it. It does not claim root cause, impact, prevalence,
or that every matching trace has been found.

Native stream outputs remain available in `artifacts`, such as IA2's feature and clustering
results or IA3's findings and cards. They support diagnostics and evaluation but are not part
of the synthesis interface. Each stream owns the projection from its native analysis into
Problems, including its own evidence threshold.

The result does not carry generic coverage, status, version, or metrics fields. A stream that
cannot run raises an error; an empty `problems` tuple means it found nothing worth surfacing.
Stream-specific diagnostics remain in its typed artifacts. Input capability diagnostics, such
as IA3's per-rule coverage report, remain separate because they do not share a useful generic
shape across evidence streams.

Stable algorithm outputs are typed at the stream boundary as well. IA2 returns an
`AnomalyAndPatternsAnalysis` with typed anomalies and recurring-failure groups; IA3 returns
typed `ToolIssueCard` values with typed representative evidence. Temporary detector structures
can remain local dictionaries, but code outside the algorithm uses validated model attributes.

Each run explicitly registers its configured streams in an in-process
`EvidenceStreamRegistry`. Registration calls the stream's own
`validate_configuration()` method for runtime preflight checks such as provider credentials or
required dependencies. The registry rejects duplicate names and runs the selected streams
through the common interface; it does not discover imports or configure streams itself.

Each stream owns a typed configuration model that validates its algorithm settings. CLI or
service inputs are normalized into those models before stream construction; the registry never
receives untyped option mappings.

The Analyst collects the registered results and passes them directly to
`InsightsGeneration`. Collection is ordinary orchestration, not a separate merge or
composition phase.

## 3. InsightsGeneration

```mermaid
flowchart TB
    snapshot["Input<br/>TraceSnapshot"]
    evidence["Input<br/>all EvidenceStreamResults"]
    context["Optional input<br/>ETHOS.md and agent context"]
    generator["Concrete example<br/>LLM Analyst"]
    output["Output<br/>Insights"]

    snapshot --> generator
    evidence --> generator
    context --> generator
    generator --> output
```

`InsightsGeneration` is a required, concrete Analyst capability—not a generic postprocessor.
It uses the evidence streams as a map, inspects supporting normalized traces from the snapshot, and
authors customer-readable Insights. It may consolidate overlapping evidence from multiple
streams, but evidence streams themselves remain independent.

The implementation is an explicit stage object:

```python
generation = InsightsGeneration(
    snapshot=snapshot,
    evidence=stream_results,
    agent=agent_name,
    corpus=corpus_description,
)
result = generation.generate()
insights = result.insights
```

Concrete example: the existing Analyst receives Problems from IA2 and IA3, fetches their
supporting canonical traces, and produces Insights through a versioned LLM prompt. It has no
dependency on either stream's native artifact type. The implementation and its prompt live
together under `insights_generation/`.

Historical reconciliation, lifecycle management, and comprehensive trace categorization can
be implemented inside this concrete capability as they are added. They do not require generic
postprocessor abstractions or additional top-level boxes.

## 4. Insights

Insights are the product contract, not an optional artifact. A successful run always emits an
Insights collection; an empty collection means the Analyst found nothing worth surfacing.

The current repository emits JSON objects with this shape:

```python
class Insight(TypedDict):
    name: str
    description: str
    trace_ids: list[str]
```

Stable identity, status, prevalence, and user decisions can extend this contract when insight
lifecycle management is implemented. The core requirement remains that every Insight is an
understandable claim grounded in exact traces.

## Running the Analyst

The built-in run configures one trace loader, loads one snapshot, runs the current evidence
streams, and invokes Insights generation:

```bash
uv run insight-agent run-all traces.jsonl -o out
```

Focused engine commands remain useful for development, regression testing, and ablation:

```bash
uv run insight-agent run-ia2 traces.jsonl -o out/ia2
uv run insight-agent run-ia3 traces.jsonl -o out/ia3
uv run insight-agent run-all traces.jsonl -o out/evidence-only --no-analyst
```

The architecture does not require YAML, import-time plugin discovery, or a general-purpose
workflow engine. A future `analyze` command may provide a clearer product entry point, but it
should construct this concrete Analyst directly rather than expose its stages as a pipeline
language.
