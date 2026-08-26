# Analyst first-phase implementation plan

## Objective

Land one reviewable PR that establishes the smallest version of the target Analyst
architecture:

```text
TraceSnapshot -> EvidenceStream(s) -> InsightsGeneration -> Insights
```

This phase introduces the richer normalized `Trace` and `Span` contracts, exposes traces
through one re-iterable in-memory `TraceSnapshot.scan()`, and runs IA2 and IA3 against
deterministic projections of that new shape.

IA2 and IA3 algorithms, thresholds, findings, digests, cards, and artifacts must behave as
they did before this refactor. The change is their input boundary, not their analysis.

The detailed target contracts are specified in
[pipeline-architecture.md](pipeline-architecture.md).

## Scope

### In scope

- Define clean, local `Trace`, `Span`, `SpanKind`, `SpanStatus`, and `UNSET`
  contracts modeled after NeMo Platform Intake's trace/span structure.
- Preserve structured JSON input and output, including complete LLM messages.
- Represent span topology with a flat, canonically ordered array and `parent_span_id`.
- Define `TraceSnapshot` as an immutable in-memory corpus with a re-iterable `scan()`.
- Define the minimal `EvidenceStream` and evidence-result handoffs needed by the current
  Analyst.
- Adapt IA2 to project the new normalized traces into its existing `NormalizedTrace`
  input and run unchanged.
- Adapt IA3 to project the new normalized traces into its existing `TraceRecord` input
  and run unchanged.
- Keep the existing Insights authoring path working with the same IA2 digest, IA3 cards,
  and supporting traces.
- Remove or bypass generic preprocessor/postprocessor abstractions that conflict with the
  concrete architecture.
- Preserve focused CLI behavior and deterministic output artifacts.

### Out of scope

- Normalization from additional trace shapes or providers.
- A NeMo Platform import, SDK dependency, API dependency, or shared model package.
- TraceSnapshot readers other than the initial in-memory, re-iterable `scan()`.
- S3 readers, file-backed lazy readers, batching, paging, caching, or indexed lookup.
- New IA2 or IA3 algorithms, features, rules, thresholds, or semantics.
- New evidence streams.
- Insight reconciliation, persistence, lifecycle management, or comprehensive
  categorization.
- A generic workflow engine, YAML pipeline language, component registry, DAG, or plugin
  system.
- The 100,000-trace performance target. This phase must avoid obvious regressions but does
  not claim that target.

## Architectural decisions

### 1. The normalized unit is Trace

`Trace` is an end-to-end agent run with summary fields and an ordered collection of
`Span` values. A span represents one unit of work such as an LLM invocation, tool call,
agent, chain, retriever, evaluator, or guardrail.

The new Pydantic v2 models live in this repository. They may resemble NeMo Platform's public
trace and span shapes, but no NeMo code is imported. Pydantic owns field validation,
construction, serialization, and JSON Schema generation; handwritten validation is limited
to cross-span graph and ordering invariants.

The contract must preserve:

- Trace and span identity.
- Structured root, LLM, and tool input/output.
- Parent-child topology.
- Canonical span order.
- Timing and status.
- Model, provider, tool, token, cost, and error data.
- Tool schemas and evaluation context when present.
- Source provenance and implementation-specific attributes.
- The difference between an absent value and an observed JSON `null`.

Cross-cutting semantic fields belong on `Trace` or `Span`. Typed `ToolCall` metadata carries
the invocation identity and provenance used by tool-analysis streams.

### 2. Span order is normalized once

`Trace.spans` is one flat authoritative array.

Normalization establishes a stable topological-temporal order:

1. Parents precede descendants.
2. Otherwise `started_at` determines order.
3. Stable source order breaks timestamp ties when available.
4. `span_id` is the final deterministic tie-breaker.

Evidence streams consume this order and never reconstruct it independently.
`parent_span_id` retains arbitrarily deep topology, and a tree view can be derived when
needed without becoming a second serialized representation.

### 3. TraceSnapshot initially has one read path

The first implementation is intentionally concrete:

```python
class TraceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    snapshot_id: str
    source: str
    traces: tuple[Trace, ...] = Field(repr=False, exclude=True)

    @property
    def trace_count(self) -> int:
        return len(self.traces)

    def scan(self) -> Iterator[Trace]:
        return iter(self.traces)
```

Every call to `scan()` returns a fresh iterator over the same trace objects in the same
order. Two evidence streams can scan independently without sharing iterator state.

Do not add a storage protocol or reader hierarchy before a second implementation exists.
When a file or S3-backed implementation is required, the public contract can be extracted
from this proven behavior.

### 4. Evidence streams own projections into their engines

The public boundary is the richer snapshot:

```python
class EvidenceStream(Protocol):
    name: str
    version: str

    def analyze(self, snapshot: TraceSnapshot) -> EvidenceStreamResult: ...
```

IA2 and IA3 retain their current internal dataclasses. Each wrapper owns one deterministic
projection:

```text
Trace -> IA2 NormalizedTrace -> existing run_ia2()
Trace -> IA3 TraceRecord     -> existing detect() / build_cards()
```

This avoids coupling the reusable normalized trace contract to either research engine and
keeps engine parity directly testable.

### 5. InsightsGeneration remains concrete

This phase does not create a generic postprocessor API. The existing Analyst remains the
concrete Insights generator:

```python
insights = generate_insights(
    snapshot=snapshot,
    evidence=stream_results,
    agent_context=agent_context,
)
```

It continues to consume IA2's digest and IA3's recurrence-qualified cards and may scan the
snapshot for supporting traces. Reconciliation and persistence remain future internal
capabilities of the Analyst rather than new top-level abstractions.

## Repository shape

Keep the implementation small:

```text
src/insight_agent/
├── traces.py       # Trace, Span, enums, UNSET, TraceSnapshot, validation
├── streams.py      # EvidenceStream contract and IA2/IA3 wrappers
├── ia2_pipeline.py # Existing IA2 engine, behavior unchanged
├── ia3_tid.py      # Existing IA3 engine, behavior unchanged
└── analyst.py      # Existing Insights generation
```

Do not build a second framework around these files. Direct orchestration is sufficient:

```python
snapshot = TraceSnapshot.from_traces(traces, source=source)
stream_results = tuple(stream.analyze(snapshot) for stream in streams)
insights = generate_insights(snapshot, stream_results, agent_context)
```

The collection of stream results is ordinary orchestration, not an `EvidenceBundle` phase
or merge engine.

## Input normalization during this phase

The new core starts at `TraceSnapshot`; provider normalization is not part of the phase.

The repository's `insight-trace/v1` records are normalized once at the loader boundary. The
normalizer must:

- return the public `Trace` model rather than an engine input;
- preserve existing missing/null, tool-catalog, logical-case, source-pointer, and verdict
  semantics;
- introduce no provider-specific behavior;
- remain the only record-to-`Trace` conversion path.

No OpenAI, Anthropic, OTel, ATIF, LangSmith, MLflow, or NeMo Intake normalization work is
added in this phase.

## IA2 migration

Add one pure projection:

```python
def to_ia2_trace(trace: Trace) -> NormalizedTrace: ...
```

The projection must preserve the inputs IA2 currently observes:

| New trace data | IA2 input |
|---|---|
| `Trace.id` | `NormalizedTrace.trace_id` |
| Canonically ordered spans | `NormalizedStep` sequence |
| `TOOL` spans | `NormalizedCall` sequence |
| Tool-span input | call arguments |
| Tool-span output | call result |
| Span timing | call duration |
| Span/trace provenance | source pointers |
| Trace outcome attributes | observed verdict |
| Trace cost and numeric attributes | cost and metrics |

Current IA2 code treats an unobserved result and JSON `null` identically; the projection
may preserve that behavior even though the richer normalized model retains the distinction
for other consumers.

`IA2EvidenceStream` scans the snapshot, projects traces, and invokes `run_ia2()`. Do not
split IA2's anomaly, clustering, failure, or verdict analysis into separate streams in this
phase.

Parity requirements:

- Feature rows are equivalent.
- Anomaly scores, flags, and reasons are equivalent.
- Trajectory, verdict, failure, and cross-tool groups are equivalent.
- Failure events are equivalent.
- `digest.md` is byte-identical after excluding intentionally changed provenance paths.

## IA3 migration

Add one pure projection:

```python
def to_ia3_trace(trace: Trace) -> TraceRecord: ...
```

The projection must preserve:

| New trace data | IA3 input |
|---|---|
| `Trace.id` | `TraceRecord.trace_id` |
| Logical/evaluation case identity | `logical_case_id` |
| `Trace.tool_catalog` | tool catalog |
| Canonically ordered `TOOL` spans | call sequence |
| Tool-span ID | call ID |
| Tool name | tool name |
| Tool input | arguments |
| Tool output | result |
| Absent tool output | the existing IA3 `MISSING` singleton |
| Explicit JSON `null` output | Python `None`, not `MISSING` |
| Error status/type | explicit error and outcome marker |
| Provenance attributes | source pointers and grounding context |
| Typed tool metadata | result IDs/counts, aliases, orphan results |

The `MISSING` mapping is load-bearing and must be tested by identity. Do not reconstruct
the sentinel.

`ToolIssueEvidenceStream` scans the snapshot, projects tool spans, and invokes the current
`detect()` and `build_cards()` functions unchanged.

Parity requirements:

- Every finding field and ordering is equivalent.
- Card grouping and IDs are equivalent.
- Independent-case counts and eligibility are equivalent.
- Representative evidence and source pointers are equivalent.
- Capability coverage and abstention reasons are equivalent.

## Commit sequence

The PR uses two reviewable commits. The boundary migration is atomic so no temporary dual
loader API or engine-specific conversion path is introduced.

### 1. `docs: define normalized analyst architecture`

- Finalize `pipeline-architecture.md`.
- Replace this implementation plan.
- Record scope and non-goals.
- Make the Trace/Span, ordering, snapshot, and evidence-stream boundaries unambiguous.

Gate:

```bash
git diff --check
```

### 2. `refactor: run IA2 and IA3 from normalized traces`

- Add the local Trace/Span models and enums.
- Add `UNSET` semantics.
- Add validation for unique IDs, root/parent references, cycles, canonical order, timestamps,
  and immutable attributes.
- Add the concrete in-memory `TraceSnapshot.scan()`.
- Normalize the current fixture format once at the loader boundary.
- Add `to_ia2_trace()`.
- Route `IA2EvidenceStream` through `TraceSnapshot.scan()`.
- Preserve the focused IA2 command and artifacts.
- Add `to_ia3_trace()`.
- Route `ToolIssueEvidenceStream` through `TraceSnapshot.scan()`.
- Preserve the focused IA3 command and artifacts.
- Keep the existing Analyst path functional using the new snapshot.
- Remove the old loader projections so every analysis path uses normalized traces.
- Do not modify IA2 or IA3 algorithms.

Gate:

- Unit tests cover deeply nested traces, invalid graphs, structured LLM messages, missing
  versus null, and independent repeated scans.
- Unit-test every IA2 and IA3 projection field.
- Compare IA2 and IA3 deterministic artifacts byte-for-byte with the base commit.
- Run the Analyst dry-run and verify its evidence prompt remains equivalent.
- Run the bundled corpus and Tau-bench end to end.
- Run the full suite, lint, build, and diff checks:

```bash
uv run pytest -q
uv run ruff check .
uv build
git diff --check
```

## Regression strategy

### Freeze the baseline

Before code changes:

- Run the full suite on the rebased branch.
- Save the current synthetic-corpus IA2 digest and IA3 findings as comparison inputs.
- Run Tau-bench evidence generation and record trace count, call count, anomalies, groups,
  findings, cards, runtime, and peak memory.
- Save an Analyst dry-run prompt.

Do not hard-code stale counts into this plan. Record the values produced by the exact base
commit in the PR description.

### Contract tests

Test:

- Structured LLM messages remain structured.
- Span order is deterministic and parents precede descendants.
- Deep nesting is reconstructable from `parent_span_id`.
- Trace and span attributes cannot be mutated through frozen models.
- `UNSET`, JSON null, empty objects, empty arrays, and empty strings remain distinct.
- Repeated snapshot scans are complete, stable, and independent.

### Engine parity tests

For every shared fixture, produce inputs through `TraceSnapshot.scan()` and the relevant
projection, run the analysis engine, and compare deterministic artifacts byte-for-byte to the
baseline captured from the base commit. Projection unit tests cover every consumed field.

### End-to-end checks

Validate:

- Focused IA2 execution.
- Focused IA3 execution.
- Evidence-only execution with both streams.
- Analyst dry-run.
- The bundled synthetic corpus.
- Tau-bench.

An empty result, explicit abstention, and execution failure must remain distinct. Errors must
identify the failing stream and must never be converted into successful empty evidence.

## Acceptance criteria

The phase is complete when:

- The repository has one documented local normalized Trace/Span model.
- The model preserves full LLM and tool input/output plus nested span topology.
- `TraceSnapshot.scan()` is the only new read API and is deterministic and re-iterable.
- IA2 and IA3 consume the new shape only through explicit projection functions.
- IA2 and IA3 algorithms remain unchanged.
- Existing deterministic IA2 and IA3 outputs are equivalent on shared fixtures.
- Focused IA2/IA3 commands and the current Analyst dry-run still work.
- The core architecture has no required preprocessor, generic postprocessor, YAML workflow,
  DAG, registry, or NeMo Platform dependency.
- All tests, lint checks, builds, and diff checks pass.
- The commit history cleanly separates the architecture decision from its atomic migration.
- No additional provider normalization or TraceSnapshot reader is introduced.
