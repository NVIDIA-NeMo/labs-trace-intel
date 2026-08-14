# Examples

## `tau_bench_traces.jsonl`

200 real agent traces (1,003 tool calls) from [τ-bench](https://github.com/sierra-research/tau-bench),
the public tool-agent benchmark, already converted to canonical
`insight-trace/v1`. Nothing to adapt — point the CLI at it and run:

```bash
insight-agent validate examples/tau_bench_traces.jsonl
insight-agent coverage examples/tau_bench_traces.jsonl
insight-agent run-all  examples/tau_bench_traces.jsonl -o out
open out/index.md
```

The agent under test is a customer-service assistant working against stateful
tools. All customer names, addresses and order IDs are τ-bench's own synthetic
fixtures — there is no real user data here.

| | |
|---|---|
| Traces | 200 (139 telecom, 51 retail, 10 airline) |
| Tool calls | 1,003 |
| Distinct `logical_case_id`s | 153 |
| Tools in catalog | 13, with runtime-recovered schemas |

### What it produces

```
IA3 rules evaluable : 14/19
findings            : 64
cards               : 3, all 3 eligible for the Analyst
```

| Card | Findings | Independent cases |
|---|---:|---:|
| `explicit_tool_failure:tau_result_error` | 52 | 26 |
| `modified_retry_same_failure:tau_result_error` | 6 | 6 |
| `unknown_tool:unknown_tool` | 6 | 4 |

This is the useful part: unlike a small or synthetic corpus, enough recurs here
that all three cards clear the three-independent-case gate, so the Analyst
stage has real evidence to author from.

`modified_retry_same_failure` is the one worth reading by hand — the agent
changes an argument and re-issues a call that fails the same way, which is a
pattern no single trace reveals.

### Why five rules abstain

Not a defect. The τ-bench export genuinely lacks the evidence:

| Abstaining rule | Missing evidence |
|---|---|
| `missing_tool_result` | every call carries a result; the harness never drops one |
| `duplicate_tool_result` | no call sets `result_count > 1` |
| `orphan_tool_result` | no orphaned results captured |
| `mapped_instrumentation_alias` | no instrumentation aliasing in the source |
| `explicitly_rejected_ungrounded_identifier` | `complete_provenance_context` is not asserted |

Reading that list is the point of `coverage`. Compare it against the bundled
sample corpus (`insight-agent demo`), which is synthesised specifically to make
all nineteen rules fire.

### Known warnings

`validate` reports 14 `tool_catalog_disagreement` warnings. These are real: the
schemas were recovered from the runtime, and a few tools genuinely present a
different shape in different traces. They are left in rather than normalised
away, because that disagreement is itself evidence about the source.

### Provenance and two deliberate departures from the raw export

Converted from the τ-bench runtime state with the trace-level `source_pointer`
rewritten to `{"dataset": "tau-bench", ...}`. Two changes were made to the raw
export, both measured rather than assumed:

- **`explicit_error: false` was dropped** where the source asserted it
  corpus-wide (945 calls); the 58 genuine `true` values are kept. A blanket
  false is a success *assertion* that disables all text-based failure decoding
  — trap 2 in the bundled adapter skill. Ablation confirmed IA2 and IA3 output
  is byte-identical either way, so nothing is lost and ~180 spurious warnings
  go away.
- **Synthesised per-call `steps` were omitted.** τ-bench has no step model
  beyond the calls themselves, and the IA2 digest is identical with or without
  them, so inventing a trajectory would have added structure the source does
  not have.
