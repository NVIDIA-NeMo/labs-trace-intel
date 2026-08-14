---
name: insight-trace-adapter
description: Write, debug, or verify an adapter that converts agent traces into Insight Agent canonical JSONL (insight-trace/v1) for the IA2/IA3 pipelines. Use when the user says "write an adapter", "onboard a new venue", "onboard a new dataset", "convert my traces", "my traces are in <some format>", "get my logs into Insight Agent", "insight-agent validate is failing", "why aren't any TID rules firing", or "why is my corpus not clustering". Covers the required-vs-optional field contract, the missing-result sentinel, tool-catalog wiring, and the validate/coverage iteration loop.
---

# Writing an Insight Agent trace adapter

## What you are producing

One JSON object per line. One line per complete trace. That is the entire
deliverable.

You are **not** writing Python against the engines' dataclasses. Emit JSONL in
whatever language suits the source data; `insight_agent.loader` fans each
record out into IA2's `NormalizedTrace` and IA3's `TraceRecord`. If you find
yourself importing `NormalizedCall` or `CallRecord`, stop — you are working at
the wrong layer.

Only **four top-level and four per-call fields are required**:

```json
{"schema_version": "insight-trace/v1", "trace_id": "run-0001", "calls": [{"call_id": "c0", "call_index": 0, "tool_name": "Search", "arguments": {"query": "x"}}]}
```

That validates. Everything else is optional and unlocks specific capability.

## The verify loop

Run this every iteration. Do not batch several changes and validate once — the
whole reason this loop exists is that adapter defects are silent downstream.

1. **`insight-agent schema`** — read the contract. Field descriptions state what
   omitting each field costs.
2. **Inspect 2–3 raw source records first**, and solve the join (below) before
   writing anything.
3. **Write the adapter for required fields only.** Resist adding more yet.
4. **`insight-agent validate out.jsonl`** — iterate until it exits 0.
5. **`insight-agent coverage out.jsonl`** — read which rules abstain and why.
6. **Add exactly one optional field**, then repeat 4–5. One at a time, so you
   can attribute any change in output to the field you just added.
7. **`insight-agent run-ia3 out.jsonl -o out` and `run-ia2 out.jsonl -o out`**,
   then open `out/ia3/findings.json` (per-call findings; the field you want is
   `issue_type`), `out/ia3/cards.json` (recurrence-qualified) and
   `out/ia2/digest.md`, and **manually check three findings against the raw
   source**.

Step 7 is not optional. A rule that fires on 100% of calls is an adapter bug,
not a discovery — and **intermediate iterations are expected to show exactly
that**. Seeing `malformed_tool_call` or `missing_tool_result` on every call in
round one is the loop working, not a disaster.

**Ablation beats inspection.** When you are unsure whether a field is helping
or lying, remove it, rerun `run-ia3`, and diff the findings. Adding a field
should *change* findings in ways you can explain; if removing a field makes
findings appear, the field was suppressing them. This is the only reliable way
to diagnose several of the traps below, whose stated symptoms can be masked by
a single well-formed call.

## Solving the join

This is the actual work; everything else is renaming.

A trace has a list of calls, and each call may have a result. In almost every
source format the call and its result are **not in the same object**. Find the
id that connects them, then use two passes:

1. **Pass one — index the results.** Walk the whole trace and build
   `{result_id: result}`. Count duplicates while you are there; two results for
   one id becomes `result_count: 2`.
2. **Pass two — emit the calls.** Walk again in order. For each call, look up
   its id in the index. Found: attach the result. **Not found: emit no
   `result` key at all** — that is what makes `missing_tool_result` fire.
3. **Anything left over in the index** — results whose id matched no call — goes
   into `orphan_results`.

Where the ids live, by format family:

| Source shape | Call side | Result side |
|---|---|---|
| Anthropic messages | `tool_use.id` | `tool_result.tool_use_id` |
| OpenAI messages | `tool_calls[].id` | `role:"tool"` message's `tool_call_id` |
| OTel / tracing spans | often the *same span* carries both | (no join needed) |
| Log streams | a call event's id | a later result event's ref |

If your format needs no join because call and result share one object, say so
in a comment and move on — but still handle the absent-result case explicitly.

## What to add, in order

Each row unlocks the capability on the right. Work down the list.

| Add | Unlocks |
|---|---|
| `tool_catalog` | **6 rules at once** — every argument-contract check |
| `result` present/absent discipline | `missing_tool_result`, `explicit_tool_failure` |
| `logical_case_id` | Honest card eligibility (see traps) |
| `source_pointer` | Evidence the Analyst can reopen |
| `steps` | Real IA2 trajectory tokens |
| `observed_verdict`, `cost` | IA2 verdict grouping |
| `outcome_marker` (with `explicit_error: true`) | Names the failure mechanism, which becomes the card identity |
| `complete_provenance_context` + `prior_user_text` | `explicitly_rejected_ungrounded_identifier` |
| `duration_ms`, `metrics` | Richer anomaly features |

`tool_catalog` is almost always the highest-value thing you can add, and it is
usually already in the export. There is no cross-vendor convention for where —
`tools`, `toolDefinitions`, `functions`, `available_tools`, or a sidecar file
are all common, and tracing formats such as OTLP have no standard slot at all.
Search the whole export for tool names before concluding you do not have them.

It often lives **once at the top** rather than on each record, so thread it
through as corpus-level context (the bundled template shows this) or supply it
at run time with `--tool-catalog`.

One row deserves expansion:

**`outcome_marker`** is only read when you also set `explicit_error: true`. It
names *why* the call failed and becomes the card's `mechanism_key`, so it is
the card's identity. Use a short stable slug from the source —
`connection_timeout`, `permission_denied`, an OTel `error.type`. Engine-derived
markers use `lower_snake_case` (`error_prefix`, `no_active_session`,
`python_traceback`); match that so your markers do not look like a second
naming scheme in the same findings file. Do not invent one when the source has
no failure taxonomy — omit it and the engine will derive a marker itself.

## The seven traps

Each one validates cleanly and fails silently. The symptom is what you would
otherwise observe and misdiagnose.

**1. Omitting `result` ≠ `"result": null`.**
Absence means no result was ever recorded. `null` means the tool returned null.
*Symptom:* zero `missing_tool_result` findings on a corpus you know drops
results. *Fix:* omit the key entirely, or set `result_missing: true`.

**2. `explicit_error: false` disables all text decoding for that call.**
It is an authoritative success assertion, not a default. Setting it on calls
that merely lack an error flag marks them succeeded, so their result text is
never examined.

*Blast radius:* wider than it looks. Beyond `explicit_tool_failure`, every rule
that depends on a call having *failed* goes quiet too —
`unresolved_placeholder_argument`, `repeated_identical_failed_call`,
`modified_retry_same_failure`.

*Do not diagnose this by looking for zero `explicit_tool_failure`.* One call
with a genuine `true` flag keeps that count non-zero while everything else is
silently suppressed. Diagnose it by ablation instead: rerun with the field
omitted entirely and diff the findings. If removing it *adds* findings, you
were over-asserting success.

*Fix:* only set it when the source carries a real success flag. Otherwise omit.
*Three-valued sources* (OTel `UNSET`/`OK`/`ERROR`, HTTP-ish statuses): map the
error state to `true`, and map **both** the success and the unset state to
*omitted*. Do not map "OK" to `false` — many instrumentations set it
reflexively.

**3. Textual results go under `content`, never `output`.**
IA3 unwraps `content` and nothing else; IA2 unwraps five keys.
*Symptom:* IA2 reports failures IA3 never saw. *Check:*
`insight-agent explain-failures out.jsonl --only-disagreements`.

**4. Do not populate `result_id` from the wrong id namespace.**
It should only carry a genuinely independent result→call reference.
*Symptom:* `call_result_id_mismatch` on ~100% of calls. *Fix:* omit it.

**5. Synthesize unique `call_id`s.**
Use `"{trace_id}#{call_index}"` when the source has none.
*Symptom:* `duplicate_call_id` on every call.

**6. Leave `arguments` as the agent actually emitted them.**
A non-object value is legitimate evidence and produces `malformed_tool_call`.
Do not stringify an object, and do not repair a broken one.

The judgement call is *whether the string is the agent's output or the
transport's encoding*:

- **The string is the agent's output** (OpenAI `function.arguments`, raw model
  text) → try to parse; **if it does not parse, pass the raw string through**.
  The parse failure is the finding.
- **The transport forces strings** (OTel span attributes, protobuf scalars,
  CSV columns, most tracing backends) → **parse it**. Every value is a string
  there, so "leave it raw" would report every single call as malformed.

*Symptom of getting it wrong either way:* `malformed_tool_call` on ~100% of
calls (you failed to parse a transport encoding), or real malformed calls
missing entirely (you repaired or dropped them).

**7. `complete_provenance_context: true` is a strong claim.**
It asserts this record captured *every* user message and *every* prior tool
result. It is **per record, not per corpus** — set it on the traces where it
holds and omit it on the rest — unlike `steps`, varying it across a corpus is
correct and expected, because completeness is a property of each capture.

It is false whether the loss came from your adapter (dropped system messages,
truncated results) *or* from the instrumentation itself: if any prior result is
unrecorded, an identifier that looks ungrounded may simply have come from the
result you do not have. A usable test: *for this trace, can I reconstruct every
user message and every tool result the agent saw before the call in question?*
If a call in the trace has no recorded result, the answer is no. When in doubt,
omit it — the cost is one abstaining rule, and the cost of being wrong is
findings that accuse the agent of inventing identifiers it was actually given.

*Symptom:* findings claiming the agent hallucinated identifiers that were
present all along.

Full detail with before/after JSON: [reference/traps.md](reference/traps.md).

## Two things with no single right answer

The skill cannot decide these for you, but you should decide them *explicitly*
and record the decision in a comment, because they change the output.

**Which source events become `steps`.** Emitting `steps` changes IA2's
`trajectory_step_count` and its whole clustering token stream, so two adapters
over the same corpus that disagree here produce incomparable results. The
convention that keeps things comparable:

- one `tool` step per call, in order — always
- `user` steps for user turns, `agent` steps for model reasoning text
- one `evaluation` step for a terminal verdict, if the source has one
- **do not** emit a step for the trace envelope itself (a root span, a
  conversation wrapper). It is not an action the agent took.

**Whether to emit `steps` at all.** If the source has no step model beyond the
calls, omitting `steps` is fine and honest: IA2 falls back to one token per
call. Just be consistent across
the whole corpus, and never mix — half the traces with steps and half without
is the one genuinely broken option.

## Worked example

`src/insight_agent/adapters/messages.py` is a real, tested adapter for
Anthropic and OpenAI message lists. Everything you need is already in this
document — read that file if you want to see the two-pass join and the traps
handled in working code, not because the skill is incomplete without it. Its
tests in `tests/test_adapters.py` are written as a specification: each test
names a decision and what breaks otherwise.

Calibrate your expectations for real data: production exports routinely carry
no tool catalog, never set `logical_case_id`, and assert `explicit_error:
false` corpus-wide because the source had no error channel to read. Partial
coverage is the norm, not a sign you did it wrong — `insight-agent coverage`
names each gap, and the honest move is to report it rather than to synthesise
the missing field.

To scaffold your own:

```bash
insight-agent init-adapter mysource            # -> ./adapters/mysource.py
insight-agent init-adapter mysource --dir src/myproj/adapters
```

It writes one standalone script relative to your current directory, refuses to
overwrite without `--force`, and prints the verify loop. The script does not
need to live inside any package — it is run directly and writes JSONL to stdout.

## Acceptance checklist

- [ ] `insight-agent validate out.jsonl` exits 0
- [ ] `insight-agent coverage out.jsonl` lists the rules you intended to support
- [ ] `insight-agent explain-failures out.jsonl --only-disagreements` is empty
- [ ] No rule fires on ~100% of calls
- [ ] ≥ 3 traces for clustering to run at all — but IA2's anomaly stage is
      degenerate at that size (`contamination=0.02` expects 0.06 flags on 3
      traces). For a meaningful IA2 run you want tens of traces, or
      `--contamination 0.15` on a small sample
- [ ] ≥ 3 distinct `logical_case_id`s **that hit the same issue type** if you
      want an eligible card (see below)
- [ ] Three findings traced back to the raw source **by hand**
- [ ] A test asserting the adapter's output validates

### Zero eligible cards is often correct

Card eligibility needs three independent logical cases exhibiting **the same
issue type** — not three cases in the corpus. A small corpus of unrelated tasks
will legitimately produce zero eligible cards even with `logical_case_id`
perfectly populated, because nothing recurred yet. That is the precision gate
doing its job.

`run-ia3` prints a hint about checking `logical_case_id` when no card
qualifies. Ignore it if `coverage` already shows the field populated — and
never merge distinct tasks under one case id to clear the threshold. Recurrence
you manufactured is not recurrence.

### "Evaluable" is not "will fire"

`coverage` reporting 15/19 evaluable and IA3 reporting 6 distinct issue types
is not a contradiction. Evaluable means *the evidence needed to check this rule
is present*; firing means *the defect actually occurred*. A healthy corpus has
many evaluable rules that never fire. Only worry about the gap when a rule you
expected to fire is listed as abstaining.

Put adapter tests next to the adapter. If the repo's `pyproject.toml` sets
`testpaths`, a bare `pytest` will not collect them — run
`pytest path/to/your/tests` explicitly, or add your directory to `testpaths`.
When a test shells out to the CLI, resolve it from the running interpreter
(`Path(sys.executable).parent / "insight-agent"`) rather than assuming `PATH`.

## When to stop

**Abstention is designed behaviour.** If `coverage` says a rule abstains
because the source genuinely lacks that evidence, that is the correct outcome.

Do not fabricate `tool_catalog` entries, invent `outcome_marker` values, guess
`logical_case_id`s, or set `complete_provenance_context: true` to make more
rules fire. A detector that abstains is telling the truth; one that fires on
invented evidence is worse than useless, because someone will act on it.

## Reference

- [reference/canonical-schema.md](reference/canonical-schema.md) — every field
- [reference/capability-matrix.md](reference/capability-matrix.md) — what each
  field unlocks, which rules abstain without it, and the exact source line
- [reference/traps.md](reference/traps.md) — the seven traps, in full
- [assets/adapter_template.py](assets/adapter_template.py) — runnable skeleton
