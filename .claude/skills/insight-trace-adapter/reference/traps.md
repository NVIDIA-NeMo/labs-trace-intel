# The seven traps

Every one of these produces a corpus that **validates cleanly and still yields
wrong results**. They are listed with the symptom you would otherwise observe
and misdiagnose as a detector problem.

---

## 1. Omitting `result` is not the same as `"result": null`

The one place in the format where *key absence* carries meaning.

IA3 uses a sentinel object compared by identity to distinguish "no result was
ever recorded" from "the tool returned null". The loader produces that sentinel
only when the key is absent.

```json
// no result was ever recorded -> fires missing_tool_result
{"call_id": "c0", "call_index": 0, "tool_name": "Read", "arguments": {"path": "/x"}}

// the tool genuinely returned null -> NOT a missing result
{"call_id": "c0", "call_index": 0, "tool_name": "Read", "arguments": {"path": "/x"}, "result": null}
```

**Symptom:** zero `missing_tool_result` findings on a corpus you know has
dropped results.

**Escape hatch:** some emitters (ORMs, protobuf→JSON bridges, `jq` pipelines)
cannot omit a key. Use `"result_missing": true` or `"result_count": 0` instead.

**Check:** `insight-agent coverage` reports how many calls are missing a result.
`0` on a corpus that should have some means this trap.

---

## 2. `explicit_error: false` disables all text-based failure decoding

`explicit_error` is an *authoritative* flag, not a hint. `true` forces failure;
`false` forces success and returns before any text is examined.

```json
// WRONG: asserts success on every non-flagged call
{"tool_name": "Search", "result": {"content": "Error: timed out"}, "explicit_error": false}
// -> no explicit_tool_failure, because the flag said it succeeded

// RIGHT: let the decoder read the result
{"tool_name": "Search", "result": {"content": "Error: timed out"}}
// -> explicit_tool_failure (error_prefix)
```

**Symptom:** zero `explicit_tool_failure` findings across an entire corpus.

**Rule:** only set `explicit_error` when the source carries a genuine success
flag (Anthropic's `is_error`, for instance). Otherwise omit it.

**Check:** the validator warns when `explicit_error: false` appears on >90% of
calls.

---

## 3. Textual results belong under `content`, never `output`

IA2 unwraps `content`, `output`, `message`, `error` and `summary`.
IA3 unwraps **`content` only**, and serialises everything else to JSON — and
its error-prefix regex is anchored to the start of the string, so a serialised
object never matches.

```json
// WRONG
{"result": {"output": "Error: connection refused"}}
// IA2: fires tool_output_error_prefix
// IA3: sees {"output":"Error: connection refused"} -> anchored regex fails -> nothing

// RIGHT
{"result": {"content": "Error: connection refused"}}
// both engines agree
```

You may keep the original keys alongside `content`:

```json
{"result": {"content": "Error: connection refused", "output": "Error: connection refused", "code": 111}}
```

**Symptom:** IA2 reports failures IA3 never saw; findings look mysteriously
sparse relative to the digest.

**Check:** `insight-agent explain-failures traces.jsonl --only-disagreements`
should print `No disagreements.` See [docs/failure-decoders.md](../../../../docs/failure-decoders.md).

---

## 4. Do not populate `result_id` from the wrong id namespace

`result_id` means "the call id this result *claims* to answer". It exists to
catch instrumentation corruption, so it fires whenever it differs from
`call_id`. Filling it from a message id, span id or row id makes every call
look corrupted.

```json
// WRONG: message id in a call-id field
{"call_id": "call_abc", "result_id": "msg_00917"}   // mismatch on 100% of calls

// RIGHT: omit unless the source has a real result->call reference
{"call_id": "call_abc"}
```

**Symptom:** `call_result_id_mismatch` on essentially every call.

**Check:** the validator warns when `result_id` differs from `call_id` on more
than half the calls.

---

## 5. Synthesize unique `call_id`s

`call_id` must be unique within the trace. When the source has no call ids,
derive one — do not leave it constant or empty.

```json
// WRONG
{"call_id": "call", "call_index": 0, ...}
{"call_id": "call", "call_index": 1, ...}   // duplicate_call_id on every call

// RIGHT
{"call_id": "run-0001#0", "call_index": 0, ...}
{"call_id": "run-0001#1", "call_index": 1, ...}
```

**Symptom:** `duplicate_call_id` on every call.

**Note:** if the source genuinely *does* reuse ids, keep them. That is real
instrumentation evidence and the rule is correctly reporting it. The validator
warns rather than errors precisely because it cannot tell the two apart.

---

## 6. Leave `arguments` raw

A non-object `arguments` value is legitimate evidence: it produces
`malformed_tool_call`, which is often exactly the defect worth finding.

```json
// The model emitted truncated JSON. Pass it through.
{"tool_name": "Search", "arguments": "{\"query\": \"api\", \"limit\": "}
// -> malformed_tool_call

// WRONG: dropping the call, or coercing to {} , hides the real defect
{"tool_name": "Search", "arguments": {}}
```

The question to answer is **whose string is it**.

*The agent's string* — OpenAI's `function.arguments`, raw model output. Parse
it when it parses; when it does not, pass the raw string through unchanged. The
parse failure is the evidence.

*The transport's string* — OTel span attributes, protobuf scalars, CSV columns.
These encodings cannot represent an object at all, so **everything** arrives as
a string and there is no way to tell a malformed blob from a well-formed one
that got serialised. Parse it. Treating these as raw reports every call in the
corpus as malformed:

```json
// OTel: attribute values must be scalars, so this is the transport's doing
{"key": "gen_ai.tool.call.arguments", "value": {"stringValue": "{\"query\": \"x\"}"}}
// -> parse to {"query": "x"}; do NOT pass the string through
```

Do not stringify an object that was already parsed either — that fires
`malformed_tool_call` on healthy calls.

**Symptom:** `malformed_tool_call` on ~100% of calls (transport encoding not
parsed), or real malformed calls vanishing (over-eager repair).

---

## 7. `complete_provenance_context: true` is a strong claim

It asserts the record contains **every** user message and **every** prior tool
result for the trace. Only then can IA3 conclude an identifier was *invented*
rather than merely *not captured*.

```json
{
  "complete_provenance_context": true,
  "task_text": "Open the session document and append the note.",
  "calls": [{
    "tool_name": "SessionTool",
    "arguments": {"document_id": "DOC-99213"},
    "result": {"content": "Error: invalid document DOC-99213: not found"}
  }]
}
// -> explicitly_rejected_ungrounded_identifier: DOC-99213 appears nowhere in context
```

If your adapter drops system messages, truncates long results, or samples the
conversation, leave this `false`.

**Symptom:** findings claiming the agent hallucinated identifiers that were
actually present in a message you did not capture.

---

## Bonus: `logical_case_id` and card eligibility

Not a correctness trap, but the most common way to overstate results.

IA3 promotes a card to `eligible_for_analyst` only after **three independent
logical cases**. Without `logical_case_id`, every trace counts as its own case
— so 500 retries of one failing task become 500 "independent" cases and the
precision gate stops meaning anything.

```json
{"trace_id": "run-0001", "logical_case_id": "case-export-042"}
{"trace_id": "run-0002", "logical_case_id": "case-export-042"}   // same task, retried
{"trace_id": "run-0003", "logical_case_id": "case-import-118"}
```

**Check:** `insight-agent coverage` reports the distinct logical case count and
warns when no record sets the field at all.

Note the display: with the field absent, coverage still prints a case count,
because it defaults to `trace_id`. It marks that case explicitly —

```
3 traces, 6 calls, 9 steps, 3 distinct logical cases (defaulted from trace_id)
```

— so `(defaulted from trace_id)` means you have not set it yet. Once you do,
coverage cannot tell whether a 1:1 ratio is wrong: three traces of three
genuinely different tasks legitimately gives three cases. Only you know whether
your corpus contains retries of the same task.
