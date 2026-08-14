# The two failure decoders

IA2 and IA3 were developed independently, and each has its own idea of what
counts as a tool failure. The canonical trace format does **not** unify them,
because unifying them would mean changing measured algorithm behaviour. Instead
the divergence is documented here and made observable with
`insight-agent explain-failures`.

This matters to anyone writing an adapter, because a result payload that reads
as a failure in one engine can read as a success in the other.

## Where each decoder lives

| | Function | File |
|---|---|---|
| IA2 | `decode_explicit_failure` | `src/insight_agent/ia2_pipeline.py` |
| IA3 | `strict_failure` | `src/insight_agent/ia3_tid.py` |

## What each one accepts

| Evidence in the result | IA2 | IA3 |
|---|:---:|:---:|
| `isError: true` | yes | yes |
| `is_error: true` | yes | **no** |
| `success: false` | yes | yes |
| `ok: false` | yes | **no** |
| `called: false` | **no** | yes |
| non-empty `error` field | **no** | yes |
| `status: "error" / "failed" / "failure"` | yes | yes |
| `status: "crashed" / "timeout"` | yes | **no** |
| `exit_code` / `returncode` != 0 | yes | **no** |
| `"Process exited with code N"` in text | **no** | yes |
| nested `result.isError` / `result.success` | **no** | yes |
| text starting with `Error:` / `Failed:` | yes | yes |
| Python traceback (code-execution tools only) | yes | yes |
| `explicit_error` field | not read | authoritative |

## The text-extraction trap

This is the one that actually bites.

| | Keys unwrapped from a result object |
|---|---|
| IA2 | `content`, `output`, `message`, `error`, `summary` |
| IA3 | **`content` only** |

When IA3 finds no `content` key it serialises the whole object to JSON. Its
error-prefix regex is anchored to the start of the string, so a serialised
object never matches:

```json
{"result": {"output": "Error: connection refused"}}
```

- IA2 extracts `Error: connection refused` → fires `tool_output_error_prefix`.
- IA3 sees `{"output":"Error: connection refused"}` → the anchored regex fails
  → **no finding at all**.

The result is a corpus where IA2 reports failures that IA3 never saw, which
looks like a detector bug and is actually an adapter bug.

**Always put textual results under `content`.** The validator warns when it
sees text under `output`, `message` or `summary` with no `content` key.

## Checking your own corpus

```bash
insight-agent explain-failures traces.jsonl --only-disagreements
```

On a well-formed corpus this prints `No disagreements.` Anything else is worth
investigating before you trust the findings:

```
trace                        call                  tool         IA2                IA3
--------------------------------------------------------------------------------------
run-0007                     call-2                Search       FAIL error_prefix  ok
```

Add `--json` for machine-readable output.

## Why not just merge them?

Three reasons:

1. IA2's decoder feeds `explicit_failure_rate`, one of the eleven features
   behind anomaly detection. Changing it changes which traces are flagged, and
   the current behaviour is what was measured.
2. IA3's decoder is deliberately stricter. It is a rule-based auditor whose
   whole value is that it does not guess; broadening it would trade precision
   for recall in a component chosen for precision.
3. The two engines answer different questions. "This trace looks statistically
   odd" and "this call demonstrably violated a contract" do not need the same
   threshold for what counts as failure.

The canonical format's job is to make the difference *visible and predictable*,
not to hide it.
