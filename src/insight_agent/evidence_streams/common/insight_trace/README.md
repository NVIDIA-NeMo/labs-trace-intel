# The canonical trace format (`insight-trace/v1`)

One JSON object per line; one line per complete trace. This page is generated
from `src/insight_agent/evidence_streams/common/insight_trace/schema.json` — print the schema
itself with `uv run insight-agent schema`.

Adapters can emit this format in any language. `InsightTraceLoader` validates it and
normalizes each record into the local `Trace` and `Span` models. Evidence streams then own
their projections into any algorithm-specific types.

## Trace object

Required: `calls`, `schema_version`, `trace_id`

| Field | Type | Required | What it does / what omitting it costs |
|---|---|:---:|---|
| `schema_version` | `insight-trace/v1` | **yes** | Format gate. Must be the literal string 'insight-trace/v1'. |
| `trace_id` | string | **yes** | Stable identifier for this trace, unique across the corpus. Used as the join key in every IA2 and IA3 output. |
| `calls` | array | **yes** | Ordered tool calls. May be empty, in which case IA2 features are all zero and IA3 emits nothing for this trace. |
| `steps` | array | no | Observable trajectory steps, including non-tool steps such as planning, agent turns and evaluation boundaries. OMITTING THIS CHANGES IA2 NUMBERS: trajectory tokens fall back to one 'tool:{name}' token per call and trajectory_step_count becomes the call count, so clusters and anomaly scores shift. Two adapters over the same corpus, one emitting steps and one not, are not comparable. |
| `source_pointer` | object | no | Free-form pointer back to the original record (dataset, run id, URI, offsets). Carried verbatim into every digest row and finding. Omitting it makes the digest print 'not supplied' and leaves the Analyst unable to reopen the evidence. |
| `logical_case_id` | string/null | no | Identifier of the underlying task this trace attempts. Several traces of the same task share one logical_case_id. Defaults to trace_id when omitted, WHICH INFLATES independent_case_count: IA3 promotes a card only after three independent logical cases, so omitting this turns N retries of one task into N 'independent' cases and makes the precision gate meaningless. |
| `observed_verdict` | string/null | no | Explicit terminal verdict recorded by the harness, e.g. 'completed' or 'dead_end: no artifact produced'. Grouped as evidence by IA2 stage 5. Omitting it makes the digest's terminal-verdict section print an explicit abstention. Never used for anomaly selection or trajectory clustering. |
| `cost` | number/null | no | Total known cost of the trace in whatever unit the venue uses. Feeds verdict-group cost totals and representative ranking only. |
| `metrics` | object | no | Extra numeric features merged into the IA2 anomaly-detection feature vector. Keys MUST NOT collide with the eleven built-in feature names, because a collision silently replaces a measured feature. Values must be finite numbers. |
| `tool_catalog` | object/null | no | Maps tool name to the JSON Schema for that tool's arguments. HIGHEST-VALUE OPTIONAL FIELD: omitting it makes six IA3 rules abstain (unknown_tool, missing_required_argument, unknown_argument, argument_type_mismatch, argument_enum_violation, json_schema_violation). malformed_tool_call still fires for non-object arguments with or without a catalog. A null value for a specific tool name means 'this tool is known but its schema is unavailable', which enables unknown_tool detection while abstaining from argument checks. |
| `complete_provenance_context` | boolean | no | Assert that this record captures EVERY user message and EVERY prior tool result for the trace. Only then can IA3 conclude an identifier was ungrounded rather than merely unseen. Setting this true without genuinely complete capture produces false positives; leaving it false makes explicitly_rejected_ungrounded_identifier abstain. |
| `task_text` | string | no | The originating user request. Used as the fallback for each call's prior_user_text when that is not set per call. |
| `orphan_results` | array | no | Tool results observed with no matching call. Evidence of instrumentation loss. Without this, orphan_tool_result never fires. |
| `extra` | object | no | Free-form vendor passthrough. Ignored by every engine. Exists so the rest of the record can stay strict: any unrecognised key outside 'extra' is an error, which turns a typo such as 'tool_catelog' into a loud failure instead of six silently disabled rules. |

## Call object

Required: `arguments`, `call_id`, `call_index`, `tool_name`

| Field | Type | Required | What it does / what omitting it costs |
|---|---|:---:|---|
| `call_id` | string | **yes** | Identifier for this call, unique WITHIN the trace. If the source has no call ids, synthesise them as '{trace_id}#{call_index}'. A constant or repeated value fires duplicate_call_id on every call. |
| `call_index` | integer | **yes** | Zero-based position of this call in the trace. Used as the ordering key by both engines. |
| `tool_name` | string | **yes** | Name of the invoked tool, matching the keys of tool_catalog. Drives catalog lookup, dominant_tool_share and code_execution_share. |
| `arguments` | any | **yes** | The arguments as the agent supplied them. EMIT RAW: a non-object value is legitimate evidence and produces malformed_tool_call. Do not parse a JSON-string argument blob into an object unless the source runtime itself parsed it, and do not stringify an object. |
| `result` | any | no | The tool's result. KEY ABSENCE IS SEMANTIC: omitting 'result' entirely means no result was ever recorded and fires missing_tool_result, whereas "result": null means the tool genuinely returned null. If the result is text, emit a plain string or {"content": "..."} — use 'content', never 'output' or 'message', because IA3 only unwraps 'content' and will otherwise see serialised JSON and miss error prefixes and tracebacks. |
| `result_missing` | boolean | no | Explicit escape hatch meaning 'no result was recorded', for emitters that cannot omit a key. Equivalent to omitting 'result'. |
| `result_count` | integer | no | How many results were observed for this call. 0 is treated as missing; values above 1 fire duplicate_tool_result. |
| `result_id` | string/null | no | The identifier the result itself claims to answer. ONLY populate this when the source has a genuinely independent result-to-call reference. Filling it from a different id namespace (a message id, say) fires call_result_id_mismatch on essentially every call. |
| `duration_ms` | number/null | no | Wall-clock duration of the call in milliseconds. Feeds tool_duration_sec_total. |
| `explicit_error` | boolean/null | no | Authoritative success flag from the source, when one exists. true forces a failure; FALSE FORCES SUCCESS AND DISABLES ALL TEXT-BASED FAILURE DECODING for this call. Prefer null (or omit) unless the source really carries a reliable flag, otherwise explicit_tool_failure goes silent across the whole corpus. |
| `outcome_marker` | string/null | no | Names the failure mechanism when explicit_error is true. Becomes the card's mechanism_key, so it determines card identity. |
| `instrumentation_alias_of` | string/null | no | The real tool name when tool_name is an instrumentation artefact. Fires mapped_instrumentation_alias. |
| `prior_user_text` | string | no | User text visible before this call, used to decide whether an identifier was grounded in the request. Falls back to the trace-level task_text. |
| `returned_data` | boolean | no | False means the call succeeded but returned no rows or matches. Drives returned_data_false_rate. The loader injects this into the result mapping, so it has no effect when the result is a bare string. |
| `source_pointer` | object | no | Free-form pointer back to this specific call in the source record. Carried verbatim into every finding. |
| `extra` | object | no | Free-form vendor passthrough for this call. Ignored by every engine. |

## Step object

Required: `step_index`, `step_type`

| Field | Type | Required | What it does / what omitting it costs |
|---|---|:---:|---|
| `step_index` | integer | **yes** | Zero-based position of this step in the trajectory. |
| `step_type` | string | **yes** | Kind of step. Conventional values are 'tool', 'agent', 'planning', 'evaluation' and 'user'. Steps typed 'evaluation' collapse to a single token during clustering so per-run verdict text does not fragment the clusters. |
| `name` | string | no | Step label, typically the tool name for tool steps. Becomes part of the trajectory token. |
| `content` | string | no | Text produced at this step. Only read for agent-ish steps and for the terminal evaluation. |
| `source_pointer` | object | no | Free-form pointer back to this step in the source record. |

## Orphan result object

Required: `result_id`

| Field | Type | Required | What it does / what omitting it costs |
|---|---|:---:|---|
| `result_id` | string | **yes** | The call id this stray result claims to answer. |
| `tool_name` | string | no | Tool the result appears to come from, if known. Documentation only: IA3 records orphans under '<unknown>'. |
| `content` | any | no | The stray result payload, if captured. |
| `source_pointer` | object | no | Pointer back to the stray result. Passed through as the finding's source_pointer. |

## A record that exercises everything

This single record triggers an argument type mismatch, a state failure, an
unresolved placeholder, a missing result, an orphan result and a Python
traceback:

```json
{
  "schema_version": "insight-trace/v1",
  "trace_id": "run-0117",
  "logical_case_id": "case-export-step-042",
  "source_pointer": {
    "dataset": "agent-eval",
    "uri": "s3://traces/runs/0117.json"
  },
  "observed_verdict": "dead_end: no artifact produced",
  "cost": 1.42,
  "metrics": {
    "turn_count": 14.0
  },
  "complete_provenance_context": true,
  "task_text": "Export part 44821 to STEP, then run the mesh check.",
  "tool_catalog": {
    "FileSearchTool": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string"
        },
        "limit": {
          "type": "integer",
          "minimum": 1
        }
      },
      "required": [
        "query"
      ],
      "additionalProperties": false
    },
    "SessionTool": null
  },
  "calls": [
    {
      "call_id": "toolu_01AB",
      "call_index": 0,
      "tool_name": "FileSearchTool",
      "arguments": {
        "query": "mesh report",
        "limit": "five"
      },
      "result": {
        "content": "Error: limit must be an integer"
      },
      "returned_data": false,
      "duration_ms": 96.0,
      "source_pointer": {
        "message_index": 5
      }
    },
    {
      "call_id": "toolu_01AC",
      "call_index": 1,
      "tool_name": "SessionTool",
      "arguments": {
        "document_id": "DOC-99213"
      },
      "result": {
        "content": "[NO_ACTIVE_SESSION] invalid document DOC-99213: not found"
      },
      "source_pointer": {
        "message_index": 7
      }
    },
    {
      "call_id": "toolu_01AE",
      "call_index": 2,
      "tool_name": "FileReadTool",
      "arguments": {
        "path": "<output_dir>"
      },
      "source_pointer": {
        "message_index": 11
      }
    }
  ],
  "steps": [
    {
      "step_index": 0,
      "step_type": "planning",
      "name": "plan",
      "content": "Find, export, verify."
    },
    {
      "step_index": 1,
      "step_type": "tool",
      "name": "FileSearchTool",
      "duration_ms": 96.0
    },
    {
      "step_index": 2,
      "step_type": "tool",
      "name": "SessionTool"
    },
    {
      "step_index": 3,
      "step_type": "evaluation",
      "name": "boundary",
      "content": "The export completed successfully."
    }
  ],
  "orphan_results": [
    {
      "result_id": "toolu_ORPHAN_1",
      "source_pointer": {
        "message_index": 13
      }
    }
  ]
}
```

The third call has **no `result` key**. That is what fires `missing_tool_result`;
writing `"result": null` instead would mean the tool returned null.
