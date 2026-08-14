# Capability matrix

Which canonical field each of the nineteen IA3 finding types depends on, and
what happens without it. Generated from `insight_agent.coverage.RULE_REQUIREMENTS`;
a test asserts every finding type appears here.

Check your own corpus with `insight-agent coverage traces.jsonl`.

## Requires `tool_catalog`

| Finding type | Family | Gate | Notes |
|---|---|---|---|
| `argument_enum_violation` | tool_contract_and_arguments | `ia3_tid.py:168` | Needs an enum constraint in the schema. |
| `argument_type_mismatch` | tool_contract_and_arguments | `ia3_tid.py:167` | Needs typed properties in the schema. |
| `json_schema_violation` | tool_contract_and_arguments | `ia3_tid.py:169` | Any schema keyword other than required/additionalProperties/type/enum. |
| `missing_required_argument` | tool_contract_and_arguments | `ia3_tid.py:165` | Needs the per-tool argument schema. |
| `unknown_argument` | tool_contract_and_arguments | `ia3_tid.py:166` | Needs a schema with additionalProperties:false. |
| `unknown_tool` | tool_contract_and_arguments | `ia3_tid.py:157` | The catalog is what makes a tool name 'unknown'. |

## Requires nothing (always evaluable)

| Finding type | Family | Gate | Notes |
|---|---|---|---|
| `duplicate_call_id` | call_result_integrity | `ia3_tid.py:298` | Needs call_id, which is required. |
| `malformed_tool_call` | tool_contract_and_arguments | `ia3_tid.py:155` | Fires on non-object arguments with or without a catalog. |

## Requires `result` key *absent* on some calls

| Finding type | Family | Gate | Notes |
|---|---|---|---|
| `missing_tool_result` | call_result_integrity | `ia3_tid.py:303` | Fires only when the 'result' key is absent (or result_missing / result_count:0). An adapter that always emits a result, even null, silences this rule. |

## Requires `result_count` > 1

| Finding type | Family | Gate | Notes |
|---|---|---|---|
| `duplicate_tool_result` | call_result_integrity | `ia3_tid.py:307` | Needs result_count > 1. |

## Requires `result_id`

| Finding type | Family | Gate | Notes |
|---|---|---|---|
| `call_result_id_mismatch` | call_result_integrity | `ia3_tid.py:318` | Only populate result_id from a genuine result-to-call reference. |

## Requires `orphan_results`

| Finding type | Family | Gate | Notes |
|---|---|---|---|
| `orphan_tool_result` | call_result_integrity | `ia3_tid.py:416` | Needs the adapter to capture results with no matching call. |

## Requires `instrumentation_alias_of`

| Finding type | Family | Gate | Notes |
|---|---|---|---|
| `mapped_instrumentation_alias` | trace_instrumentation | `ia3_tid.py:329` | Needs the adapter to know the real tool behind an alias. |

## Requires a decodable `result`

| Finding type | Family | Gate | Notes |
|---|---|---|---|
| `explicit_prerequisite_or_state_failure` | explicit_prerequisite_or_state | `ia3_tid.py:398` | Needs result text matching the venue's state patterns. |
| `explicit_tool_failure` | explicit_tool_outcome | `ia3_tid.py:341` | Needs a decodable result. explicit_error:false disables it entirely. |
| `modified_retry_same_failure` | recovery_and_retries | `ia3_tid.py:439` | Needs three failing calls sharing a failure class across >=2 argument sets. |
| `repeated_identical_failed_call` | recovery_and_retries | `ia3_tid.py:429` | Needs three failing calls with byte-identical arguments. |
| `unresolved_placeholder_argument` | argument_provenance | `ia3_tid.py:366` | Needs a failed or rejected call whose argument is exactly a placeholder. |

## Requires `complete_provenance_context`

| Finding type | Family | Gate | Notes |
|---|---|---|---|
| `explicitly_rejected_ungrounded_identifier` | argument_provenance | `ia3_tid.py:377` | Set true only if every user message and prior result was captured. |

## Summary

| Field | Rules it unlocks |
|---|---:|
| `tool_catalog` | 6 |
| a decodable `result` | 5 |
| nothing (always evaluable) | 2 |
| `result` key *absent* on some calls | 1 |
| `result_count` > 1 | 1 |
| `result_id` | 1 |
| `orphan_results` | 1 |
| `instrumentation_alias_of` | 1 |
| `complete_provenance_context` | 1 |

Total: 19 finding types.

`tool_catalog` is the single highest-leverage field: it alone gates six rules,
and it is usually already present in the source as a `tools` array.

