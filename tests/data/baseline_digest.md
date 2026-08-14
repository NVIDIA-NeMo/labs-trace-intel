# IA2 cited evidence digest

## Reader contract

IA2 answers **what is unusual?** and **what recurs?** It does not author an Insight, prove causality, or turn an anomaly into an error. The Analyst must inspect cited traces and may file zero Insights.

## Inventory

- Complete traces: 4
- Ordered tool calls: 32
- Isolation Forest flags: 1
- Trajectory groups: 2
- Observed-verdict groups: 1
- Recurring strict-failure signatures: 0

## Unusual traces

| Trace | Score | Feature reasons | Source pointer |
|---|---:|---|---|
| `base-outlier-0` | 0.00599 | high identical-call repetition; high average tool-output size; high largest tool output | `{"trace_id":"base-outlier-0"}` |

## Recurring ordered trajectory patterns

| Group | Traces | Representative | Characteristic sequence terms |
|---:|---:|---|---|
| 0 | 3 | `base-search-0` | tool:FileSearchTool, tool:FileSearchTool tool:FileSearchTool, tool:SessionTool evaluation:boundary, tool:SessionTool, tool:FileSearchTool tool:SessionTool, planning:plan tool:FileSearchTool |
| 1 | 1 | `base-code-0` | tool:CodeExecutionTool, tool:CodeExecutionTool tool:CodeExecutionTool, tool:CodeExecutionTool evaluation:boundary, agent:reflect tool:CodeExecutionTool, agent:reflect, evaluation:boundary |

## Explicit terminal-verdict patterns

| Verdict | Traces | Representative pointers |
|---|---:|---|
| dead_end: no artifact produced | 3 | `base-outlier-0` `{"trace_id":"base-outlier-0"}`; `base-search-1` `{"trace_id":"base-search-1"}`; `base-search-0` `{"trace_id":"base-search-0"}` |

## Recurring strict tool-failure signatures

These are observed tool/runtime outputs, not proof that the agent chose the wrong tool.

| Signature | Traces | Events | Representative calls |
|---|---:|---:|---|

## Analyst authoring rules

1. File only recurring, operationally useful, evidence-backed Insights.
2. Cite exact trace IDs and call IDs, then open raw evidence when needed.
3. Distinguish observed tool/runtime failure from inferred agent misuse.
4. Do not infer root cause or quality from anomaly or cluster membership.
5. Withhold weak or duplicate claims and state what remains unproven.
