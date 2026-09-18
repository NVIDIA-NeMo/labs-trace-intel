<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trace ingestion

Canonical trace models and provider loaders for use outside Trace Analyst.

Install a pinned Git revision:

```bash
uv add "git+https://github.com/NVIDIA-NeMo/labs-trace-intel#subdirectory=packages/trace-ingest" --rev <commit-sha> --extra mlflow
```

Load an MLflow export:

```python
from trace_ingest.loaders.mlflow import MLflowFileTraceConfig, MLflowFileTraceLoader

loader = MLflowFileTraceLoader(MLflowFileTraceConfig(path="traces.json"))
for trace in loader.load():
    print(trace.id)
```

Use the `mlflow` extra for both MLflow file and live loaders, the `langsmith`
extra for its live loader, and the `langfuse` extra for its live loader.

## NeMo Gym rollout exports

Load Gym's persisted rollout JSONL with its `ng_trajectory` schema `1.0`
attachment. No Gym installation or optional extra is needed:

```python
from trace_ingest.loaders import GymTraceConfig, GymTraceLoader

loader = GymTraceLoader(GymTraceConfig(path="rollouts.jsonl"))
for trace in loader.load():
    print(trace.id, trace.attributes["logical_case_id"])
print(loader.describe())
```

Use `format="json"` for one pretty-printed rollout object. Each JSONL record
remains one rollout; task IDs identify logical cases across repeated rollouts.
Loading is offline, cached and disk-backed. Duplicate rollout IDs are rejected.

The loader maps the existing Gym trajectory contract directly to canonical spans:

- Invocations become agent spans, nested by recorded parent IDs. Missing parents
  remain roots with an explicit loader gap; cycles are rejected.
- Captured model calls become LLM spans with request/response payloads, recorded
  timestamps, model identity and token counts. Ownership uses exact model-call
  IDs or model-server/response-ID pairs. Ambiguous joins are rejected; unresolved
  references remain explicit gaps. Partial token counts remain in metadata,
  never filled with zero. Whole-rollout totals are not inferred from partial capture.
- Invocation conversations supply function arguments, call/result pairing and
  preceding user text. Tool observations add recorded timing, status and output.
  Call IDs are scoped to their invocation. Namespaced functions and unsupported conversation item types
  fail explicitly rather than disappearing from tool analysis.
- Semantic turns become chain spans with recorded question/answer payloads and
  additional decision evidence in metadata, without duplicate LLM accounting.
  Rewards are source-reported `gym.reward` and
  `gym.reward_components` signals, not independently verified success.
- The complete source object remains in `trace.attributes["gym"]`, including
  evidence gaps, structured media, reasoning summaries and extension fields.
  Media and embedded filesystem references are never opened or fetched.

Within each parent, recorded start times determine order; ties and unavailable
times retain stable source order, with untimed spans last. This ordering is not
an assertion that untimed events happened after timed events.

Gym's optional tool-observation `output: null` may mean unavailable. It remains
null, but does not establish a matched result unless an explicit conversation
result exists; a non-null enriched output does establish one.

This supports Gym's
[TrajectoryRecord](https://github.com/NVIDIA-NeMo/Gym/blob/399e6783e0e879424fc20a23f8e2d46ffa4cfdee/nemo_gym/rollout_observability.py)
and [rollout attachment](https://github.com/NVIDIA-NeMo/Gym/blob/399e6783e0e879424fc20a23f8e2d46ffa4cfdee/nemo_gym/rollout_collection.py)
contracts. Gym's ATIF adapter runs in the opposite direction (ATIF → Gym for
reverification); its MLflow/W&B exporter view omits `ng_trajectory`.
Use the persisted rollout JSONL, not that exporter view.

Legacy Responses-only records without `ng_trajectory` are rejected. Enable Gym
observability when collecting new runs, or load retained original ATIF with
`ATIFTraceLoader`. The loader does not reconstruct an unobserved trajectory from
the final response or rerun Gym's capture-joining logic.

For local development, use an editable checkout:

```bash
uv add --editable /path/to/labs-trace-intel/packages/trace-ingest --extra mlflow
```
