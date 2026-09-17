<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Add context and checks

All five checks are enabled by default. Each runs when its prerequisites are available.
Use the section below that matches a skipped check in your report.

- [Ethos divergence](#ethos-divergence): supply your agent’s business rules.
- [User sentiment](#user-sentiment): configure embeddings for complaint detection.
- [Evaluation failures](#evaluation-failures): include recorded scores or feedback.
- [Tool issues](#tool-issues): preserve tool calls and results.
- [Anomalies and patterns](#anomalies-and-patterns): analyze behavior across traces.

## Ethos divergence

Provide a Markdown file describing your agent’s purpose, business rules, and boundaries.
Use concrete requirements, such as:

> Help customers manage orders. Issue refunds only within 30 days of purchase.
> Ask for confirmation before cancelling an order.

Save your own rules in `ethos.md`, then add this to your configuration:

```yaml
evidence_streams:
  ethos_divergence:
    ethos_path: ethos.md
```

The file must exist and contain text. Its path resolves from your working directory.

## User sentiment

This check looks for recurring user complaints. It needs recorded user messages and
an embedding backend serving **Qwen/Qwen3-Embedding-8B with 4,096-dimensional output**.
The bundled classifier was trained for this model; other embedding models are not supported.
Local embeddings are off by default. Without a remote endpoint or local embeddings enabled,
this check is skipped.

For a remote endpoint, add:

```yaml
evidence_streams:
  user_sentiment:
    litellm:
      model: openai/your-qwen3-embedding-8b-alias
      api_base: https://embeddings.example.com/v1
      api_key_env: EMBEDDING_API_KEY
```

Set `EMBEDDING_API_KEY` in your [environment file](model-access.md#credentials).
Replace the model alias and endpoint with your provider’s values.

To run the 8B embedding model locally, add `local-embedding` to the extras in your
install command. Keep your source extra too. For LangSmith:

```bash
uv tool install \
  'insight-agent[langsmith,local-embedding] @ git+https://github.com/NVIDIA-NeMo/labs-trace-intel.git@main'
insight-agent --config config.yaml --evidence-streams.user-sentiment.local-embeddings
```

Or enable local embeddings in your configuration:

```yaml
evidence_streams:
  user_sentiment:
    local_embeddings: true
```

Local inference downloads model weights and needs enough memory to run them.
Hardware is selected automatically. A configured `litellm` endpoint takes precedence.

Both local and remote embeddings still use your main inference model to investigate
complaints. See [data access](model-access.md#where-data-goes).

## Evaluation failures

Include Braintrust root-span scores, Langfuse scores, MLflow assessments, or LangSmith feedback.
For canonical JSONL traces, populate `evaluator_results`.
If you already record these, check that your selected traces and exports include them.

Trace Analyst uses those signals to investigate recurring failures. It does not run
your evaluation suite or create missing scores. Setup and export details live in your
[source guide](../README.md#start-here).

## Tool issues

Load traces with recorded tool spans, names, arguments, and results.
Include tool schemas when available so argument-validation checks can run.
If the report says “No tool calls,” check that your selected traces and export preserve them.

## Anomalies and patterns

This check looks for unusual traces and recurring behavior across the loaded set.
A small or repetitive set may not support trajectory grouping; other analysis can still finish.
Load more varied traces from the behavior you want to investigate.

## Disable a check

Set a check to `false` to disable it. Disabled checks produce no report entries or skip messages:

```yaml
evidence_streams:
  ethos_divergence: false
  user_sentiment: false
```

The other keys are `eval_failure_patterns`, `tool_issues`, and `anomaly_and_patterns`.
Omit a key, or use `true` or `{}`, for defaults. Keep at least one check enabled.
When combining examples, put their settings under a single `evidence_streams` key.

## Check findings against code

Add `code_base: ../my-agent` to your configuration to consult your agent’s local source.
The validator removes candidate problems contradicted by the code and retains those
it cannot resolve from the repository. Its tools can search and read files; they cannot
edit or execute your agent’s code.
