<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze Braintrust traces

Choose a project or evaluation experiment containing your agent’s traces.
Recorded scores on root spans are included automatically in evaluation failure analysis.

## Connect and run

With [uv and Git installed](../../README.md#start-here), install the CLI:

```bash
uv tool install \
  'insight-agent @ git+https://github.com/NVIDIA-NeMo/labs-trace-intel.git@main'
```

No Braintrust extra is required. [Configure your inference model and key](../model-access.md#choose-a-model).
Add your Braintrust key to `.env`:

```dotenv
BRAINTRUST_API_KEY=your-braintrust-key
```

Save this as `config.yaml`, replacing the project ID and dates:

```yaml
max_tokens: 16384
trace:
  max_traces: 100
  braintrust:
    project_id: your-project-id
    from_timestamp: 2026-09-01T00:00:00Z
    to_timestamp: 2026-09-02T00:00:00Z
```

```bash
insight-agent --config config.yaml
```

Open `insights.yml` if the run produced insights. [Read your results](../results.md)
for help with findings or skipped checks.

## Select the right traces

For an evaluation experiment, replace `project_id` with `experiment_id`.
Supply exactly one ID. Both timestamps are required and must include a timezone.
The window applies to root creation times, inclusive at the start and exclusive at the end.

The default limit is 100 complete traces. Selection follows descending Braintrust pagination-key
order. All pages of spans for each selected trace are loaded, including children outside the
window, and the resulting traces are ordered by root creation time and ID.
Use a window of completed runs; the loader does not wait for in-flight spans.

For a single run:

```bash
insight-agent --config config.yaml --trace.max-traces 25
```

The API endpoint defaults to `https://api.braintrust.dev`. Set `BRAINTRUST_API_URL` in `.env`
for a different deployment, or set `api_url` under `trace.braintrust` to override it.
Use `https://api-eu.braintrust.dev` for EU data or your self-hosted data plane URL.
The loader uses Braintrust’s SQL query API, requiring data plane v1.1.29 or later.

## What is preserved

Inputs, outputs, errors, scores, and provider metadata are retained. Missing outputs stay
missing; explicit JSON null remains a recorded result. Root scores become trace evaluation
results, span `metrics.estimated_cost` becomes cost, and whole-trace latency is derived from
span times. Other metrics remain in `attributes.braintrust.metrics`.

Tool catalogs are not normalized into schemas, which limits tool-schema checks.
Duplicate IDs, missing parents, cycles, and multiple-parent spans produce an error.
HTTP failures, including rate limits, are reported without automatic retries.

Braintrust supports live queries only in this loader. For local input, use
[canonical trace files](files.md).
