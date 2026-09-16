<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Try Trace Analyst

Run the example customer-service traces to explore the insights Trace Analyst produces.
You only need an inference API key; no trace-platform account is required.

With [uv and Git installed](../README.md#start-here), install the CLI:

```bash
uv tool install --python 3.12 \
  'insight-agent @ git+https://github.com/NVIDIA-NeMo/labs-trace-intel.git@main'
```

Download the [example traces](tau_bench_traces.jsonl) and
[configuration](trace-analyst-config.yaml) into one folder, keeping their filenames.
In that folder, [configure your model and API key](../docs/model-access.md#choose-a-model), then run:

```bash
insight-agent --config trace-analyst-config.yaml
```

The terminal reports completed and skipped checks. If it finds actionable insights,
it saves them to `insights.yml`. Results vary with the model.

[Read your results](../docs/results.md), or [connect your own traces](../README.md#start-here).

## About the example data

The file contains 200 traces from τ-bench: 139 telecom, 51 retail, and 10 airline traces.
Customer names, addresses, and order IDs are synthetic benchmark fixtures.
The data is covered by the [τ-bench MIT license](../third_party/tau-bench-LICENSE.txt).

The conversion preserves observed tool calls and per-trace tool schemas. It removes
blanket `explicit_error: false` annotations, retains recorded errors, and uses dataset
references in source pointers. It adds no unobserved agent steps.
