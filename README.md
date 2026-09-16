<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trace Analyst

Trace Analyst analyzes your agent’s behavior and surfaces actionable insights,
grounded in its execution traces.

Bring traces from your existing tools, and explore each insight through the evidence behind it.

For example, an insight might say:

> **Retries repeat the same invalid request.** After a tool rejects an argument,
> the agent sends it again without correcting it. Seen in traces `run-12` and `run-38`.

This is a research preview for collaboration and evaluation.

## Start here

You’ll need Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).
Clone the repository, then follow the guide for your data:

```bash
git clone https://github.com/NVIDIA-NeMo/labs-trace-intel.git
cd labs-trace-intel
```

| Your starting point | Guide |
| --- | --- |
| Try it before connecting your data | [Run the example](examples/README.md) |
| LangSmith traces and feedback | [LangSmith](docs/sources/langsmith.md) |
| Langfuse traces and scores | [Langfuse](docs/sources/langfuse.md) |
| MLflow traces and assessments | [MLflow](docs/sources/mlflow.md) |
| NeMo Platform Intake | [Intake](docs/sources/intake.md) |
| JSONL, ATIF, or a custom source | [Trace files](docs/sources/files.md) |

## After your first run

- [Read your results](docs/results.md) — understand insights, skipped checks, and saved output.
- [Add context and checks](docs/checks.md) — supply business rules, configure sentiment, or check findings against code.
- [Data and model access](docs/model-access.md) — configure your inference endpoint and understand where data goes.
- [Configuration reference](docs/configuration.md) — overrides, limits, and repeatable runs.

## Project

[Development](DEVELOPMENT.md) · [Architecture](docs/architecture.md)

Licensed under [Apache 2.0](LICENSE). See [NOTICE](NOTICE),
[dependency licenses](THIRD_PARTY_LICENSES.md), and the [license inventory](third_party/licenses.jsonl).
The bundled traces come from [τ-bench](third_party/tau-bench-LICENSE.txt).
This project is currently not accepting contributions.
