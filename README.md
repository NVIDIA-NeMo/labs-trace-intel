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

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Git](https://git-scm.com/downloads/), then choose a guide:

| Your starting point | Guide |
| --- | --- |
| Try it before connecting your data | [Run the example](examples/README.md) |
| Braintrust project logs and experiments | [Braintrust](docs/sources/braintrust.md) |
| LangSmith traces and feedback | [LangSmith](docs/sources/langsmith.md) |
| Langfuse traces and scores | [Langfuse](docs/sources/langfuse.md) |
| MLflow traces and assessments | [MLflow](docs/sources/mlflow.md) |
| NeMo Platform Intake | [Intake](docs/sources/intake.md) |
| JSONL, ATIF, or a custom source | [Trace files](docs/sources/files.md) |

If `insight-agent` is not found after installation, run `uv tool update-shell`
and restart your terminal.

## After your first run

- [Read your results](docs/results.md) — understand insights, skipped evidence streams, and saved output.
- [Evidence streams](docs/evidence-streams.md) — supply business rules, configure sentiment, or check findings against code.
- [Data and model access](docs/model-access.md) — configure your inference endpoint and understand where data goes.
- [Configuration reference](docs/configuration.md) — overrides, limits, and repeatable runs.

[Development](DEVELOPMENT.md) · [Architecture](docs/architecture.md)

## License

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This project is licensed under the [Apache License, Version 2.0](LICENSE). See
[NOTICE](NOTICE) for project attributions,
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for human-readable dependency license
disclosures, and [third_party/licenses.jsonl](third_party/licenses.jsonl) for the machine-readable
inventory.

This project is currently not accepting contributions.
