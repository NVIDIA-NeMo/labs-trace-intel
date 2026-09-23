<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trace Analyst

![Status: Research Preview](https://img.shields.io/badge/Status-Research%20Preview-orange)

Agent developers collect large numbers of execution traces, but finding a recurring,
fixable problem in them is hard. Trace Analyst explores how to turn those traces into
actionable findings, which we call **insights**, with links back to the executions
that support each finding. It brings traces from existing observability tools into
evidence streams that look for different kinds of problems.

For example, an insight might say:

> **Retries repeat the same invalid request.** After a tool rejects an argument,
> the agent sends it again without correcting it. Seen in traces `run-12` and `run-38`.

> [!WARNING]
> **Research preview**
>
> This is an early research example for experimentation and collaboration with
> developers. Its APIs, configuration, and output formats may change without
> backward compatibility. Findings require review against their supporting
> traces before you act on them. This project is not intended for production use.

We are sharing the implementation and techniques that have helped us find signal
in agent traces. We want to learn which problems matter to other developers and
which parts of this approach are useful in their own tools.

## Questions we are exploring

- How can we find recurring, fixable agent problems in trace stores containing
  thousands or millions of executions?
- How can we estimate how often a problem occurs and how much it matters without
  asking an LLM to inspect every trace?
- How can trace evidence, evaluation results, and read-only checks against agent
  code help distinguish real problems from misleading patterns?
- How should we measure whether an insight is correct, useful, and worth a
  developer's time to investigate?

This repository explores these questions through trace loaders, [evidence
streams](docs/evidence-streams.md), and insight compilation. The [architecture
guide](docs/architecture.md) explains what each part does and where judgment or
coverage limits remain.

## Help shape the research

We are especially interested in feedback from teams building agent observability
or evaluation platforms, and from developers investigating their own agents:

- Which recurring issues are hardest to find today? What would make a finding
  actionable: examples, frequency, estimated impact, or something else?
- Where do the insights miss an important problem, group unrelated behavior,
  or suggest a cause the traces do not support?
- Which layer would you use in your own workflow: the full analyst, individual
  evidence streams or trace loaders, or lower-level techniques such as embeddings
  and retrieval? What would you need to integrate it?

If an NVIDIA contact shared this preview with you, please send your feedback
through that contact and ask them to route it to the Trace Analyst research team.
Otherwise, use the [NVIDIA Developer contact form](https://developer.nvidia.com/contact)
with **Trace Analyst research preview feedback** in the subject. Your role or
platform, approximate trace volume, an anonymized example, and the part of the
approach you would use are especially helpful. Please avoid sending raw traces,
prompts, credentials, or user data in an initial message.

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
| NeMo Gym rollout trajectories | [NeMo Gym](docs/sources/gym.md) |
| JSONL or ATIF files | [Trace files](docs/sources/files.md) |
| Another trace store | [Convert your traces](docs/sources/custom.md) |

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
