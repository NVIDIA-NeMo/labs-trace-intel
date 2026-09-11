<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Insight Agent Research Preview

The NVIDIA team has been working on agent-driven trace analysis techniques. 

An early version of that work has shipped open source and is available here as a "research preview": [Open Source Repo](https://github.com/NVIDIA-NeMo/nemo-platform/blob/f57bb6ca64d84e505742e73f6234fd31e27e7a4a/plugins/nemo-insights/README.md) and [Documentation](https://docs.nvidia.com/nemo-platform/documentation/agents/optimize-agents/insight-driven-optimization).

That implementation should be viewed as a steel thread to illustrate the architecture. In that example, the "Analyst Agent's" job is to conduct trace analysis to turn usage data from traces into high-signal insights that can drive downstream optimization.

## Where we're going next
That version of the Analyst Agent is a relatively naive initial implementation. The agent is given a set of tools to explore traces and is tasked with coming up with insights for the agent under test. While it has proven to produce valuable results, it hits limits with large volumes of traces as its ability to explore the whole search space declines. 

Future iterations will aim to improve the scalability and reliability of the agent by adding diverse preprocessing steps across the whole trace corpus, providing a "roadmap" of sorts to guide the Analyst Agent's exploration and more quickly zero in on the most significant traces. 

**This repo represents an early preview of the architecture we are exploring.** It's shared for collaboration purposes only and is not meant to be shared widely or to be used in a production environment. 

## V2 Analyst Agent Architecture
This version of the Analyst Agent implements a series of preprocessing steps that we call "evidence streams".

Current evidence streams:

1) **Anomaly and Pattern Analysis** - Which traces are unusual, and which patterns recur?

    A set of 11 features are extracted from each trace and the resulting feature vector feeds an Isolation Forest model which identifies statistical anomalies.

    Traces are clustered based on TFIDF after structural metadata projection and error extraction. 

2) **Tool Issue Detection** - Which tool use problems recur across the trace sample?

    The tools in each trace are checked against a set of deterministic rules to detect specific tool calling issues. 

3) **Ethos Divergence** - Where does observed agent behavior violate the business purpose and requirements in a supplied `ethos.md`?

    An LLM checks the traces against the document. Select it with `--evidence-streams.ethos-divergence.ethos-path /path/to/ethos.md`.

4) **Evaluation Failure Patterns** - What recurring behavior is linked to recorded evaluation results?

    Adapters must populate `Trace.evaluator_results`; the LLM does not discover evaluation fields.

After the evidence streams run, each returns candidate `Problem` objects with a description and supporting trace IDs. Those Problems are passed to the Analyst Agent to guide its analysis. The Analyst can also look up normalized supporting traces from the same snapshot and uses that evidence as a starting point for more detailed exploration and synthesis.

The Analyst Agent ultimately produces a set of "insights" which are meant to describe a recurring and actionable problem observed from the trace corpus. 

## Running the Agent on Example Traces
This repo includes example traces derived from the
[Tau benchmark](https://github.com/sierra-research/tau-bench), which is licensed under the MIT
License. The complete Tau Bench license is preserved in
[`third_party/tau-bench-LICENSE.txt`](third_party/tau-bench-LICENSE.txt).

Configure inference before the first run. Copy the environment template, then set
`INSIGHT_AGENT_API_KEY` for the selected model provider:

```bash
cp .env.example .env
```

If you use an OpenAI-compatible gateway, also set `INSIGHT_AGENT_MODEL` and
`INSIGHT_AGENT_API_BASE` in `.env`. The API base commonly ends in `/v1`, but this is
provider-specific: use the base URL documented by your inference provider rather than guessing the
suffix or supplying the full `/chat/completions` endpoint. Direct providers such as OpenAI or
Anthropic normally do not need `INSIGHT_AGENT_API_BASE`.

Then run the Analyst:

```bash
uv sync --locked

uv run insight-agent --config examples/insight-analyst.yaml
```

`uv sync --locked` creates `.venv`, installs the package and development tools,
and reproduces the dependency versions committed in `uv.lock`.

The command prints the complete Insight collection as YAML and writes it to
`insights.yml`. Insight compilation requires an API key.

The architecture is intentionally small:

```text
TraceLoader -> TraceSnapshot -> EvidenceStream(s) -> optional code validation -> InsightsGeneration -> Insights
```

### Reusable run configuration

The default command can load its trace source, evidence-stream selection,
output path, and non-secret model settings from YAML. YAML is optional, and
explicit CLI options override individual file values. See
[run configuration](docs/configuration.md) for the full schema and examples.

### Code layout

```text
src/insight_agent/
├── traces.py             # normalized Trace, Span, and TraceSnapshot contracts
├── trace_loaders/        # loader contracts and source-specific trace loaders
├── evidence_streams/     # evidence-stream contracts and implementations
│   ├── anomaly_and_patterns/  # anomaly-and-pattern stream implementation
│   ├── tool_issues/           # tool-issue stream implementation and coverage helper
│   └── eval_failure_patterns.py  # LLM review of evaluation-linked failures
├── insights_generation/  # LLM-backed synthesis and its configuration
└── cli/                  # command orchestration and final YAML output
```

`traces.py` is intentionally the only shared domain module at package top level. The other
implementation modules live with the stage or interface that owns them.

The [anomaly-and-pattern](src/insight_agent/evidence_streams/anomaly_and_patterns/README.md)
and [tool-issue](src/insight_agent/evidence_streams/tool_issues/README.md) packages document
their own configuration, analysis, and outputs. The canonical input format is documented with
the public [`Trace` model](src/insight_agent/traces.py).

### Reading the output

The CLI prints and writes a YAML list of final Insights. Each Insight contains
a name, description, and the trace references that support it.

## Analyze your own traces

The Analyst can read live projects or native exports from LangSmith, Langfuse, and MLflow. Every
source is normalized before the same evidence streams run. First set the model credential used to
produce the final insights:

```bash
export INSIGHT_AGENT_API_KEY=<model-provider-api-key>
```

Then choose one integration below. Each YAML snippet is a complete `insight-analyst.yaml` file;
replace the example project, endpoint, and time range with your own values.

### LangSmith

```bash
uv sync --locked --extra langsmith
export LANGSMITH_API_KEY=<langsmith-api-key>
```

```yaml
trace:
  max_traces: 100
  langsmith:
    project: my-agent
    start_time: 2026-09-01T00:00:00Z
output_path: insights.yml
evidence_streams:
  anomaly_and_patterns: {}
  tool_issues: {}
```

```bash
uv run --no-sync insight-agent --config insight-analyst.yaml
```

See [LangSmith configuration](docs/configuration.md#langsmith) for filters, self-hosted endpoints,
workspace selection, native exports, and supported versions.

### Langfuse

```bash
uv sync --locked --extra langfuse
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://langfuse.example.com
```

```yaml
trace:
  max_traces: 100
  langfuse:
    from_timestamp: 2026-09-01T00:00:00Z
    to_timestamp: 2026-09-02T00:00:00Z
output_path: insights.yml
evidence_streams:
  anomaly_and_patterns: {}
  tool_issues: {}
```

```bash
uv run --no-sync insight-agent --config insight-analyst.yaml
```

Langfuse API keys identify the project, so no project name is needed. See
[Langfuse configuration](docs/configuration.md#langfuse) for advanced filters, complete native
exports, tool catalogs, and the currently supported v3 server and SDK versions.

### MLflow

```bash
uv sync --locked --extra mlflow
export MLFLOW_TRACKING_URI=https://mlflow.example.com
```

```yaml
trace:
  max_traces: 100
  mlflow_experiment:
    experiment: my-agent
output_path: insights.yml
evidence_streams:
  anomaly_and_patterns: {}
  tool_issues: {}
```

```bash
uv run --no-sync insight-agent --config insight-analyst.yaml
```

See [MLflow configuration](docs/configuration.md#mlflow) for authentication, search filters,
pagination limits, and native exports.

### Canonical JSONL

For an already normalized file containing one serialized `Trace` per line:

```bash
uv sync --locked
uv run --no-sync insight-agent \
  --trace.filesystem.path traces.jsonl \
  --evidence-streams.anomaly-and-patterns '{}' \
  --evidence-streams.tool-issues '{}'
```

If your source is not supported, the repository includes a
[trace-loader skill](.claude/skills/trace-loader/SKILL.md) for implementing another adapter.

Progress is written to stderr while the final YAML is printed to stdout and saved to
`output_path` (by default, `insights.yml`).

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for validation, dependency licensing, and release instructions.

## License

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This project is licensed under the [Apache License, Version 2.0](LICENSE). See
[NOTICE](NOTICE) for project attributions,
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for human-readable dependency license
disclosures, and [third_party/licenses.jsonl](third_party/licenses.jsonl) for the machine-readable
inventory.

This project is currently not accepting contributions.
