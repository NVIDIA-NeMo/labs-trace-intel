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

There are 2 current evidence streams, and we can expect to add more in the future:

1) **Anomaly and Pattern Analysis** - Which traces are unusual, and which patterns recur?

    A set of 11 features are extracted from each trace and the resulting feature vector feeds an Isolation Forest model which identifies statistical anomalies.

    Traces are clustered based on TFIDF after structural metadata projection and error extraction. 

2) **Tool Issue Detection** - Which tool use problems recur across the trace sample?

    The tools in each trace are checked against a set of deterministic rules to detect specific tool calling issues. 

After the evidence streams run, each returns candidate `Problem` objects with a description and supporting trace IDs. Those Problems are passed to the Analyst Agent to guide its analysis. The Analyst can also look up normalized supporting traces from the same snapshot and uses that evidence as a starting point for more detailed exploration and synthesis.

The Analyst Agent ultimately produces a set of "insights" which are meant to describe a recurring and actionable problem observed from the trace corpus. 

## Running the Agent on Example Traces
This repo includes example traces from the [Tau benchmark](https://github.com/sierra-research/tau-bench).

First follow the .env.example to configure some keys for the LLM bits. 

Then run the Analyst:

```bash
uv sync --locked

uv run insight-agent --config examples/analyst.yaml
open out/index.md
```

`uv sync --locked` creates `.venv`, installs the package and development tools,
and reproduces the dependency versions committed in `uv.lock`.

This will write the outputs to an /out directory

To see the
deterministic stages alone, with no key and no cost:

```bash
uv run insight-agent --config examples/analyst.yaml --no-analyst.enabled
```

### Run directly from Git

UV can build and run the CLI without cloning the repository:

```bash
uvx --from 'git+https://github.com/NVIDIA/nemo-platform-insights-preview.git' \
  insight-agent demo --no-analyst.enabled
```

The architecture is intentionally small:

```text
TraceLoader -> TraceSnapshot -> EvidenceStream(s) -> InsightsGeneration -> Insights
```

### Reusable run configuration

The default command loads its trace, evidence-stream selection, output, and
non-secret Analyst settings from YAML. Explicit CLI options still override
the file, which is useful for one-off experiments. See
[run configuration](docs/configuration.md) for the full schema and examples.

### Code layout

```text
src/insight_agent/
├── traces.py             # normalized Trace, Span, and TraceSnapshot contracts
├── trace_loaders/        # loader contracts and source-specific trace loaders
├── evidence_streams/     # evidence-stream contracts and implementations
│   ├── anomaly_and_patterns/  # anomaly-and-pattern stream implementation
│   └── tool_issues/           # tool-issue stream implementation and coverage helper
├── insights_generation/  # LLM-backed synthesis and its configuration
└── cli/                  # command orchestration and artifact writing
```

`traces.py` is intentionally the only shared domain module at package top level. The other
implementation modules live with the stage or interface that owns them.

The [anomaly-and-pattern](src/insight_agent/evidence_streams/anomaly_and_patterns/README.md)
and [tool-issue](src/insight_agent/evidence_streams/tool_issues/README.md) packages document
their own configuration, analysis, and outputs. The canonical input format is documented with
the public [`Trace` model](src/insight_agent/traces.py).

### Reading the outputs
./out/analyst contains the final output in insights.json. It also contains a prompt.md which is the full interpolated prompt sent to the Analyst Agent. 

./out/anomaly_and_patterns contains the artifacts from the anomaly-and-pattern evidence stream.
`digest.md` retains its native diagnostic summary; `problems.json` contains the generic handoff sent to Insights generation.

out/tool_issues contains the artifacts from the tool-issue evidence stream.
`cards.json` retains all native tool-issue cards; `problems.json` contains the recurrence-qualified handoff sent to Insights generation.

## Running the agent on your own traces

Evidence streams consume a normalized `TraceSnapshot`. Choose the source loader that matches
where your traces already live; neither path changes the downstream analysis.

### Read an MLflow experiment directly

Install the optional lightweight MLflow client, point it at your tracking server, and name the
experiment:

```bash
uv sync --locked --extra mlflow

# In analyst.yaml, select the live loader:
# trace:
#   mlflow_experiment:
#     experiment: my-agent
#     tracking_uri: https://mlflow.example.com
  uv run --no-sync insight-agent --config analyst.yaml
```

Authentication uses the MLflow SDK's standard environment variables. Set
`MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` for HTTP Basic authentication, or
`MLFLOW_TRACKING_TOKEN` for a bearer token; Basic authentication takes precedence when both are
present. TLS options include `MLFLOW_TRACKING_SERVER_CERT_PATH`, `MLFLOW_TRACKING_CLIENT_CERT_PATH`,
and `MLFLOW_TRACKING_INSECURE_TLS` (not recommended). See MLflow's
[authentication and encryption documentation](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/#authentication-and-encryption).
There are intentionally no Analyst auth flags, which keeps credentials in MLflow's configuration;
provider-specific or custom auth may require its corresponding MLflow extra or plugin.

Add `filter: "trace.status = 'ERROR'"` under `trace.mlflow_experiment` to select a subset. MLflow's
[trace search filters](https://mlflow.org/docs/latest/genai/tracing/search-traces/) can target
timestamps, names, span types, tags, and metadata. Set `trace.max_traces` in YAML (or override it
with `--trace.max-traces`) to control the final number of complete traces materialized in memory;
it defaults to 10,000. The loader transparently requests
sequential pages of at most 500 traces, matching MLflow's documented
[SearchTraces limit](https://mlflow.org/docs/latest/api_reference/rest-api.html#searchtracesv3).
Large or span-heavy traces may exhaust local memory before the trace-count limit, so production
runs should use a coherent time window plus stable tags or metadata rather than an unbounded query.

If the traces are already exported from MLflow, pass the native JSON directly:

```bash
uv sync --locked --extra mlflow
# In analyst.yaml, select the export loader:
# trace:
#   mlflow_export:
#     path: traces.json
uv run --no-sync insight-agent --config analyst.yaml
```

The loader accepts complete JSON from `mlflow traces search --output json`, `mlflow traces get`,
or MLflow's `Trace.to_json()` / `Trace.to_dict()` serializers. It also accepts arrays of native
trace objects and JSONL containing one native trace per line. Exports must contain spans; the loader
rejects metadata-only results. Export-time selection is authoritative: the offline loader does not
apply MLflow filters or follow a continuation token, and it reads the complete file before applying
`trace.max_traces`.

### Read traces from the filesystem

Filesystem input is JSONL with one serialized `Trace` per non-empty line. `FSDataLoader`
parses each line directly with the Pydantic model; it does not normalize or remap fields.

If your traces are not in a compatible format, the repo ships with a skill for implementing a
source-specific loader: [`.claude/skills/trace-loader/SKILL.md`](.claude/skills/trace-loader/SKILL.md).

After converting your traces, validate the format:

```bash
# Check the format
uv run insight-agent validate --traces traces.jsonl

# Check what your data actually supports
uv run insight-agent coverage traces.jsonl
```

Then run the full analysis:
```bash
uv run insight-agent --config analyst.yaml
```

Results are written to the `out` directory.

For a dry run or a run without LLM synthesis:
```bash
uv run insight-agent run-analyst traces.jsonl -o out --dry-run
uv run insight-agent --config analyst.yaml --no-analyst.enabled
```

## Validation

When adding NVIDIA-authored files or changing dependencies, update the tracked
licensing artifacts first:

```bash
make update-copyright-headers
make update-licenses
```

License generation requires `osv-scanner` on `PATH`; CI pins the same 2.3.3
release used by NeMo Platform. `make update-licenses` refreshes the OSV
dependency inventory.

Run the same read-only checks used by CI:

```bash
uv lock --check
make check-copyright-headers
make check-licenses
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

## License

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This project is licensed under the [Apache License, Version 2.0](LICENSE). See
[NOTICE](NOTICE) for project attributions and
[third_party/licenses.jsonl](third_party/licenses.jsonl) for the dependency license inventory.

This project is currently not accepting contributions.
