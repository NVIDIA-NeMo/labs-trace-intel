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

First follow the .env.example to configure some keys for the LLM bits. 

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
TraceLoader -> TraceSnapshot -> EvidenceStream(s) -> InsightsGeneration -> Insights
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
a name, description, and the trace references that support it. Intermediate
evidence-stream artifacts are passed to Insight compilation in memory rather
than written as a directory tree.

## Running the agent on your own traces

Evidence streams consume a normalized `TraceSnapshot`. Choose the source loader that matches
where your traces already live; none of these paths changes the downstream analysis.

### Read LangSmith traces

Install the optional LangSmith client, set `LANGSMITH_API_KEY`, and select either the LangSmith API
or a trace export file created by `langsmith trace export --full`:

```yaml
trace:
  max_traces: 100
  langsmith:
    project: my-agent
    start_time: 2026-09-01T00:00:00Z

# Or replace langsmith with:
# langsmith_trace_export_file:
#   path: traces/
```

```bash
uv sync --locked --extra langsmith
export LANGSMITH_API_KEY=<api-key>
uv run --no-sync insight-agent --config insight-analyst.yaml
```

The LangSmith Trace Loader currently supports LangSmith's v1 query API as tested against
self-hosted LangSmith 0.15. See [run configuration](docs/configuration.md#langsmith) for endpoint
and workspace environment variables, filters, CLI overrides, export requirements, and
compatibility details.

### Read Langfuse traces

#### Live project

Install the optional Langfuse client and provide project-scoped API keys through the environment:

```bash
uv sync --locked --extra langfuse

export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://langfuse.example.com
```

Select a bounded trace timestamp window in YAML. The loader pages through matching trace summaries
and then fetches every selected trace with all of its observations, so `max_traces` never splits a
trace:

```yaml
trace:
  max_traces: 100
  langfuse:
    from_timestamp: 2026-08-01T00:00:00Z
    to_timestamp: 2026-08-02T00:00:00Z
    filter: >-
      [{"type":"string","column":"environment","operator":"=","value":"production"}]
```

The keys identify the Langfuse project, so there is no separate project field. Set
`LANGFUSE_BASE_URL` for the self-hosted deployment, or use the equivalent
`trace.langfuse.base_url` YAML/CLI setting; an explicit setting takes precedence. For example,
`--trace.langfuse.base-url https://langfuse.example.com` overrides the environment for one run.
The CLI loads these variables from the shell or a local `.env` file.

`filter` accepts Langfuse's JSON-encoded advanced filter conditions. The loader combines them with
its required time bounds; `from_timestamp` is inclusive and `to_timestamp` is exclusive. Filters
apply to trace fields such as environment, name, session, tags, metadata, cost, and error counts.

Tool-contract analysis reads available tool definitions from the top-level `input.tools` array on
`GENERATION` observations. Langfuse normalizes supported instrumentation fields, including
OpenTelemetry's `gen_ai.tool.definitions`, into this API-visible representation. All explicit
generation catalogs must agree because the canonical trace model currently has one trace-wide
catalog. A root `AGENT` catalog is accepted only as a compatibility fallback for reduced traces and
exports with no generation observations. Missing, malformed, or conflicting generation catalogs
make schema-dependent tool rules abstain rather than infer what the model could access.

#### Native export files (offline)

Use the [Langfuse Python SDK](https://github.com/langfuse/langfuse-python), installed by the
`langfuse` extra above, to export a time range. This example exports all traces from September 9
UTC: the start is inclusive and the end is exclusive. Change the dates for your corpus.

With the project environment variables above set in your shell, run:

```bash
uv run --locked --extra langfuse python - <<'PY'
from datetime import UTC, datetime
from itertools import count
from langfuse import Langfuse

client = Langfuse(tracing_enabled=False)
start = datetime(2026, 9, 9, tzinfo=UTC)
end = datetime(2026, 9, 10, tzinfo=UTC)

with open("langfuse-traces.jsonl", "x", encoding="utf-8") as output:
    for page in count(1):
        result = client.api.trace.list(
            from_timestamp=start, to_timestamp=end,
            page=page, limit=100, order_by="timestamp.asc",
        )
        for summary in result.data:
            trace = client.api.trace.get(summary.id)
            output.write(trace.json(by_alias=True) + "\n")
        if page >= result.meta.total_pages:
            break
PY
```

The SDK reads `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` from the
environment. Unlike the Analyst CLI, this snippet does not load `.env`. It fetches every page;
`get()` retrieves each trace's full observation tree because `list()` returns summaries.
The file uses native v3 records, not Analyst-format traces, and `"x"` prevents overwriting an
existing export. Exporting needs a Langfuse connection; subsequent file-based analysis does not:

```bash
uv run insight-agent \
  --trace.langfuse-export.path langfuse-traces.jsonl \
  --trace.max-traces 200 \
  --evidence-streams.tool-issues '{}' \
  --output-path insights.yml
```

No conversion is needed. The loader accepts these SDK exports, raw v3
`GET /api/public/traces/{id}` response bodies, and Langfuse CLI JSON response envelopes.
You can also supply a single trace-detail `.json` file or a directory of `.json`/`.jsonl` files
(only immediate files are read). The equivalent YAML configuration is:

```yaml
trace:
  max_traces: 200
  langfuse_export:
    path: ./langfuse-traces.jsonl
```

`max_traces` defaults to 100 and selects the newest complete traces by timestamp, breaking ties
by trace ID, then analyzes them oldest-first. All exported records are validated before applying
the limit. Duplicate IDs, invalid records, and missing observation parents fail the load.

Trace-list responses and UI table/CSV exports are not supported: use complete trace-detail
responses with embedded observations, not observation IDs or summary rows. Offline loading needs
the same `langfuse` extra, but no Langfuse credentials, base URL, or network access. Insight
compilation still needs the configured model service and `INSIGHT_AGENT_API_KEY`.

#### Supported versions

This research-preview integration supports only the self-hosted Langfuse v3 API contract that we
can test. It is validated against Langfuse 3.205.1 with the Python SDK 3.15. It does not support
Langfuse v4. A future migration will replace the v3 trace endpoints with the v4 observations-first
API once we have a v4 environment to test against.

### Read an MLflow experiment directly

Install the optional lightweight MLflow client, then select the tracking server and experiment in
the run configuration:

```bash
uv sync --locked --extra mlflow

# In insight-analyst.yaml, select the live loader:
# trace:
#   mlflow_experiment:
#     experiment: my-agent
#     tracking_uri: https://mlflow.example.com
uv run --no-sync insight-agent --config insight-analyst.yaml
```

MLflow's
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
# In insight-analyst.yaml, select the export loader:
# trace:
#   mlflow_export:
#     path: traces.json
uv run --no-sync insight-agent --config insight-analyst.yaml
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

After converting your traces, run the analysis. The filesystem loader validates
the canonical trace records while loading them:

```bash
uv run insight-agent --config insight-analyst.yaml
```

Results are printed and written to the configured `output_path`, which defaults
to `insights.yml`.

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
