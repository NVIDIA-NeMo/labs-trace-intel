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

The focused CLI commands retain the `run-ia2` and `run-ia3` names from their research lineage.

After the evidence streams run, each returns candidate `Problem` objects with a description and supporting trace IDs. Those Problems are passed to the Analyst Agent to guide its analysis. The Analyst can also look up normalized supporting traces from the same snapshot and uses that evidence as a starting point for more detailed exploration and synthesis.

The Analyst Agent ultimately produces a set of "insights" which are meant to describe a recurring and actionable problem observed from the trace corpus. 

## Running the Agent on Example Traces
This repo includes example traces from the [Tau benchmark](https://github.com/sierra-research/tau-bench).

First follow the .env.example to configure some keys for the LLM bits. 

Then run the Analyst:

```bash
uv sync --locked

uv run insight-agent run-all examples/tau_bench_traces.jsonl -o out
open out/index.md
```

`uv sync --locked` creates `.venv`, installs the package and development tools,
and reproduces the dependency versions committed in `uv.lock`.

This will write the outputs to an /out directory

To see the
deterministic stages alone, with no key and no cost:

```bash
uv run insight-agent run-all examples/tau_bench_traces.jsonl \
  -o out --no-analyst
```

### Run directly from Git

UV can build and run the CLI without cloning the repository:

```bash
uvx --from 'git+https://github.com/NVIDIA/nemo-platform-insights-preview.git' \
  insight-agent demo --no-analyst
```

The architecture is intentionally small:

```text
TraceLoader -> TraceSnapshot -> EvidenceStream(s) -> InsightsGeneration -> Insights
```

`run-ia2` and `run-ia3` remain useful for focused development and ablation.

### Code layout

```text
src/insight_agent/
├── adapters/             # source-specific conversion helpers
├── traces.py             # normalized Trace, Span, and TraceSnapshot contracts
├── evidence_streams/     # evidence-stream contracts and shared configuration
│   ├── anomaly_and_patterns/  # IA2 stream implementation
│   ├── common/                # shared trace input contract and loader
│   └── tool_issues/           # IA3 stream implementation and coverage helper
├── insights_generation/  # LLM-backed synthesis and its configuration
└── cli/                  # command orchestration and artifact writing
```

`traces.py` is intentionally the only shared domain module at package top level. The other
implementation modules live with the stage or interface that owns them.

The [anomaly-and-pattern](src/insight_agent/evidence_streams/anomaly_and_patterns/README.md)
and [tool-issue](src/insight_agent/evidence_streams/tool_issues/README.md) packages document
their own configuration, analysis, and outputs. Shared input documentation lives in
[evidence-stream common](src/insight_agent/evidence_streams/common/README.md).

### Reading the outputs
./out/analyst contains the final output in insights.json. It also contains a prompt.md which is the full interpolated prompt sent to the Analyst Agent. 

./out/ia2 contains the artifacts from the anomaly-and-pattern evidence stream.
`digest.md` retains its native diagnostic summary; `problems.json` contains the generic handoff sent to Insights generation.

out/ia3 contains the artifacts from the tool-issue evidence stream.
`cards.json` retains all native tool-issue cards; `problems.json` contains the recurrence-qualified handoff sent to Insights generation.

## Running the agent on your own traces

**Step 1: Load traces into the normalized format**
Evidence streams consume a normalized `TraceSnapshot`. The current CLI uses
`InsightTraceLoader`, so source traces must first be converted to `insight-trace/v1`.

If your traces are already in OpenAI or Anthropic message format you can use the built in adapter

```bash
uv run insight-agent adapt-messages my-conversations.json -o traces.jsonl
```

If your traces are not in a compatible format, the repo ships with a skill for writing a custom adapter. Find it at .claude/skills/insight-trace-adapter/SKILL.md

After you've converted your traces you can validate the format with:

```bash
# Check the format
uv run insight-agent validate traces.jsonl

# Check what your data actually supports
uv run insight-agent coverage traces.jsonl
```

Then run the full analysis with:
```bash
uv run insight-agent run-all traces.jsonl -o out
```

Results will be written to the /out directory. 

If you want to do a dry run or a run without the LLM synthesis you can use
```bash
uv run insight-agent run-analyst traces.jsonl --agent "My agent" -o out --dry-run
uv run insight-agent run-all traces.jsonl -o out --no-analyst
```

## Validation

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```
