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

There are 3 current evidence streams, and we can expect to add more in the future: 

1) **Anomaly Detection** - Which traces are unusual?

    A set of 11 features are extracted from each trace and the resulting feature vector feeds an Isolation Forest model which identifies statistical anomalies.

2) **Recurring Patterns** - Which trace patterns repeat?

    Traces are clustered based on TFIDF after structural metadata projection and error extraction. 

3) **Tool Analysis** - Which tool use problems recur across the trace sample?

    The tools in each trace are checked against a set of deterministic rules to detect specific tool calling issues. 

Evidence streams 1 and 2 happen as part of "IA2", and evidence stream 3 happens as part of "IA3".

After each of the evidence streams run, their output is passed to the Analyst Agent to guide its analysis. The Analyst Agent also has tools that it can use to look up raw traces from a database. The idea is that the evidence streams in the pre-processing step identify potentially problematic traces, and then the Analyst Agent can use that evidence as a starting point for more detailed exploration and synthesis. It is instructed not to rely exclusively on the evidence streams themselves.

The Analyst Agent ultimately produces a set of "insights" which are meant to describe a recurring and actionable problem observed from the trace corpus. 

## Running the Agent on Example Traces
This repo includes some example traces from the [Tau benchmark](https://github.com/sierra-research/tau-bench) that can be used to demo the pipeline. 

First follow the .env.example to configure some keys for the LLM bits. 

Then you can run the pipeline against the example traces.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

insight-agent run-all examples/tau_bench_traces.jsonl -o out
open out/index.md
```

This will write the outputs to an /out directory

To see the
deterministic stages alone, with no key and no cost:

```bash
insight-agent run-all examples/tau_bench_traces.jsonl -o out --no-analyst
```

### Reading the outputs
./out/analyst contains the final output in insights.json. It also contains a prompt.md which is the full interpolated prompt sent to the Analyst Agent. 

./out/ia2 contains the raw artifacts from the anomaly detection and clustering evidence streams. 
Those intermediate artifacts are aggregated into a digest.md which is injected into the system prompt of the Analyst. 

out/ia3 contains the raw artifacts from the tool issue detection evidence stream. 
Those artifacts are aggregated into cards.json which is injected into the system prompt of the Analyst. 

## Running the agent on your own traces

**Step 1: Convert Traces to a Common Format**
Because the pipeline relies on deterministic parsing and checks, the traces must first be converted to a common format. 

If your traces are already in OpenAI or Anthropic message format you can use the built in adapter

```bash
insight-agent adapt-messages my-conversations.json -o traces.jsonl
```

If your traces are not in a compatible format, the repo ships with a skill for writing a custom adapter. Find it at .claude/skills/insight-trace-adapter/SKILL.md

After you've converted your traces you can validate the format with:

```bash
# Check the format
insight-agent validate traces.jsonl

# Check what your data actually supports
insight-agent coverage traces.jsonl
```

Then you can run the full pipeline with 
```bash
insight-agent run-all traces.jsonl -o out
```

Results will be written to the /out directory. 

If you want to do a dry run or a run without the LLM synthesis you can use
```bash
insight-agent run-analyst traces.jsonl --agent "My agent" -o out --dry-run
insight-agent run-all traces.jsonl -o out --no-analyst 
```
