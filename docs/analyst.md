# The Analyst stage

The anomaly-and-pattern and tool-issue streams stop deliberately short of a conclusion. Each stream
projects its native analysis into candidate `Problem` objects containing a
description and supporting trace IDs. Only the Analyst LLM turns those Problems
into formal Insights.

`run-analyst` sends the Problems from every configured evidence stream to a
model through [litellm](https://github.com/BerriAI/litellm), gives it a tool for
inspecting normalized traces from the snapshot, and writes back the authored
Insights.

```bash
cp .env.example .env      # then fill in INSIGHT_AGENT_API_KEY

uv run insight-agent --config analyst.yaml      # includes the Analyst when enabled
cat out/analyst/insights.json
```

The default config-driven command runs the Analyst when `analyst.enabled` is
true, and `demo` runs it by default. Because a run that stops at evidence is a
partial run, an unusable Analyst is a hard failure rather than a silent skip:
if litellm is missing or no key resolves, the command exits 1 and says so.
`--no-analyst.enabled` provides a one-off override for a deterministic-only
run, including `demo`.

## Output

Each Insight has exactly three fields:

```json
[
  {
    "name": "Agent invents tool names",
    "description": "The agent repeatedly calls tools that are absent from the active catalog...",
    "trace_ids": ["run-0117", "run-0244", "run-0391"]
  }
]
```

`out/analyst/` also contains:

| File | What it is |
|---|---|
| `insights.json` | the deliverable |
| `prompt.md` | the opening prompt that was sent, system and user turns |
| `run.json` | model, token usage, prompt version, Problem counts, corpus description, and every trace the model fetched |
| `analyst_raw.txt` | only on a parse failure — the unparsed response, kept rather than discarded |

## What the Analyst sees

The Analyst receives every `EvidenceStreamResult`, grouped by stream, but only
its stream name and Problems. It does not import or interpret stream-native
artifact types.

The anomaly-and-pattern stream creates Problems for statistical outliers, recurring strict failures, and
failures whose normalized message crosses tool boundaries. Its digest,
features, and clustering output remain diagnostic artifacts. The tool-issue stream creates one
Problem per recurrence-qualified tool-issue card. Audit-only cards remain in
`cards.json`; `--all-cards` also exposes them as Problems.

### And normalized traces, on request

The Problems are a *map*, not the territory. They are compact and lossy by
design. An Analyst reading only the descriptions can restate, rank, and merge
them, but it cannot discover a mechanism that requires inspecting the actual
messages, calls, and results.

So the model is also given a tool:

```
fetch_traces(trace_ids, max_chars_per_result=2000) -> the full canonical traces
```

It takes a list, so one call can pull several traces to compare. Each result is
truncated to `max_chars_per_result`, which the model raises when it needs to
read a large payload in full; the whole response is capped at 120,000
characters regardless. IDs that do not exist come back under `not_found` rather
than as an empty trace, so a hallucinated ID is visible to the model as a miss
instead of silently reading as "nothing happened here".

The loop runs at most `--max-tool-rounds` times (default 8). On the last round
the tools are withdrawn and the model is asked once more for its answer, so an
enthusiastic reader ends the turn with Insights rather than with a truncated
tool call.

**The tool is offered only if the prompt names it** and trace records are
available. A custom prompt that never mentions `fetch_traces` gets no tools —
there would be nothing telling the model what it is for.

### Size

The opening prompt scales with the number and size of candidate Problems, not
with raw trace or finding count. The anomaly-and-pattern stream caps the supporting IDs on its
aggregate anomaly Problem, and the tool-issue stream's Problem count is bounded by distinct
`(issue_type, mechanism_key)` pairs. Corpus-scale prompt measurements should be
tracked as additional streams are added.

What is *not* bounded is the rest of the turn. Cost now depends on how many
traces the model opens and how large they are, which is a property of the
corpus and of the model's curiosity rather than of the pipeline. `--max-tool-rounds`
is the ceiling, `run.json` records what was actually fetched, and the
per-result truncation keeps a single 38k-call trace from swallowing the
context. Budget for more than the opening-prompt figure above.

## Cost control

`--dry-run` assembles the prompt, writes `prompt.md`, prints a token estimate,
and exits without calling anything:

```bash
uv run insight-agent run-analyst traces.jsonl -o out --dry-run
```

Use it to inspect exactly what would be sent before spending anything.

## Configuration

Three settings, resolved **CLI flag → exported environment variable → `.env`**:

| Setting | `.env` variable | Flag |
|---|---|---|
| Model | `INSIGHT_AGENT_MODEL` | `--model` |
| Endpoint | `INSIGHT_AGENT_API_BASE` | `--api-base` |
| Credential | `INSIGHT_AGENT_API_KEY` | — (never a flag; it would land in shell history) |

`.env` is gitignored; `.env.example` is the committed template. `OPENAI_API_KEY`,
`OPENAI_API_BASE` and `ANTHROPIC_API_KEY` are honoured as fallbacks, so an
environment already configured for litellm needs no `.env` at all.

An unfilled `KEY=` line in a `.env` is treated as absent rather than as an empty
string — a template left half-filled will not mask a real exported credential.

**`--env-file` is only authoritative before litellm loads.** Importing litellm
runs `dotenv.load_dotenv()` as a side effect, which pulls any `.env` in the
current directory into the environment. The credential check therefore resolves
config *before* importing litellm, so an explicit `--env-file` wins — but be
aware that once litellm is imported, a stray `.env` beside you has already been
merged into `os.environ`. Run from a clean directory if that matters.

### Providers

Any [litellm](https://docs.litellm.ai/docs/providers) model string works.
Anthropic direct is the package default:

```dotenv
INSIGHT_AGENT_MODEL=anthropic/claude-opus-5
INSIGHT_AGENT_API_KEY=sk-ant-...
```

OpenAI, Bedrock and the rest follow the same pattern:

```bash
uv run insight-agent run-analyst traces.jsonl -o out --model openai/gpt-5.2
uv run insight-agent run-analyst traces.jsonl -o out --model bedrock/anthropic.claude-opus-5
```

### OpenAI-compatible gateways

Any endpoint that speaks the OpenAI API works through an `openai/` model prefix
plus an `api_base`:

```dotenv
INSIGHT_AGENT_API_BASE=https://your-gateway.example.com/v1
INSIGHT_AGENT_MODEL=openai/your-model-name
INSIGHT_AGENT_API_KEY=...
```

One gotcha worth knowing: litellm strips the **first** `openai/` prefix to pick
the provider and sends the rest as the model name in the request body. If your
gateway's own model names already begin with `openai/`, you need the prefix
twice. When a gateway rejects the model name its error usually names what it
actually received ("Tried to access ..."), which tells you whether to add or
drop a prefix. `--dry-run` prints the exact string before anything is spent.

### Sampling parameters are opt-in

**Nothing is sent by default.** `temperature`, `top_p` and `top_k` are *removed*
on Claude Opus 5 and return a 400 — passing one "to be safe" is the easiest way
to break the call. Behaviour is steered through the prompt instead, and there is
a test asserting none of them appear in the litellm kwargs.

`--temperature 0.7` sends it explicitly, for a provider that accepts it. Note
that Insight authoring wants low variance, so the default of not sending it is
usually also the right choice.

## Re-running without recomputing

`--digest` and `--cards` bypass both evidence streams and rebuild Problems from existing
artifacts, so a prompt change can be re-issued against a frozen evidence set:

```bash
uv run insight-agent run-analyst traces.jsonl -o out \
  --digest out/anomaly_and_patterns/digest.md --cards out/tool_issues/cards.json
```

The config-driven run reuses the in-memory `EvidenceStreamResult` values it
just computed.

## Reading the output honestly

**This is the only non-deterministic artifact in the repo.** Everything else
reproduces within a scikit-learn version; this does not, and on Opus 5
temperature cannot even be set to pretend otherwise. `prompt.md` and `run.json`
are written beside every result so a given `insights.json` can be traced back
to the exact evidence and prompt version that produced it.

**Zero Insights is a valid outcome.** The spec says the Analyst "may file zero
Insights", and in the recorded trials one run produced 6 candidates of which
strict review accepted 0. The prompt says so explicitly, because a model that is not
told this will manufacture something. A small corpus, or one where nothing
recurred across enough independent cases, legitimately yields `[]`.

**Trace IDs are not validated.** There is no checker gating the output. The CLI
prints an informational count of cited IDs that appear neither in the evidence
nor among the traces the model fetched — but it does not fail the run. If any
cited ID is absent from the corpus *entirely*, the summary says so in stronger
terms, because that is fabrication rather than an undocumented lookup. Invented
pointers are the failure that killed all six candidate Insights in those
recorded trials, so if that warning ever appears, treat the run as unreliable.

**The shipped prompt treats Problems as evidence, not conclusions.** It directs
the Analyst to inspect supporting traces, consolidate only when mechanisms
match, and avoid upgrading anomaly or rule-match evidence into unsupported root
cause or impact claims. Judge descriptions on the cited traces; evidence
streams identify where to investigate but do not measure consequences.

## Prompt

`src/insight_agent/insights_generation/prompts/analyst.md`, packaged with the wheel. Git records
the prompt's source revision; `run.json` records its name and `prompt.md` echoes
the assembled text sent to the model.

To try your own, drop a `<name>.md` beside it and pass `--prompt-version <name>`.
Two requirements:

- It must contain the marker `{evidence}`, which is where evidence-stream
  Problems are substituted. A template without it is an
  error rather than an append, so a typo cannot send a paid request whose
  evidence landed somewhere the prompt never refers to.
- Name `fetch_traces` in it to get the tool loop; omit it for a single-shot,
  evidence-only run.

If a prompt does not state the output contract itself — detected by looking for
`trace_ids` — the contract is appended to the user turn, so a rough prompt still
returns parseable JSON.
