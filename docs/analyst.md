# The Analyst stage

IA2 and IA3 stop deliberately short of a conclusion. The specification is
explicit about it — *"only the Analyst LLM authors formal Insights"* — and the
IA2 digest ends with five numbered "Analyst authoring rules" addressed to a
reader that, until now, this repo did not contain.

`run-analyst` is that reader. It sends the IA2 digest and the recurrence-
qualified IA3 cards to a model through [litellm](https://github.com/BerriAI/litellm),
gives it a tool for pulling raw traces out of the corpus, and writes back the
authored Insights.

```bash
cp .env.example .env      # then fill in INSIGHT_AGENT_API_KEY

insight-agent run-all traces.jsonl -o out      # includes the Analyst
cat out/analyst/insights.json
```

`run-all` and `demo` run the Analyst by default. Because a run that stops at
evidence is a partial run, an unusable Analyst is a hard failure rather than a
silent skip: if litellm is missing or no key resolves, the command exits 1 and
says so. `--no-analyst` is the supported way to ask for IA2/IA3 only.

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
| `run.json` | model, token usage, prompt version, card counts, corpus description, and every trace the model fetched |
| `analyst_raw.txt` | only on a parse failure — the unparsed response, kept rather than discarded |

## What the Analyst sees

Two streams up front, kept separate, exactly as the spec requires:

1. **The IA2 digest**, verbatim — including its own reader contract and
   authoring rules.
2. **The IA3 cards** as JSON, so `card_id`, `independent_case_count`,
   `attribution`, and every `representative_evidence` pointer stay quotable.

**Only cards marked `eligible_for_analyst` are shown by default.** Audit-only
cards are real findings that did not recur across three independent logical
cases; showing them by default would hand the model a pile of single-case
findings and invite it to promote them. The prompt states how many were
withheld, so the evidence set is never silently filtered. `--all-cards`
includes them.

`findings.json` is deliberately **not** sent. It grows linearly with finding
count — 67 findings is already 60 KB — and `cards.json` is its bounded
projection.

### And the raw traces, on request

The digest and the cards are a *map*, not the territory. Both are lossy by
design: the digest reports that a trace is anomalous without saying what
happened in it, and a card quotes one line of observation per representative
call. An Analyst reading only those two streams can restate them, rank them,
and merge them — but it cannot discover a mechanism neither stage was built to
look for.

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

The **opening** prompt is bounded, not linear in corpus size — the digest caps
every variable section, and card count is bounded by distinct
`(issue_type, mechanism_key)` pairs. Measured across corpora spanning two
orders of magnitude, the opening prompt held at roughly 3,000 tokens
throughout.

No chunking layer is needed: a 38k-call corpus opens with the same size prompt
as a 193-call one.

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
insight-agent run-analyst traces.jsonl --agent "My agent" -o out --dry-run
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
insight-agent run-analyst traces.jsonl --agent A -o out --model openai/gpt-5.2
insight-agent run-analyst traces.jsonl --agent A -o out --model bedrock/anthropic.claude-opus-5
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

`--digest` and `--cards` bypass IA2 and IA3 and use existing artifacts, so a
prompt change can be re-issued against a frozen evidence set:

```bash
insight-agent run-analyst traces.jsonl --agent A -o out \
  --digest out/ia2/digest.md --cards out/ia3/cards.json
```

`run-all` does this automatically, reusing the digest and cards it just
computed. `--agent` is only a label for the system under test and defaults to
the corpus filename.

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

**The shipped prompt does not forbid impact claims.** Earlier drafts carried the
spec's guardrails verbatim — impact is `not_established` and may not be
upgraded, cite at least three traces, do not restate an anomaly flag as a
defect. `analyst_v3` deliberately drops them in favour of a shorter prompt that
leans on the tool loop, and the tradeoff is visible in the output: on one real
corpus it authored two single-trace Insights and used impact language ("interrupts
workflows", "wastes turns") that the earlier prompts produced none of. In
exchange it named mechanisms neither evidence stream contains — that the failing
tool required the formula output to be passed as a parameter — because it went
and read the calls. Judge the descriptions on their evidence, not on their
confidence; nothing upstream measured consequences.

## Prompt

`src/insight_agent/insights_generation/prompts/analyst_v3.md`, packaged with the wheel. The spec
assumes a frozen prompt and output contract across matched comparisons, so the
version is recorded in every `run.json` and the assembled text is echoed into
every `prompt.md`.

To try your own, drop a `<name>.md` beside it and pass `--prompt-version <name>`.
Two requirements:

- It must contain the marker `{evidence}`, which is where the digest, the cards,
  and the withheld/uncovered counts are substituted. A template without it is an
  error rather than an append, so a typo cannot send a paid request whose
  evidence landed somewhere the prompt never refers to. `{agent}` is also
  substituted if present.
- Name `fetch_traces` in it to get the tool loop; omit it for a single-shot,
  evidence-only run.

If a prompt does not state the output contract itself — detected by looking for
`trace_ids` — the contract is appended to the user turn, so a rough prompt still
returns parseable JSON.
