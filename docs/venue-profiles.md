# Venue profiles

A "venue" is one source of traces. The research code was measured against a
handful of internal venues, and a few of their local conventions ended up as
literals inside the algorithms:

- the tool that ran code was called `CodeExecutionTool`
- one tool answered with `[NO_ACTIVE_SESSION]`
- the trajectory vocabulary was `planning` / `agent` / `evaluation`

None of that generalises. But all of it is *measured* behaviour that must not
change by accident, so every such literal moved into `VenueProfile` with its
original value as the default. `VenueProfile()` reproduces the original
behaviour exactly; `tests/test_venue_profile.py` asserts the IA2 digest and IA3
findings are byte-identical to golden files captured before the refactor.

## Settings

| Field | Default | Effect |
|---|---|---|
| `code_execution_tools` | `{"CodeExecutionTool"}` | Which tools may have their output decoded as a Python traceback, and which count toward the `code_execution_share` feature |
| `agent_step_types` | `{"agent", "agent_step", "planning"}` | Which step types carry agent reasoning |
| `evaluation_step_type` | `"evaluation"` | The terminal step type; collapsed to a single `evaluation:boundary` token during clustering so per-run verdict text does not fragment clusters |
| `returned_data_key` | `"returned_data"` | The key inside a result whose `false` value means "succeeded but returned nothing" |
| `state_patterns` | 4 regexes | `(name, regex)` pairs detecting a tool refusing to act because a precondition is unmet |
| `name`, `notes` | `"default"`, `{}` | Recorded in `run.json` for provenance |

## Why the tool-name gate exists

It looks odd to gate traceback detection on a tool name. The reason is
precision: an ordinary search tool that happens to return a document
*containing* a Python traceback should not be scored as a code-execution
failure. The gate is what stops that. Renaming it is fine; removing it would
change what the detector means.

## Using one

```bash
insight-agent sample --copy ./sample          # includes venue_profile_example.json
insight-agent run-ia3 traces.jsonl --profile ./sample/venue_profile_example.json -o out
```

```json
{
  "name": "my-venue",
  "code_execution_tools": ["PythonSandbox", "ShellTool"],
  "agent_step_types": ["think", "plan"],
  "evaluation_step_type": "verdict",
  "returned_data_key": "had_rows",
  "state_patterns": [
    ["needs_login", "(?i)please log in"],
    ["needs_workspace", "(?i)open a workspace"]
  ]
}
```

Unknown fields are rejected, and an invalid regex fails at load rather than
partway through a run.

In Python:

```python
from insight_agent.venue import DEFAULT_PROFILE, load_profile

profile = load_profile("my_venue.json")
profile = DEFAULT_PROFILE.with_overrides(code_execution_tools=frozenset({"PythonSandbox"}))

run_ia2(traces, profile=profile)
detect(records, profile=profile)
```

## Effect on the sample corpus

Running the bundled corpus with `venue_profile_example.json`, which renames the
code-execution tool and replaces the state patterns:

| | Default | Renamed profile |
|---|---:|---:|
| `python_traceback` findings | 1 | 0 |
| `explicit_prerequisite_or_state_failure` findings | 7 | 3 |

The traceback finding disappears because `CodeExecutionTool` is no longer a
code-execution tool under that profile — which is exactly the point. If your
venue calls it something else, you would otherwise be silently missing every
code-execution failure in your corpus.

## What is deliberately *not* configurable

- `contamination`, `n_estimators` and the random seeds — these are run-time
  parameters on the functions themselves, not venue identity.
- The nineteen IA3 rules. Adding or removing rules changes what the detector
  version means; `detector_version` is stamped on every finding for that reason.
- The eleven IA2 features. You can *add* features through a trace's `metrics`
  object, but the built-in eleven are fixed.
