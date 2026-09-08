"""`.env` loading and credential resolution.

Only the Analyst reads these, so the tests care about two things: that a real
exported variable is never silently overwritten by a stale file, and that an
unfilled template line is treated as absent rather than as an empty value.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from insight_agent.config import load_run_config
from insight_agent.insights_generation.config import (
    ENV_API_BASE,
    ENV_API_KEY,
    ENV_MODEL,
    find_dotenv,
    load_dotenv,
    resolve,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        ENV_MODEL,
        ENV_API_BASE,
        ENV_API_KEY,
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def write_env(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


# -- parsing ---------------------------------------------------------------


def test_loads_simple_assignments(tmp_path):
    env = write_env(tmp_path / ".env", f"{ENV_MODEL}=openai/openai/gpt-5.2\n{ENV_API_KEY}=secret\n")
    assert load_dotenv(env) == {ENV_MODEL: "openai/openai/gpt-5.2", ENV_API_KEY: "secret"}
    assert os.environ[ENV_MODEL] == "openai/openai/gpt-5.2"


def test_ignores_comments_and_blank_lines(tmp_path):
    env = write_env(tmp_path / ".env", f"# a comment\n\n  \n{ENV_API_KEY}=k\n")
    assert load_dotenv(env) == {ENV_API_KEY: "k"}


def test_handles_export_prefix_and_quotes(tmp_path):
    env = write_env(
        tmp_path / ".env",
        f'export {ENV_API_BASE}="https://gateway.example.com/v1"\n'
        f"{ENV_MODEL}='openai/openai/gpt-5.2'\n",
    )
    loaded = load_dotenv(env)
    assert loaded[ENV_API_BASE] == "https://gateway.example.com/v1"
    assert loaded[ENV_MODEL] == "openai/openai/gpt-5.2"


def test_a_value_containing_equals_survives(tmp_path):
    env = write_env(tmp_path / ".env", f"{ENV_API_KEY}=abc=def==\n")
    assert load_dotenv(env)[ENV_API_KEY] == "abc=def=="


def test_unfilled_template_lines_are_skipped(tmp_path, monkeypatch):
    """An empty `KEY=` must not mask a real exported credential."""
    monkeypatch.setenv(ENV_API_KEY, "from-the-environment")
    env = write_env(tmp_path / ".env", f"{ENV_API_KEY}=\n")

    assert load_dotenv(env) == {}
    assert os.environ[ENV_API_KEY] == "from-the-environment"


def test_the_real_environment_wins_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_MODEL, "exported")
    env = write_env(tmp_path / ".env", f"{ENV_MODEL}=from-file\n")

    assert load_dotenv(env) == {}
    assert os.environ[ENV_MODEL] == "exported"


def test_override_forces_the_file_to_win(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_MODEL, "exported")
    env = write_env(tmp_path / ".env", f"{ENV_MODEL}=from-file\n")

    assert load_dotenv(env, override=True) == {ENV_MODEL: "from-file"}
    assert os.environ[ENV_MODEL] == "from-file"


def test_a_missing_file_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == {}


# -- discovery -------------------------------------------------------------


def test_find_dotenv_prefers_a_file_beside_the_given_path(tmp_path):
    beside = write_env(tmp_path / ".env", "X=1\n")
    corpus = tmp_path / "traces.jsonl"
    corpus.write_text("", encoding="utf-8")
    assert find_dotenv(corpus) == beside


def test_find_dotenv_walks_up_to_the_repo_root(tmp_path, monkeypatch):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    root_env = write_env(tmp_path / ".env", "X=1\n")

    monkeypatch.chdir(nested)
    assert find_dotenv() == root_env


# -- resolution ------------------------------------------------------------


def test_resolve_prefers_the_repo_specific_name(monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "mine")
    monkeypatch.setenv("OPENAI_API_KEY", "theirs")
    assert resolve(ENV_API_KEY) == "mine"


@pytest.mark.parametrize("fallback", ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
def test_resolve_falls_back_to_provider_variables(monkeypatch, fallback):
    """An environment already configured for litellm needs no `.env`."""
    monkeypatch.setenv(fallback, "inherited")
    assert resolve(ENV_API_KEY) == "inherited"


def test_resolve_returns_the_default_when_nothing_is_set():
    assert resolve(ENV_MODEL, "fallback-model") == "fallback-model"
    assert resolve(ENV_MODEL) is None


# -- the shipped template --------------------------------------------------


def test_env_example_is_committed_and_parses():
    example = REPO_ROOT / ".env.example"
    assert example.is_file(), ".env.example is the committed template; it must exist"

    body = example.read_text(encoding="utf-8")
    for name in (ENV_API_BASE, ENV_MODEL, ENV_API_KEY):
        assert f"{name}=" in body, name


def test_env_example_ships_no_credential():
    """A template with a real key in it is the classic way to leak one."""
    for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if "KEY" in key or "TOKEN" in key or "SECRET" in key:
            assert value == "", f"{key} must ship empty, found {value!r}"


def test_dotenv_is_gitignored():
    """It holds a credential; a missing ignore rule is a real hazard."""
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert ".env" in ignore


def test_run_config_resolves_paths_relative_to_its_yaml_file(tmp_path):
    config_file = tmp_path / "settings" / "analyst.yaml"
    config_file.parent.mkdir()
    config_file.write_text(
        """trace:
  filesystem:
    path: corpus.jsonl
output:
  directory: results
evidence_streams:
  anomaly_and_patterns: {}
analyst:
  env_file: .env
""",
        encoding="utf-8",
    )

    config = load_run_config(config_file)
    assert config.trace.filesystem is not None
    assert config.trace.filesystem.path == config_file.parent / "corpus.jsonl"
    assert config.output.directory == config_file.parent / "results"
    assert config.analyst.env_file == config_file.parent / ".env"


def test_run_config_supports_mlflow_sources(tmp_path):
    config_file = tmp_path / "analyst.yaml"
    config_file.write_text(
        """trace:
  max_traces: 25
  mlflow_export:
    path: exports/traces.json
evidence_streams:
  anomaly_and_patterns: {}
""",
        encoding="utf-8",
    )

    config = load_run_config(config_file)
    assert config.trace.mlflow_export is not None
    assert config.trace.mlflow_export.path == tmp_path / "exports" / "traces.json"
    assert config.trace.max_traces == 25


def test_run_config_requires_exactly_one_trace_loader(tmp_path):
    config_file = tmp_path / "analyst.yaml"
    config_file.write_text(
        """trace:
  filesystem:
    path: traces.jsonl
  mlflow_export:
    path: traces.json
evidence_streams:
  anomaly_and_patterns: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one loader"):
        load_run_config(config_file)


def test_run_config_rejects_unsupported_filesystem_trace_limit(tmp_path):
    config_file = tmp_path / "analyst.yaml"
    config_file.write_text(
        """trace:
  max_traces: 25
  filesystem:
    path: traces.jsonl
evidence_streams:
  anomaly_and_patterns: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not supported by the filesystem loader"):
        load_run_config(config_file)


def test_run_config_requires_a_configured_evidence_stream(tmp_path):
    config_file = tmp_path / "analyst.yaml"
    config_file.write_text(
        """trace:
  filesystem:
    path: traces.jsonl
evidence_streams:
  {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one evidence stream"):
        load_run_config(config_file)
