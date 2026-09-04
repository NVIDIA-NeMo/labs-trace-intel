"""Shared test configuration.

``tests/`` is on ``sys.path`` so helper modules such as ``baseline_corpus`` can
be imported directly by test modules without a package indirection.
"""

from __future__ import annotations

import pathlib
import sys

import litellm
import pytest

TESTS_DIR = pathlib.Path(__file__).parent
REPO_ROOT = TESTS_DIR.parent
DATA_DIR = TESTS_DIR / "data"

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

CREDENTIAL_VARS = (
    "INFERENCE_API_KEY",
    "INFERENCE_API_BASE",
    "INFERENCE_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def no_real_model_calls(monkeypatch, tmp_path):
    """Make a real API call impossible, and a developer's `.env` invisible.

    Insight compilation runs by default in `run-all` and `demo`, so a test that forgets
    to pass ``--no-insights`` would otherwise reach
    the network — spending money and coupling the suite to a live endpoint. It
    fails loudly here instead.

    ``fake_credentials`` supplies the key that lets the preflight pass.
    """
    for var in CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)
    # `find_dotenv` walks up from the cwd to the repo root, so running from
    # anywhere inside the checkout would pick up a developer's real `.env`.
    # A tmp cwd is outside that walk entirely.
    monkeypatch.chdir(tmp_path)

    def _blocked(**kwargs):
        raise AssertionError(
            "a test tried to reach a real model endpoint. Pass --no-insights if the "
            "test is about the deterministic path."
        )

    monkeypatch.setattr(litellm, "completion", _blocked, raising=False)


@pytest.fixture
def fake_credentials(monkeypatch):
    """Satisfy Insight compilation preflight without a real key."""
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key-not-real")
