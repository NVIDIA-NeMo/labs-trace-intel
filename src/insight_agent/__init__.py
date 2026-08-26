"""Insight Agent: deterministic trace-evidence preprocessing (IA2, IA3).

Kept deliberately import-light. The heavy engines pull in numpy/scikit-learn,
which costs about a second, and commands like ``insight-agent validate`` have no
reason to pay for it. Submodules are resolved lazily on first attribute access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

CANONICAL_SCHEMA_VERSION = "insight-trace/v1"

_LAZY = {
    "ia2_pipeline",
    "ia3_tid",
    "loader",
    "validate",
    "venue",
    "coverage",
    "serialize",
    "streams",
    "traces",
}

__all__ = ["__version__", "CANONICAL_SCHEMA_VERSION", *sorted(_LAZY)]

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from . import (
        coverage,
        ia2_pipeline,
        ia3_tid,
        loader,
        serialize,
        streams,
        traces,
        validate,
        venue,
    )


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY)
