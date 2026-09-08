"""LLM-backed Insight compilation."""

from insight_agent.insight import Insight
from insight_agent.insights_generation.insight_compilation import InsightCompilation

DEFAULT_MODEL = "openai/azure/openai/gpt-5.6-luna"
DEFAULT_MAX_TOKENS = 300_000
DEFAULT_MAX_TOOL_ROUNDS = 50

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_TOOL_ROUNDS",
    "DEFAULT_MODEL",
    "Insight",
    "InsightCompilation",
]
