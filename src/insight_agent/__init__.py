"""Insight Agent: trace evidence and Insight generation."""

from importlib.metadata import version

from insight_agent.insight import Insight

__version__ = version("insight-agent")

__all__ = ["Insight", "__version__"]
