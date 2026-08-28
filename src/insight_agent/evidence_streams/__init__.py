"""Built-in evidence streams and their shared handoff contracts."""

from insight_agent.evidence_streams.contracts import EvidenceStream, EvidenceStreamResult, Problem
from insight_agent.evidence_streams.registry import EvidenceStreamRegistry

__all__ = ["EvidenceStream", "EvidenceStreamRegistry", "EvidenceStreamResult", "Problem"]
