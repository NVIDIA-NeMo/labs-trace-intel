# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit, in-process registration for evidence streams."""

from __future__ import annotations

from insight_agent.evidence_streams.evidence_streams import EvidenceStream, EvidenceStreamResult
from insight_agent.traces import TraceSnapshot


class EvidenceStreamRegistry:
    """Validated evidence streams keyed by their public names.

    Registration is explicit and preserves insertion order. A stream owns its
    configuration validation; the registry only enforces the shared contract.
    """

    def __init__(self) -> None:
        self._streams: dict[str, EvidenceStream] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._streams)

    def register(self, stream: EvidenceStream) -> None:
        name = stream.name.strip()
        if not name:
            raise ValueError("evidence stream name must not be empty")
        if name in self._streams:
            raise ValueError(f"evidence stream {name!r} is already registered")
        stream.validate_configuration()
        self._streams[name] = stream

    def analyze(self, name: str, snapshot: TraceSnapshot) -> EvidenceStreamResult:
        try:
            stream = self._streams[name]
        except KeyError as exc:
            available = ", ".join(self.names) or "none"
            raise KeyError(
                f"evidence stream {name!r} is not registered; available: {available}"
            ) from exc
        result = stream.analyze(snapshot)
        if result.stream_name != name:
            raise ValueError(f"evidence stream {name!r} returned result for {result.stream_name!r}")
        return result

    def analyze_all(self, snapshot: TraceSnapshot) -> tuple[EvidenceStreamResult, ...]:
        return tuple(self.analyze(name, snapshot) for name in self._streams)


__all__ = ["EvidenceStreamRegistry"]
