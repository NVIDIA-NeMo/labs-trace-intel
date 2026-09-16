# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit, in-process registration for evidence streams."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from insight_agent.evidence_streams.evidence_streams import EvidenceStream, EvidenceStreamResult
from insight_agent.traces import TraceSnapshot


class EvidenceStreamRegistry:
    """Evidence streams keyed by their public names.

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
        self._streams[name] = stream

    async def analyze(
        self, name: str, snapshot: TraceSnapshot, *, on_start: Callable[[], None] | None = None
    ) -> EvidenceStreamResult:
        try:
            stream = self._streams[name]
        except KeyError as exc:
            available = ", ".join(self.names) or "none"
            raise KeyError(
                f"evidence stream {name!r} is not registered; available: {available}"
            ) from exc
        skip_reason = await asyncio.to_thread(stream.check_prerequisites, snapshot)
        if not len(snapshot):
            skip_reason = "No traces loaded"
        if skip_reason is not None:
            return EvidenceStreamResult(stream_name=name, problems=(), skip_reason=skip_reason)
        if on_start is not None:
            on_start()
        result = await stream.analyze(snapshot)
        if result.stream_name != name:
            raise ValueError(f"evidence stream {name!r} returned result for {result.stream_name!r}")
        return result

    async def analyze_all(self, snapshot: TraceSnapshot) -> tuple[EvidenceStreamResult, ...]:
        return tuple(
            await asyncio.gather(*(self.analyze(name, snapshot) for name in self._streams))
        )


__all__ = ["EvidenceStreamRegistry"]
