# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complaint classification using the Transformers encoder and trained projection."""

from __future__ import annotations

from functools import cached_property
from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field

from insight_agent.evidence_streams.user_dissatisfaction.projection import ComplaintProjection

if TYPE_CHECKING:
    from transformers import (  # ty: ignore[unresolved-import] -- Optional encoder extra.
        BatchEncoding,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )


class ScreeningResult(BaseModel):
    """One user message's label, or why scoring was skipped; scores are uncalibrated."""

    label: Literal["complaint", "no_complaint", "no_user_messages"]
    token_count: int = 0
    scores: dict[str, float] = Field(default_factory=dict)


def validate_classifier_dependencies() -> None:
    if find_spec("torch") is None or find_spec("transformers") is None:
        raise ValueError(
            'user-dissatisfaction requires: uv pip install "insight-agent[dissatisfaction]" '
            "(from a source checkout: uv sync --extra dissatisfaction)"
        )


class ComplaintClassifier:
    """Classify one message, or expose its embedding for other callers.

    The encoder loads lazily once per instance. User text stays local. CUDA and
    MPS use bfloat16; CPU uses float32. No inference server or GPU kernel package
    is needed. Classification uses the first 20,000 characters; failures are unflagged.
    """

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self.projection = ComplaintProjection()

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        validate_classifier_dependencies()
        from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

        encoder = self.projection.metadata["encoder"]
        return AutoTokenizer.from_pretrained(
            encoder["model"], revision=encoder["revision"], padding_side="left"
        )

    @cached_property
    def model(self) -> PreTrainedModel:
        validate_classifier_dependencies()
        import torch  # ty: ignore[unresolved-import] -- Optional encoder extra.
        from transformers import AutoModel  # ty: ignore[unresolved-import]

        device = self.device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        dtype = torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16
        encoder = self.projection.metadata["encoder"]
        model = (
            AutoModel.from_pretrained(
                encoder["model"],
                revision=encoder["revision"],
                dtype=dtype,
                attn_implementation="sdpa",
                use_safetensors=True,
            )
            .to(device)
            .eval()
        )
        model.requires_grad_(False)
        model.config.use_cache = False
        return model

    def _tokens(self, text: str) -> BatchEncoding:
        return self.tokenizer(
            self.projection.metadata["encoder"]["prefix"] + text,
            truncation=False,
            return_tensors="pt",
            verbose=False,
        )

    def token_count(self, text: str) -> int:
        return self._tokens(text)["input_ids"].shape[-1]

    def embed(self, text: str) -> np.ndarray:
        """Return one float32 normalized vector, using the pinned input convention."""
        import torch  # ty: ignore[unresolved-import] -- Optional encoder extra.

        tokens = self._tokens(text)
        if tokens["input_ids"].shape[-1] > self.projection.metadata["max_tokens"]:
            raise ValueError("User message exceeds the 8192-token budget; input was not truncated")
        with torch.inference_mode():
            hidden = self.model(**tokens.to(self.model.device)).last_hidden_state
            return torch.nn.functional.normalize(hidden[:, -1].float(), dim=-1)[0].cpu().numpy()

    def score(self, text: str) -> float:
        return self.projection.score(self.projection.compress(self.embed(text)))

    def classify(self, text: str) -> ScreeningResult:
        text = text[:20000]
        if not text.strip():
            return ScreeningResult(label="no_user_messages")
        try:
            count = self.token_count(text)
            score = self.score(text)
            return ScreeningResult(
                label="complaint" if score >= self.projection.threshold else "no_complaint",
                token_count=count,
                scores={"complaint": score, "no_complaint": 1 - score},
            )
        except Exception:
            return ScreeningResult(label="no_complaint")


def load_classifier(device: str | None = None) -> ComplaintClassifier:
    validate_classifier_dependencies()
    return ComplaintClassifier(device)
