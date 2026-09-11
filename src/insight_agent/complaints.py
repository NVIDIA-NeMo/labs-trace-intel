# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable complaint classification: Transformers encoder and NumPy projection/head."""

from __future__ import annotations

import json
from functools import cached_property
from importlib.resources import files
from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field

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


class ComplaintProjection:
    """Use the bundled PCA, four-bit MSE quantizer and refitted logistic head.

    No Torch, Transformers, sklearn, pickle, or hosted inference is required.
    compress() accepts one normalized 4096-dimensional embedding and returns
    132 bytes. score() accepts those bytes and returns the complaint score.
    """

    def __init__(self) -> None:
        folder = files("insight_agent").joinpath("models")
        self.metadata = json.loads(folder.joinpath("complaint.json").read_text())
        with folder.joinpath("complaint.npz").open("rb") as handle:
            with np.load(handle, allow_pickle=False) as arrays:
                self.arrays = {key: arrays[key] for key in arrays.files}
        self.threshold = self.metadata["threshold"]

    def compress(self, embedding: np.ndarray) -> bytes:
        x = np.asarray(embedding, dtype="float32")
        if x.shape != (4096,) or not np.isfinite(x).all():
            raise ValueError("Expected one finite 4096-dimensional Qwen embedding")
        if not np.isclose(np.linalg.norm(x), 1, atol=1e-4):
            raise ValueError("Qwen embedding must be L2-normalized")
        a = self.arrays
        # Same projection order as the saved sklearn PCA transform.
        z = x @ a["pca_components"].T - a["pca_mean"] @ a["pca_components"].T
        norm = np.float32(np.linalg.norm(z))
        rotated = (z / norm if norm else z) @ a["rotation"]
        centers = a["centers"]
        codes = np.searchsorted((centers[:-1] + centers[1:]) / 2, rotated).astype("uint8")
        packed = codes[::2] | (codes[1::2] << 4)
        return packed.tobytes() + np.asarray(norm, dtype="<f4").tobytes()

    def decompress(self, packed: bytes) -> np.ndarray:
        """Reconstruct the 256 PCA features for a compatible downstream head."""
        if len(packed) != 132:
            raise ValueError("Expected one 132-byte four-bit MSE vector")
        data = np.frombuffer(packed, dtype="uint8", count=128)
        codes = np.empty(256, dtype="uint8")
        codes[::2], codes[1::2] = data & 15, data >> 4
        norm = np.frombuffer(packed, dtype="<f4", count=1, offset=128)[0]
        if not np.isfinite(norm) or norm < 0:
            raise ValueError("Quantized vector norm must be finite and nonnegative")
        a = self.arrays
        return (a["centers"][codes] @ a["rotation"].T) * norm

    def score(self, packed: bytes) -> float:
        a = self.arrays
        logit = self.decompress(packed) @ a["weight"] + a["bias"]
        return float(np.exp(-np.logaddexp(0, -logit)))


def validate_classifier_dependencies() -> None:
    if find_spec("torch") is None or find_spec("transformers") is None:
        raise ValueError("user-dissatisfaction requires: uv sync --extra dissatisfaction")


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
