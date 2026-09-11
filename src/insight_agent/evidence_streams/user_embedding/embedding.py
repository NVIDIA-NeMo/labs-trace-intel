# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable user-text encoding, PCA projection, and four-bit compression."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from importlib.resources import files
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from transformers import (  # ty: ignore[unresolved-import] -- Optional encoder extra.
        BatchEncoding,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )


@dataclass(frozen=True)
class UserEmbedding:
    """Compressed vector and token count for the recorded encoder convention."""

    data: bytes
    token_count: int


class UserEmbeddingProjection:
    """Project and compress normalized user embeddings into reusable 132-byte vectors."""

    def __init__(self) -> None:
        folder = files("insight_agent.evidence_streams.user_embedding").joinpath("models")
        self.metadata = json.loads(folder.joinpath("embedding.json").read_text())
        with folder.joinpath("embedding.npz").open("rb") as handle:
            with np.load(handle, allow_pickle=False) as arrays:
                self.arrays = {key: arrays[key] for key in arrays.files}

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


def validate_embedding_dependencies() -> None:
    if find_spec("torch") is None or find_spec("transformers") is None:
        raise ValueError(
            'User embeddings require: uv pip install "insight-agent[dissatisfaction]" '
            "(from a source checkout: uv sync --extra dissatisfaction)"
        )


class UserEmbeddingGenerator:
    """Encode user text locally; the pinned instruction remains complaint-oriented."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self.projection = UserEmbeddingProjection()

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        validate_embedding_dependencies()
        from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

        encoder = self.projection.metadata["encoder"]
        tokenizer = AutoTokenizer.from_pretrained(
            encoder["model"], revision=encoder["revision"], padding_side="left"
        )
        if tokenizer is None:
            raise ValueError("The encoder checkpoint did not provide a tokenizer")
        return tokenizer

    @cached_property
    def model(self) -> PreTrainedModel:
        validate_embedding_dependencies()
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

    def generate(self, text: str) -> UserEmbedding:
        """Encode the first 20,000 characters; propagate failures to the caller."""
        text = text[:20000]
        return UserEmbedding(
            data=self.projection.compress(self.embed(text)),
            token_count=self.token_count(text),
        )
