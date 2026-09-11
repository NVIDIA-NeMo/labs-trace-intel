# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NumPy projection, quantization, and complaint scoring."""

import json
from importlib.resources import files

import numpy as np


class ComplaintProjection:
    """Use the bundled PCA, four-bit MSE quantizer and refitted logistic head."""

    def __init__(self) -> None:
        folder = files("insight_agent.evidence_streams.user_dissatisfaction").joinpath("models")
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
