"""CPU CLIP text embeddings compatible with cached Stage-3 CLIP image embeddings."""

from __future__ import annotations

import os
from collections.abc import Sequence

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

import numpy as np
import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection


MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = torch.device("cpu")


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)

    if not np.isfinite(vectors).all():
        raise ValueError("Cannot normalize non-finite text vectors")

    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)

    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Cannot normalize zero or non-finite text vectors")

    return vectors / norms


class CLIPTextEncoder:
    """CLIP text projection in the same 512-D space as Stage-3 image embeddings."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = CLIPTextModelWithProjection.from_pretrained(
            MODEL_NAME,
            use_safetensors=False,
        )

        self.model.to(DEVICE)
        self.model.eval()

        print(
            f"CLIP text encoder: {MODEL_NAME}, CPU, normalized projected embeddings",
            flush=True,
        )

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        cleaned = [text.strip() for text in texts]

        if not cleaned or any(not text for text in cleaned):
            raise ValueError("CLIP text inputs must be non-empty")

        batches: list[np.ndarray] = []

        with torch.inference_mode():
            for offset in range(0, len(cleaned), batch_size):
                batch = cleaned[offset : offset + batch_size]

                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                )

                tokens = {
                    key: value.to(DEVICE)
                    for key, value in tokens.items()
                }

                projected = self.model(**tokens).text_embeds

                batches.append(
                    normalize_vectors(
                        projected.detach().cpu().numpy()
                    )
                )

        return np.concatenate(batches, axis=0)
