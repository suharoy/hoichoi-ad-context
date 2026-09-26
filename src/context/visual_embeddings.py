"""CPU CLIP vision embeddings; no scene classification or break scoring."""

import os

# Prevent Transformers from attempting background safetensors conversion.
# This must be set before importing transformers.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection


MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = torch.device("cpu")


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize one vector or a batch of vectors."""
    vectors = np.asarray(vectors, dtype=np.float32)

    if not np.isfinite(vectors).all():
        raise ValueError("Cannot normalize non-finite image vectors")

    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)

    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Cannot normalize zero or non-finite image vectors")

    return vectors / norms


def context_vector(vectors: np.ndarray) -> np.ndarray | None:
    """Mean a set of normalized shot vectors and L2-normalize the result."""
    vectors = np.asarray(vectors, dtype=np.float32)

    if len(vectors) == 0:
        return None

    return normalize_vectors(np.mean(vectors, axis=0))


class CLIPVisualEncoder:
    """CPU-only CLIP vision encoder producing normalized projected embeddings."""

    def __init__(self) -> None:
        # `backend="pil"` replaces the deprecated `use_fast=False`.
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            backend="pil",
        )

        # Force the ordinary PyTorch checkpoint path. This avoids the
        # safetensors auto-conversion path that previously stalled startup.
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            MODEL_NAME,
            use_safetensors=False,
        )

        self.model.to(DEVICE)
        self.model.eval()

        print(
            f"Visual encoder: {MODEL_NAME}, "
            f"{DEVICE.type.upper()}, projected normalized embeddings",
            flush=True,
        )

    def encode(
        self,
        paths: list[Path],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Encode image paths into normalized CLIP image embeddings."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if not paths:
            raise ValueError("Supply at least one image")

        batches: list[np.ndarray] = []

        with torch.inference_mode():
            for offset in range(0, len(paths), batch_size):
                batch_paths = paths[offset : offset + batch_size]
                images: list[Image.Image] = []

                try:
                    for path in batch_paths:
                        with Image.open(path) as source:
                            images.append(source.convert("RGB"))

                    inputs = self.processor(
                        images=images,
                        return_tensors="pt",
                    )

                    pixel_values = inputs["pixel_values"].to(DEVICE)

                    projected = self.model(
                        pixel_values=pixel_values
                    ).image_embeds

                    embeddings = projected.detach().cpu().numpy()
                    batches.append(normalize_vectors(embeddings))

                finally:
                    for image in images:
                        image.close()

        return np.concatenate(batches, axis=0)