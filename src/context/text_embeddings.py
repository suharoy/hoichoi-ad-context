"""CPU text embeddings for symmetric Bengali semantic comparisons."""

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"


class SemanticTextEncoder:
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        print(f"Text encoder initialized: model={MODEL_NAME}, device=cpu", flush=True)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if any(not text.strip() for text in texts):
            raise ValueError("Empty transcripts have no semantic evidence")
        return self.model.encode(
            ["query: " + text for text in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
