"""DINOv2 embedding parameters and output schemas."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ...common.schemas import BaseJobParams


class DinoEmbeddingParams(BaseJobParams):
    """Parameters for DINOv2 embedding task."""

    normalize: bool = Field(
        default=True,
        description="Whether to L2-normalize DINOv2 embeddings",
    )


class DinoEmbedding(BaseModel):
    """DINOv2 embedding result."""

    embedding: list[float] = Field(..., description="DINOv2 CLS token embedding (384D)")
    embedding_dim: int = Field(..., description="Embedding dimensionality (384)")

    def to_numpy(self) -> NDArray[np.float32]:
        """Convert embedding list to numpy array."""
        return np.asarray(self.embedding, dtype=np.float32)

    @classmethod
    def from_numpy(cls, embedding: NDArray[np.float32]) -> "DinoEmbedding":
        """Create DinoEmbedding from numpy array."""
        flat: NDArray[np.float32] = embedding.reshape(-1).astype(np.float32)
        values: list[float] = list(map(float, flat))
        return cls(
            embedding=values,
            embedding_dim=flat.shape[0],
        )


class DinoEmbeddingResult(BaseModel):
    """Embedding result for a single image."""

    file_path: str = Field(..., description="Path to the input image")
    embedding: DinoEmbedding | None = Field(
        default=None, description="DINOv2 embedding (None if extraction failed)"
    )
    status: str = Field(..., description="Status: 'success' or 'error'")
    error: str | None = Field(default=None, description="Error message if status is 'error'")
