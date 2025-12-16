"""MobileCLIP embedding schemas."""

from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ...common.schemas import BaseJobParams


class ClipEmbeddingParams(BaseJobParams):
    """Parameters for MobileCLIP image embedding."""

    normalize: bool = Field(
        default=True,
        description="Whether to L2-normalize the embedding vector (recommended for similarity search)",
    )


class ClipEmbedding(BaseModel):
    """MobileCLIP image embedding representation."""

    embedding: list[float] = Field(
        ...,
        description="MobileCLIP image embedding vector (512D for MobileCLIP-S2)",
    )
    embedding_dim: int = Field(..., description="Embedding dimensionality (512)")
    normalized: bool = Field(..., description="Whether the embedding is L2-normalized")

    def to_numpy(self) -> NDArray[np.float32]:
        """Convert embedding to numpy array.

        Returns:
            1D numpy array of float32
        """
        return np.asarray(self.embedding, dtype=np.float32)

    @classmethod
    def from_numpy(
        cls,
        embedding: NDArray[np.floating],
        normalized: bool = True,
    ) -> "ClipEmbedding":
        """Create ClipEmbedding from numpy array.

        Args:
            embedding: 1D numpy array
            normalized: Whether the embedding is L2-normalized

        Returns:
            ClipEmbedding instance
        """
        if embedding.ndim != 1:
            raise ValueError(f"Expected 1D embedding, got shape {embedding.shape}")

        return cls(
            embedding=cast(list[float], embedding.tolist()),
            embedding_dim=embedding.shape[0],
            normalized=normalized,
        )


class ClipEmbeddingResult(BaseModel):
    """Result for a single image's MobileCLIP embedding."""

    file_path: str = Field(..., description="Path to the input image file")
    embedding: ClipEmbedding | None = Field(
        None, description="MobileCLIP embedding (None if error)"
    )
    status: Literal["success", "error"] = Field(..., description="Processing status")
    error: str | None = Field(None, description="Error message if status is 'error'")
