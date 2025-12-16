"""Face embedding parameters and output schemas."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ...common.schemas import BaseJobParams

FloatArray = NDArray[np.float32]


class FaceEmbeddingParams(BaseJobParams):
    """Parameters for face embedding task."""

    normalize: bool = Field(
        default=True,
        description="Whether to L2-normalize face embeddings",
    )


class FaceEmbedding(BaseModel):
    """Face embedding result."""

    embedding: list[float] = Field(..., description="Face embedding vector (128D or 512D)")
    embedding_dim: int = Field(..., description="Embedding dimensionality (128 or 512)")
    quality_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Quality score for the face (based on blur/sharpness)",
    )

    def to_numpy(self) -> FloatArray:
        return np.asarray(self.embedding, dtype=np.float32)

    @classmethod
    def from_numpy(
        cls,
        embedding: FloatArray,
        quality_score: float | None = None,
    ) -> FaceEmbedding:
        embedding_list: list[float] = [float(x) for x in embedding.reshape(-1)]
        return cls(
            embedding=embedding_list,
            embedding_dim=len(embedding_list),
            quality_score=quality_score,
        )


class FaceEmbeddingResult(BaseModel):
    """Embedding result for a single face image."""

    file_path: str = Field(..., description="Path to the input face image")
    embedding: FaceEmbedding | None = Field(
        default=None,
        description="Face embedding (None if extraction failed)",
    )
    status: str = Field(..., description="Status: 'success' or 'error'")
    error: str | None = Field(
        default=None,
        description="Error message if status is 'error'",
    )
