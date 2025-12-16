"""Face embedding parameters and output schemas."""

import numpy as np
from pydantic import BaseModel, Field

from ...common.schemas import BaseJobParams


class FaceEmbeddingParams(BaseJobParams):
    """Parameters for face embedding task.

    Attributes:
        input_paths: List of absolute paths to face images (cropped faces)
        output_paths: Not used for face embedding (embeddings returned in task_output)
        normalize: Whether to L2-normalize the embeddings (default: True)
    """

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

    def to_numpy(self) -> np.ndarray:
        """Convert embedding list to numpy array.

        Returns:
            Numpy array of embeddings
        """
        return np.array(self.embedding, dtype=np.float32)

    @classmethod
    def from_numpy(
        cls, embedding: np.ndarray, quality_score: float | None = None
    ) -> "FaceEmbedding":
        """Create FaceEmbedding from numpy array.

        Args:
            embedding: Numpy array of embeddings
            quality_score: Optional quality score

        Returns:
            FaceEmbedding instance
        """
        return cls(
            embedding=embedding.tolist(),
            embedding_dim=len(embedding),
            quality_score=quality_score,
        )


class FaceEmbeddingResult(BaseModel):
    """Embedding result for a single face image."""

    file_path: str = Field(..., description="Path to the input face image")
    embedding: FaceEmbedding | None = Field(
        default=None, description="Face embedding (None if extraction failed)"
    )
    status: str = Field(..., description="Status: 'success' or 'error'")
    error: str | None = Field(default=None, description="Error message if status is 'error'")
