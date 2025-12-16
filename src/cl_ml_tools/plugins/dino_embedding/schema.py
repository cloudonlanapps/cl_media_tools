"""DINOv2 embedding parameters and output schemas."""

import numpy as np
from pydantic import BaseModel, Field

from ...common.schemas import BaseJobParams


class DinoEmbeddingParams(BaseJobParams):
    """Parameters for DINOv2 embedding task.

    Attributes:
        input_paths: List of absolute paths to input images
        output_paths: Not used for embedding (embeddings returned in task_output)
        normalize: Whether to L2-normalize the embeddings (default: True)
    """

    normalize: bool = Field(
        default=True,
        description="Whether to L2-normalize DINOv2 embeddings",
    )


class DinoEmbedding(BaseModel):
    """DINOv2 embedding result."""

    embedding: list[float] = Field(..., description="DINOv2 CLS token embedding (384D)")
    embedding_dim: int = Field(..., description="Embedding dimensionality (384)")

    def to_numpy(self) -> np.ndarray:
        """Convert embedding list to numpy array.

        Returns:
            Numpy array of embeddings
        """
        return np.array(self.embedding, dtype=np.float32)

    @classmethod
    def from_numpy(cls, embedding: np.ndarray) -> "DinoEmbedding":
        """Create DinoEmbedding from numpy array.

        Args:
            embedding: Numpy array of embeddings

        Returns:
            DinoEmbedding instance
        """
        return cls(
            embedding=embedding.tolist(),
            embedding_dim=len(embedding),
        )


class DinoEmbeddingResult(BaseModel):
    """Embedding result for a single image."""

    file_path: str = Field(..., description="Path to the input image")
    embedding: DinoEmbedding | None = Field(
        default=None, description="DINOv2 embedding (None if extraction failed)"
    )
    status: str = Field(..., description="Status: 'success' or 'error'")
    error: str | None = Field(default=None, description="Error message if status is 'error'")
