"""Face embedding parameters and output schemas."""

from pydantic import Field

from ...common.schema_job import BaseJobParams, TaskOutput


class FaceEmbeddingParams(BaseJobParams):
    """Parameters for face embedding task."""

    normalize: bool = Field(
        default=True,
        description="Whether to L2-normalize face embeddings",
    )


class EmbeddingOutput(TaskOutput):
    normalized: bool = Field(..., description="Whether the embedding is L2-normalized")
    embedding_dim: int = Field(..., description="Embedding dimensionality (128 or 512)")
    quality_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Quality score for the face (based on blur/sharpness)",
    )
    pass
