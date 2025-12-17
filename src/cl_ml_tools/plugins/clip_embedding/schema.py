"""MobileCLIP embedding schemas."""

from pydantic import Field

from ...common.schema_job import BaseJobParams, TaskOutput


class ClipEmbeddingParams(BaseJobParams):
    """Parameters for MobileCLIP image embedding."""

    normalize: bool = Field(
        default=True,
        description="Whether to L2-normalize the embedding vector (recommended for similarity search)",
    )


class ClipEmbeddingOutput(TaskOutput):
    embedding_dim: int = Field(..., description="Embedding dimensionality (512)")
    normalized: bool = Field(..., description="Whether the embedding is L2-normalized")
