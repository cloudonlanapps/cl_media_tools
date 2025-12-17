"""DINOv2 embedding parameters and output schemas."""

from pydantic import Field

from ...common.schema_job import BaseJobParams, TaskOutput


class DinoEmbeddingParams(BaseJobParams):
    """Parameters for DINOv2 embedding task."""

    normalize: bool = Field(
        default=True,
        description="Whether to L2-normalize DINOv2 embeddings",
    )


class DinoEmbeddingOutput(TaskOutput):
    normalized: bool = Field(..., description="Whether the embedding is L2-normalized")
    embedding_dim: int = Field(..., description="Embedding dimensionality (384)")
    pass
