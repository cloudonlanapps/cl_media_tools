"""Image conversion parameters schema."""

from typing import Literal

from pydantic import Field

from ...common.schema_job import BaseJobParams, TaskOutput


class ImageConversionParams(BaseJobParams):
    """Parameters for image conversion task."""

    format: Literal["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"] = Field(
        ..., description="Target image format"
    )
    quality: int = Field(
        default=85, ge=1, le=100, description="Output quality for lossy formats (1-100)"
    )


class EmbeddingOutput(TaskOutput):
    pass
