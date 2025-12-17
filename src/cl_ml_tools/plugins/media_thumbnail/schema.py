"""Media thumbnail parameters schema."""

from ...common.schema_job import BaseJobParams, TaskOutput


class MediaThumbnailParams(BaseJobParams):
    """Parameters for image/video thumbnail task.

    Supports both image and video resizing.
    Media type is auto-detected from file content.

    """

    width: int | None = None
    height: int | None = None
    maintain_aspect_ratio: bool = True


class MediaThumbnailOutput(TaskOutput):
    media_type: str | None = None
    pass
