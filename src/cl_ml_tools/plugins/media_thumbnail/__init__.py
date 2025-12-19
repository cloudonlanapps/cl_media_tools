"""Media thumbnail plugin."""

from .schema import MediaThumbnailOutput, MediaThumbnailParams
from .task import MediaThumbnailTask

__all__ = ["MediaThumbnailTask", "MediaThumbnailParams", "MediaThumbnailOutput"]
