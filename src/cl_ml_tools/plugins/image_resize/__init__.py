"""Image resize plugin."""

from .schema import ImageResizeParams
from .task import ImageResizeTask

__all__ = ["ImageResizeTask", "ImageResizeParams"]
