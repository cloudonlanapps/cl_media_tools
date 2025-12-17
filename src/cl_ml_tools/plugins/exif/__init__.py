"""EXIF metadata extraction plugin."""

from .schema import ExifMetadataOutput, ExifMetadataParams
from .task import ExifTask

__all__ = ["ExifTask", "ExifMetadataParams", "ExifMetadataOutput"]
