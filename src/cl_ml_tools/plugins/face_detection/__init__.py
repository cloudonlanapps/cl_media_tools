"""Face detection plugin using ONNX models."""

from .schema import BoundingBox, FaceDetectionParams, FaceDetectionResult
from .task import FaceDetectionTask

__all__ = ["FaceDetectionTask", "FaceDetectionParams", "FaceDetectionResult", "BoundingBox"]
