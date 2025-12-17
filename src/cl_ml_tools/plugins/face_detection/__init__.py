"""Face detection plugin using ONNX models."""

from .schema import BoundingBox, FaceDetectionOutput, FaceDetectionParams
from .task import FaceDetectionTask

__all__ = ["FaceDetectionTask", "FaceDetectionParams", "FaceDetectionOutput", "BoundingBox"]
