"""Face embedding plugin using ONNX models."""

from .schema import FaceEmbeddingOutput, FaceEmbeddingParams
from .task import FaceEmbeddingTask

__all__ = ["FaceEmbeddingTask", "FaceEmbeddingParams", "FaceEmbeddingOutput"]
