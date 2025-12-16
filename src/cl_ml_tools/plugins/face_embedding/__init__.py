"""Face embedding plugin using ONNX models."""

from .schema import FaceEmbedding, FaceEmbeddingParams, FaceEmbeddingResult
from .task import FaceEmbeddingTask

__all__ = ["FaceEmbeddingTask", "FaceEmbeddingParams", "FaceEmbeddingResult", "FaceEmbedding"]
