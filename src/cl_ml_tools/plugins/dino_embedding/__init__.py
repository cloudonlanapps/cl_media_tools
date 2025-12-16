"""DINOv2 embedding plugin for visual similarity using ONNX models."""

from .schema import DinoEmbedding, DinoEmbeddingParams, DinoEmbeddingResult
from .task import DinoEmbeddingTask

__all__ = ["DinoEmbeddingTask", "DinoEmbeddingParams", "DinoEmbeddingResult", "DinoEmbedding"]
