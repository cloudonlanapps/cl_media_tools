"""DINOv2 embedding plugin for visual similarity using ONNX models."""

from .schema import DinoEmbeddingOutput, DinoEmbeddingParams
from .task import DinoEmbeddingTask

__all__ = ["DinoEmbeddingTask", "DinoEmbeddingParams", "DinoEmbeddingOutput"]
