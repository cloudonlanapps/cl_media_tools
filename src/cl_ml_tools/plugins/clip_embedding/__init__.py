"""MobileCLIP embedding plugin for semantic image similarity using ONNX models."""

from .schema import ClipEmbeddingOutput, ClipEmbeddingParams
from .task import ClipEmbeddingTask

__all__ = ["ClipEmbeddingTask", "ClipEmbeddingParams", "ClipEmbeddingOutput"]
