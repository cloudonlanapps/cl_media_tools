"""MobileCLIP embedding plugin for semantic image similarity using ONNX models."""

from .schema import ClipEmbedding, ClipEmbeddingParams, ClipEmbeddingResult
from .task import ClipEmbeddingTask

__all__ = ["ClipEmbeddingTask", "ClipEmbeddingParams", "ClipEmbeddingResult", "ClipEmbedding"]
