"""MobileCLIP embedding task implementation."""

import logging
from typing import Callable, override

import numpy as np

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.clip_embedder import ClipEmbedder
from .schema import ClipEmbeddingParams

logger = logging.getLogger(__name__)


class ClipEmbeddingTask(ComputeModule[ClipEmbeddingParams]):
    """Compute module for generating MobileCLIP embeddings using ONNX model."""

    def __init__(self) -> None:
        super().__init__()
        self._embedder: ClipEmbedder | None = None

    @property
    @override
    def task_type(self) -> str:
        return "clip_embedding"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return ClipEmbeddingParams

    def _get_embedder(self) -> ClipEmbedder:
        if self._embedder is None:
            try:
                self._embedder = ClipEmbedder()
                logger.info("MobileCLIP embedder initialized successfully")
            except Exception as exc:
                logger.error("Failed to initialize MobileCLIP embedder", exc_info=exc)
                raise
        return self._embedder

    @override
    async def execute(
        self,
        job: Job,
        params: ClipEmbeddingParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Generate MobileCLIP embedding and store it on disk."""
        try:
            try:
                embedder = self._get_embedder()
            except Exception as exc:
                return TaskResult(
                    status="error",
                    error=(
                        f"Failed to initialize MobileCLIP embedder: {exc}. "
                        "Ensure ONNX Runtime is installed and the model is available."
                    ),
                )

            try:
                embedding = embedder.embed(
                    image_path=params.input_path,
                    normalize=params.normalize,
                )

                # Persist embedding
                np.save(params.output_path, embedding)

                if progress_callback:
                    progress_callback(100)

                return TaskResult(
                    status="ok",
                    task_output={
                        "embedding_dim": int(embedding.shape[0]),
                        "normalize": params.normalize,
                    },
                )

            except FileNotFoundError:
                logger.error("File not found: %s", params.input_path)
                return TaskResult(
                    status="error",
                    error="Input file not found",
                )

            except Exception as exc:
                logger.error(
                    "Failed to generate MobileCLIP embedding for %s",
                    params.input_path,
                    exc_info=exc,
                )
                return TaskResult(
                    status="error",
                    error=str(exc),
                )

        except Exception as exc:
            logger.exception("Unexpected error in ClipEmbeddingTask")
            return TaskResult(
                status="error",
                error=f"Task failed: {exc}",
            )
