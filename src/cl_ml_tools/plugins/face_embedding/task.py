"""Face embedding task implementation."""

import logging
from typing import Callable, override

import numpy as np

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.face_embedder import FaceEmbedder
from .schema import FaceEmbeddingParams

logger = logging.getLogger(__name__)


class FaceEmbeddingTask(ComputeModule[FaceEmbeddingParams]):
    """Compute module for generating a face embedding using ONNX model."""

    def __init__(self) -> None:
        super().__init__()
        self._embedder: FaceEmbedder | None = None

    @property
    @override
    def task_type(self) -> str:
        return "face_embedding"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return FaceEmbeddingParams

    def _get_embedder(self) -> FaceEmbedder:
        if self._embedder is None:
            self._embedder = FaceEmbedder()
            logger.info("Face embedder initialized successfully")
        return self._embedder

    @override
    async def execute(
        self,
        job: Job,
        params: FaceEmbeddingParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Generate a face embedding and store it on disk."""

        # Phase 1: get embedder
        try:
            embedder = self._get_embedder()
        except Exception as exc:  # noqa: BLE001
            logger.error("Face embedder initialization failed", exc_info=exc)
            return TaskResult(
                status="error",
                error=(
                    f"Failed to initialize face embedder: {exc}. "
                    "Ensure ONNX Runtime is installed and the model can be downloaded."
                ),
            )

        # Phase 2: generate + persist embedding
        try:
            embedding_array, quality_score = embedder.embed(
                image_path=params.input_path,
                normalize=params.normalize,
                compute_quality=True,
            )

            # Persist embedding array
            np.save(params.output_path, embedding_array)

            if progress_callback:
                progress_callback(100)

            return TaskResult(
                status="ok",
                task_output={
                    "embedding_dim": int(embedding_array.shape[0]),
                    "quality_score": quality_score,
                },
            )

        except FileNotFoundError:
            logger.error("File not found: %s", params.input_path)
            return TaskResult(
                status="error",
                error="Input file not found",
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to generate face embedding for %s",
                params.input_path,
                exc_info=exc,
            )
            return TaskResult(
                status="error",
                error=str(exc),
            )
