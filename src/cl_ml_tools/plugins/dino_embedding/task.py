"""DINOv2 embedding task implementation."""

import logging
from typing import Callable, override

import numpy as np

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.dino_embedder import DinoEmbedder
from .schema import DinoEmbeddingParams

logger = logging.getLogger(__name__)


class DinoEmbeddingTask(ComputeModule[DinoEmbeddingParams]):
    """Compute module for generating DINOv2 embeddings using ONNX model."""

    def __init__(self) -> None:
        super().__init__()
        self._embedder: DinoEmbedder | None = None

    @property
    @override
    def task_type(self) -> str:
        return "dino_embedding"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return DinoEmbeddingParams

    def _get_embedder(self) -> DinoEmbedder:
        if self._embedder is None:
            self._embedder = DinoEmbedder()
            logger.info("DINOv2 embedder initialized successfully")
        return self._embedder

    @override
    async def execute(
        self,
        job: Job,
        params: DinoEmbeddingParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Generate DINOv2 embedding and store it on disk."""

        # Phase 1: get embedder
        try:
            embedder = self._get_embedder()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize DINOv2 embedder", exc_info=exc)
            return TaskResult(
                status="error",
                error=(
                    f"Failed to initialize DINOv2 embedder: {exc}. "
                    "Ensure ONNX Runtime is installed and the model can be downloaded."
                ),
            )

        # Phase 2: generate + persist embedding
        try:
            embedding = embedder.embed(
                image_path=params.input_path,
                normalize=params.normalize,
            )

            np.save(params.output_path, embedding)

            if progress_callback:
                progress_callback(100)

            return TaskResult(
                status="ok",
                task_output={
                    "embedding_dim": int(embedding.shape[0]),
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
                "Failed to generate DINOv2 embedding for %s",
                params.input_path,
                exc_info=exc,
            )
            return TaskResult(
                status="error",
                error=str(exc),
            )
