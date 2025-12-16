"""Face embedding task implementation."""

import logging
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.face_embedder import FaceEmbedder
from .schema import FaceEmbedding, FaceEmbeddingParams, FaceEmbeddingResult

logger = logging.getLogger(__name__)


class FaceEmbeddingTask(ComputeModule[FaceEmbeddingParams]):
    """Compute module for generating face embeddings using ONNX model."""

    def __init__(self):
        """Initialize face embedding task."""
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
        """Get or create face embedder instance (lazy loading)."""
        if self._embedder is None:
            try:
                self._embedder = FaceEmbedder()
                logger.info("Face embedder initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize face embedder: {e}")
                raise

        return self._embedder

    @override
    async def execute(
        self,
        job: Job,
        params: FaceEmbeddingParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Generate embeddings for input face images.

        Args:
            job: Job instance
            params: FaceEmbeddingParams with input_paths and normalization settings
            progress_callback: Optional callback for progress updates (0-100)

        Returns:
            TaskResult with status and embeddings for each face image
        """
        try:
            # Initialize embedder
            try:
                embedder = self._get_embedder()
            except Exception as e:
                logger.error(f"Face embedder initialization failed: {e}")
                return {
                    "status": "error",
                    "error": (
                        f"Failed to initialize face embedder: {e}. "
                        "Ensure ONNX Runtime is installed and the model can be downloaded."
                    ),
                }

            file_results: list[dict] = []
            total_files = len(params.input_paths)

            for index, input_path in enumerate(params.input_paths):
                try:
                    # Generate embedding
                    embedding_array, quality_score = embedder.embed(
                        image_path=input_path,
                        normalize=params.normalize,
                        compute_quality=True,
                    )

                    # Create FaceEmbedding object
                    face_embedding = FaceEmbedding.from_numpy(
                        embedding=embedding_array,
                        quality_score=quality_score,
                    )

                    # Create result
                    result = FaceEmbeddingResult(
                        file_path=input_path,
                        embedding=face_embedding,
                        status="success",
                    )

                    file_results.append(result.model_dump())

                except FileNotFoundError:
                    logger.error(f"File not found: {input_path}")
                    result = FaceEmbeddingResult(
                        file_path=input_path,
                        embedding=None,
                        status="error",
                        error="File not found",
                    )
                    file_results.append(result.model_dump())

                except Exception as e:
                    logger.error(f"Failed to generate embedding for {input_path}: {e}")
                    result = FaceEmbeddingResult(
                        file_path=input_path,
                        embedding=None,
                        status="error",
                        error=str(e),
                    )
                    file_results.append(result.model_dump())

                # Report progress
                if progress_callback:
                    progress = int((index + 1) / total_files * 100)
                    progress_callback(progress)

            # Determine overall status
            all_success = all(r["status"] == "success" for r in file_results)
            any_success = any(r["status"] == "success" for r in file_results)

            if all_success:
                status = "ok"
            elif any_success:
                status = "ok"  # Partial success
                logger.warning(
                    f"Partial success: {sum(1 for r in file_results if r['status'] == 'success')}"
                    f"/{total_files} files processed successfully"
                )
            else:
                status = "error"
                return {
                    "status": status,
                    "error": "Failed to generate embeddings for all files",
                    "task_output": {
                        "files": file_results,
                        "total_files": total_files,
                    },
                }

            return {
                "status": status,
                "task_output": {
                    "files": file_results,
                    "total_files": total_files,
                    "normalize": params.normalize,
                },
            }

        except Exception as e:
            logger.exception(f"Unexpected error in FaceEmbeddingTask: {e}")
            return {"status": "error", "error": f"Task failed: {str(e)}"}
