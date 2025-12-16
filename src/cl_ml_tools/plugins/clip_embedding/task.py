"""MobileCLIP embedding task implementation."""

import logging
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.clip_embedder import ClipEmbedder
from .schema import ClipEmbedding, ClipEmbeddingParams, ClipEmbeddingResult

logger = logging.getLogger(__name__)


class ClipEmbeddingTask(ComputeModule[ClipEmbeddingParams]):
    """Compute module for generating MobileCLIP embeddings using ONNX model."""

    def __init__(self):
        """Initialize MobileCLIP embedding task."""
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
        """Get or create MobileCLIP embedder instance (lazy loading)."""
        if self._embedder is None:
            try:
                self._embedder = ClipEmbedder()
                logger.info("MobileCLIP embedder initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MobileCLIP embedder: {e}")
                raise

        return self._embedder

    @override
    async def execute(
        self,
        job: Job,
        params: ClipEmbeddingParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Generate MobileCLIP embeddings for input images.

        Args:
            job: Job instance
            params: ClipEmbeddingParams with input_paths and normalization settings
            progress_callback: Optional callback for progress updates (0-100)

        Returns:
            TaskResult with status and embeddings for each image
        """
        try:
            # Initialize embedder
            try:
                embedder = self._get_embedder()
            except Exception as e:
                logger.error(f"MobileCLIP embedder initialization failed: {e}")
                return {
                    "status": "error",
                    "error": (
                        f"Failed to initialize MobileCLIP embedder: {e}. "
                        "Ensure ONNX Runtime is installed and the model is available. "
                        "MobileCLIP may require manual ONNX conversion from PyTorch."
                    ),
                }

            file_results: list[dict] = []
            total_files = len(params.input_paths)

            for index, input_path in enumerate(params.input_paths):
                try:
                    # Generate embedding
                    embedding_array = embedder.embed(
                        image_path=input_path,
                        normalize=params.normalize,
                    )

                    # Create ClipEmbedding object
                    clip_embedding = ClipEmbedding.from_numpy(
                        embedding=embedding_array,
                        normalized=params.normalize,
                    )

                    # Create result
                    result = ClipEmbeddingResult(
                        file_path=input_path,
                        embedding=clip_embedding,
                        status="success",
                    )

                    file_results.append(result.model_dump())

                except FileNotFoundError:
                    logger.error(f"File not found: {input_path}")
                    result = ClipEmbeddingResult(
                        file_path=input_path,
                        embedding=None,
                        status="error",
                        error="File not found",
                    )
                    file_results.append(result.model_dump())

                except Exception as e:
                    logger.error(f"Failed to generate MobileCLIP embedding for {input_path}: {e}")
                    result = ClipEmbeddingResult(
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
            logger.exception(f"Unexpected error in ClipEmbeddingTask: {e}")
            return {"status": "error", "error": f"Task failed: {str(e)}"}
