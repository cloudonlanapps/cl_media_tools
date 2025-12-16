"""Face detection task implementation."""

import logging
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.face_detector import FaceDetector
from .schema import BoundingBox, FaceDetectionParams, FaceDetectionResult

logger = logging.getLogger(__name__)


class FaceDetectionTask(ComputeModule[FaceDetectionParams]):
    """Compute module for detecting faces in images using ONNX model."""

    def __init__(self):
        """Initialize face detection task."""
        super().__init__()
        self._detector: FaceDetector | None = None

    @property
    @override
    def task_type(self) -> str:
        return "face_detection"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return FaceDetectionParams

    def _get_detector(self) -> FaceDetector:
        """Get or create face detector instance (lazy loading)."""
        if self._detector is None:
            try:
                self._detector = FaceDetector()
                logger.info("Face detector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize face detector: {e}")
                raise

        return self._detector

    @override
    async def execute(
        self,
        job: Job,
        params: FaceDetectionParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Detect faces in input images.

        Args:
            job: Job instance
            params: FaceDetectionParams with input_paths and detection parameters
            progress_callback: Optional callback for progress updates (0-100)

        Returns:
            TaskResult with status and face detections for each image
        """
        try:
            # Initialize detector
            try:
                detector = self._get_detector()
            except Exception as e:
                logger.error(f"Face detector initialization failed: {e}")
                return {
                    "status": "error",
                    "error": (
                        f"Failed to initialize face detector: {e}. "
                        "Ensure ONNX Runtime is installed and the model can be downloaded."
                    ),
                }

            file_results: list[dict] = []
            total_files = len(params.input_paths)

            for index, input_path in enumerate(params.input_paths):
                try:
                    # Detect faces
                    detections = detector.detect(
                        image_path=input_path,
                        confidence_threshold=params.confidence_threshold,
                        nms_threshold=params.nms_threshold,
                    )

                    # Convert to BoundingBox objects
                    from PIL import Image

                    with Image.open(input_path) as img:
                        image_width, image_height = img.size

                    face_boxes = [
                        BoundingBox(
                            x1=det["x1"],
                            y1=det["y1"],
                            x2=det["x2"],
                            y2=det["y2"],
                            confidence=det["confidence"],
                        )
                        for det in detections
                    ]

                    # Create result
                    result = FaceDetectionResult(
                        file_path=input_path,
                        faces=face_boxes,
                        num_faces=len(face_boxes),
                        image_width=image_width,
                        image_height=image_height,
                    )

                    file_results.append(
                        {
                            "file_path": input_path,
                            "status": "success",
                            "detection": result.model_dump(),
                        }
                    )

                except FileNotFoundError:
                    logger.error(f"File not found: {input_path}")
                    file_results.append(
                        {
                            "file_path": input_path,
                            "status": "error",
                            "error": "File not found",
                        }
                    )

                except Exception as e:
                    logger.error(f"Failed to detect faces in {input_path}: {e}")
                    file_results.append(
                        {
                            "file_path": input_path,
                            "status": "error",
                            "error": str(e),
                        }
                    )

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
                    "error": "Failed to detect faces in all files",
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
                    "confidence_threshold": params.confidence_threshold,
                    "nms_threshold": params.nms_threshold,
                },
            }

        except Exception as e:
            logger.exception(f"Unexpected error in FaceDetectionTask: {e}")
            return {"status": "error", "error": f"Task failed: {str(e)}"}
