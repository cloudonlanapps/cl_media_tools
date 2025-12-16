"""Face detection task implementation."""

import json
import logging
from typing import Callable, override

from PIL import Image

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.face_detector import FaceDetector
from .schema import BoundingBox, FaceDetectionParams

logger = logging.getLogger(__name__)


class FaceDetectionTask(ComputeModule[FaceDetectionParams]):
    """Compute module for detecting faces in an image using ONNX model."""

    def __init__(self) -> None:
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
        if self._detector is None:
            self._detector = FaceDetector()
            logger.info("Face detector initialized successfully")
        return self._detector

    @override
    async def execute(
        self,
        job: Job,
        params: FaceDetectionParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Detect faces in a single image and write results to disk."""

        # Phase 1: get detector
        try:
            detector = self._get_detector()
        except Exception as exc:  # noqa: BLE001
            logger.error("Face detector initialization failed", exc_info=exc)
            return TaskResult(
                status="error",
                error=(
                    f"Failed to initialize face detector: {exc}. "
                    "Ensure ONNX Runtime is installed and the model can be downloaded."
                ),
            )

        # Phase 2: detect faces
        try:
            detections = detector.detect(
                image_path=params.input_path,
                confidence_threshold=params.confidence_threshold,
                nms_threshold=params.nms_threshold,
            )

            with Image.open(params.input_path) as img:
                image_width, image_height = img.size

            faces = [
                BoundingBox(
                    x1=det["x1"],
                    y1=det["y1"],
                    x2=det["x2"],
                    y2=det["y2"],
                    confidence=det["confidence"],
                ).model_dump()
                for det in detections
            ]

            detection_output = {
                "faces": faces,
                "num_faces": len(faces),
                "image_width": image_width,
                "image_height": image_height,
            }

            # Persist detection result as JSON
            with open(params.output_path, "w", encoding="utf-8") as f:
                json.dump(detection_output, f, indent=2)

            if progress_callback:
                progress_callback(100)

            return TaskResult(
                status="ok",
                task_output=detection_output,
            )

        except FileNotFoundError:
            logger.error("File not found: %s", params.input_path)
            return TaskResult(
                status="error",
                error="Input file not found",
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to detect faces in %s",
                params.input_path,
                exc_info=exc,
            )
            return TaskResult(
                status="error",
                error=str(exc),
            )
