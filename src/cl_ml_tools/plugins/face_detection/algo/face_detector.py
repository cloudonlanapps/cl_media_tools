"""Face detection using ONNX model (MediaPipe Face Detection).

Model Source: https://huggingface.co/qualcomm/MediaPipe-Face-Detection
Model File: MediaPipeFaceLandmarkDetector.onnx (2.45 MB)
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from ....utils.model_downloader import get_model_downloader

logger = logging.getLogger(__name__)

# Model configuration
MODEL_URL = "https://huggingface.co/qualcomm/MediaPipe-Face-Detection/resolve/main/MediaPipeFaceLandmarkDetector.onnx"
MODEL_FILENAME = "mediapipe_face_detection.onnx"
MODEL_SHA256 = None  # TODO: Add SHA256 hash for verification

# Expected input shape for MediaPipe Face Detection
INPUT_SIZE = (192, 192)  # (height, width)


class FaceDetector:
    """ONNX-based face detector using MediaPipe Face Detection model."""

    def __init__(self, model_path: str | Path | None = None):
        """Initialize face detector.

        Args:
            model_path: Path to ONNX model file. If None, downloads from Hugging Face.
        """
        if model_path is None:
            # Download model using model_downloader
            downloader = get_model_downloader()
            logger.info(f"Downloading face detection model from {MODEL_URL}")
            model_path = downloader.download(
                url=MODEL_URL,
                filename=MODEL_FILENAME,
                expected_sha256=MODEL_SHA256,
            )
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading face detection model from {model_path}")

        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Get model input/output names and shapes
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(
            f"Model loaded. Input: {self.input_name}, Outputs: {self.output_names}"
        )

    def preprocess(self, image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
        """Preprocess image for face detection.

        Args:
            image: PIL Image

        Returns:
            Tuple of (preprocessed_array, original_size)
                - preprocessed_array: NCHW format, float32, range [0, 1]
                - original_size: (width, height) of original image
        """
        original_size = image.size  # (width, height)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to model input size
        image_resized = image.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.Resampling.BILINEAR)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image_resized, dtype=np.float32) / 255.0

        # Convert HWC to CHW (channels first)
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension: CHW -> NCHW
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, original_size

    def postprocess(
        self,
        outputs: list[np.ndarray],
        original_size: tuple[int, int],
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
    ) -> list[dict[str, Any]]:
        """Post-process model outputs to extract face detections.

        Args:
            outputs: Raw model outputs
            original_size: (width, height) of original image
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS threshold for overlapping boxes

        Returns:
            List of detections with normalized coordinates and confidence scores
        """
        # Note: This is a placeholder implementation
        # The actual post-processing depends on the specific model output format
        # MediaPipe Face Detection outputs need to be parsed according to model specs

        detections = []

        # TODO: Implement actual post-processing based on model outputs
        # This requires understanding the specific output format of the ONNX model
        # For now, returning empty list
        logger.warning(
            "Face detection post-processing not fully implemented. "
            "Please update this method based on the specific ONNX model output format."
        )

        return detections

    def detect(
        self,
        image_path: str | Path,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
    ) -> list[dict[str, Any]]:
        """Detect faces in an image.

        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS threshold for overlapping boxes

        Returns:
            List of detections, each with:
                - x1, y1, x2, y2: Normalized coordinates (0.0-1.0)
                - confidence: Detection confidence score
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image
        image = Image.open(image_path)

        # Preprocess
        input_array, original_size = self.preprocess(image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_array})

        # Post-process
        detections = self.postprocess(
            outputs, original_size, confidence_threshold, nms_threshold
        )

        logger.info(f"Detected {len(detections)} faces in {image_path}")

        return detections
