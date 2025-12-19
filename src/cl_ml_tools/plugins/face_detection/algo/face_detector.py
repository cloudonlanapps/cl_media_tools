# pyright: reportAny=false

"""Face detection using ONNX model (MediaPipe Face Detection).

Model Source: https://huggingface.co/qualcomm/MediaPipe-Face-Detection
Model File: MediaPipeFaceLandmarkDetector.onnx (2.45 MB)
"""

import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from PIL import Image

from ....utils.model_downloader import get_model_downloader

logger = logging.getLogger(__name__)

# Model configuration
MODEL_URL = (
    "https://huggingface.co/qualcomm/MediaPipe-Face-Detection/"
    "resolve/main/MediaPipe-Face-Detection_FaceDetector_float.onnx.zip"
)
MODEL_ZIP_FILENAME = "mediapipe_face_detector.onnx.zip"
MODEL_FILENAME = "mediapipe_face_detector.onnx"
MODEL_SHA256: str | None = None  # TODO: Add SHA256 hash for verification

# Expected input shape for MediaPipe Face Detection
INPUT_SIZE: tuple[int, int] = (224, 224)  # (height, width)


class FaceDetection(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class FaceDetector:
    """ONNX-based face detector using MediaPipe Face Detection model."""

    session: ort.InferenceSession
    input_name: str
    output_names: list[str]

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path is None:
            downloader = get_model_downloader()
            logger.info("Downloading face detection model from %s", MODEL_URL)
            model_path = downloader.download(
                url=MODEL_URL,
                filename=MODEL_ZIP_FILENAME,
                expected_sha256=MODEL_SHA256,
                auto_extract=True,
                extract_pattern="*.onnx",
            )
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading face detection model from %s", model_path)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(
            "Model loaded. Input: %s, Outputs: %s",
            self.input_name,
            self.output_names,
        )

    def preprocess(self, image: Image.Image) -> tuple[NDArray[np.float32], tuple[int, int]]:
        original_size = image.size  # (width, height)

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_resized = image.resize(
            (INPUT_SIZE[1], INPUT_SIZE[0]),
            Image.Resampling.BILINEAR,
        )

        img_array: NDArray[np.float32] = np.asarray(image_resized, dtype=np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, original_size

    def postprocess(
        self,
        outputs: list[NDArray[np.float32]],
        original_size: tuple[int, int],
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
    ) -> list[FaceDetection]:
        """
        Post-process MediaPipe face detection outputs.

        MediaPipe Face Detection typically outputs:
        - Output 0: Bounding boxes in format [batch, num_boxes, 4] (normalized coords)
        - Output 1: Confidence scores in format [batch, num_boxes]

        Args:
            outputs: Raw model outputs
            original_size: Original image size (width, height)
            confidence_threshold: Minimum confidence to keep detection
            nms_threshold: IoU threshold for Non-Maximum Suppression

        Returns:
            List of face detections with absolute coordinates
        """
        if not outputs or len(outputs) < 2:
            logger.warning("Unexpected number of outputs: %d", len(outputs))
            return []

        # Extract bounding boxes and scores
        # Assuming outputs[0] = boxes [batch, num_boxes, 4]
        # and outputs[1] = scores [batch, num_boxes] or [batch, num_boxes, 1]
        boxes: NDArray[np.float32] = outputs[0]
        scores: NDArray[np.float32] = outputs[1]

        # Remove batch dimension
        if boxes.ndim == 3:
            boxes = boxes[0]  # [num_boxes, 4]
        if scores.ndim == 3:
            scores = scores[0]  # [num_boxes, 1]
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores.squeeze(-1)  # [num_boxes]

        # Filter by confidence threshold
        mask = scores >= confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return []

        # Apply Non-Maximum Suppression
        keep_indices = self._nms(boxes, scores, nms_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]

        # Convert normalized coordinates to absolute coordinates
        orig_width, orig_height = original_size
        detections: list[FaceDetection] = []

        for box, score in zip(boxes, scores):
            # box format: [x_center, y_center, width, height] or [x1, y1, x2, y2]
            # MediaPipe typically uses center format, normalized to [0, 1]
            if box[2] <= 1.0 and box[3] <= 1.0:  # Likely normalized width/height (center format)
                x_center, y_center, width, height = box
                x1 = (x_center - width / 2) * orig_width
                y1 = (y_center - height / 2) * orig_height
                x2 = (x_center + width / 2) * orig_width
                y2 = (y_center + height / 2) * orig_height
            else:  # Likely [x1, y1, x2, y2] format
                x1, y1, x2, y2 = box * np.array([orig_width, orig_height, orig_width, orig_height])

            detections.append(
                FaceDetection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(score),
                )
            )

        return detections

    def _nms(
        self,
        boxes: NDArray[np.float32],
        scores: NDArray[np.float32],
        iou_threshold: float,
    ) -> NDArray[np.int_]:
        """
        Non-Maximum Suppression to remove overlapping bounding boxes.

        Args:
            boxes: Bounding boxes [num_boxes, 4] (normalized or absolute)
            scores: Confidence scores [num_boxes]
            iou_threshold: IoU threshold for suppression

        Returns:
            Indices of boxes to keep
        """
        # Sort by scores (descending)
        sorted_indices = np.argsort(scores)[::-1]

        keep_indices: list[int] = []

        while len(sorted_indices) > 0:
            # Keep highest scoring box
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)

            if len(sorted_indices) == 1:
                break

            # Calculate IoU with remaining boxes
            current_box = boxes[current_idx]
            remaining_boxes = boxes[sorted_indices[1:]]

            ious = self._calculate_iou(current_box, remaining_boxes)

            # Keep boxes with IoU below threshold
            sorted_indices = sorted_indices[1:][ious < iou_threshold]

        return np.array(keep_indices, dtype=np.int_)

    def _calculate_iou(
        self,
        box: NDArray[np.float32],
        boxes: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Calculate Intersection over Union (IoU) between one box and multiple boxes.

        Args:
            box: Single box [4] (x_center, y_center, width, height) normalized
            boxes: Multiple boxes [num_boxes, 4]

        Returns:
            IoU values [num_boxes]
        """
        # Convert center format to corner format for IoU calculation
        def to_corners(b: NDArray[np.float32]) -> NDArray[np.float32]:
            if b.ndim == 1:
                x_c, y_c, w, h = b
                return np.array([x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2], dtype=np.float32)
            else:
                x_c, y_c, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
                return np.column_stack([x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2]).astype(np.float32)

        box_corners = to_corners(box)
        boxes_corners = to_corners(boxes)

        # Calculate intersection
        x1 = np.maximum(box_corners[0], boxes_corners[:, 0])
        y1 = np.maximum(box_corners[1], boxes_corners[:, 1])
        x2 = np.minimum(box_corners[2], boxes_corners[:, 2])
        y2 = np.minimum(box_corners[3], boxes_corners[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Calculate union
        box_area = (box_corners[2] - box_corners[0]) * (box_corners[3] - box_corners[1])
        boxes_area = (boxes_corners[:, 2] - boxes_corners[:, 0]) * (
            boxes_corners[:, 3] - boxes_corners[:, 1]
        )
        union = box_area + boxes_area - intersection

        # Calculate IoU
        iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero

        return iou

    def detect(
        self,
        image_path: str | Path,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
    ) -> list[FaceDetection]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as image:
            input_array, original_size = self.preprocess(image)

        outputs: list[NDArray[np.float32]] = self.session.run(
            self.output_names,
            {self.input_name: input_array},
        )

        detections = self.postprocess(
            outputs,
            original_size,
            confidence_threshold,
            nms_threshold,
        )

        logger.info(
            "Detected %d faces in %s",
            len(detections),
            image_path,
        )

        return detections
