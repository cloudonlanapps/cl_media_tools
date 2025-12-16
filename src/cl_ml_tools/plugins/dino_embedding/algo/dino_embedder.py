"""DINOv2 embedding using ONNX model.

Model Source: https://huggingface.co/RoundtTble/dinov2_vits14_onnx
Input: 224x224 RGB images with ImageNet normalization
Output: 384-dimensional CLS token embedding
"""

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from ....utils.model_downloader import get_model_downloader

logger = logging.getLogger(__name__)

# Model configuration
MODEL_URL = "https://huggingface.co/RoundtTble/dinov2_vits14_onnx/resolve/main/model.onnx"
MODEL_FILENAME = "dinov2_vits14.onnx"
MODEL_SHA256 = None  # TODO: Add SHA256 hash for verification

# Input configuration
INPUT_SIZE = (224, 224)  # (height, width)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DinoEmbedder:
    """ONNX-based DINOv2 embedding generator."""

    def __init__(self, model_path: str | Path | None = None):
        """Initialize DINOv2 embedder.

        Args:
            model_path: Path to ONNX model file. If None, downloads from Hugging Face.
        """
        if model_path is None:
            # Download model using model_downloader
            downloader = get_model_downloader()
            logger.info(f"Downloading DINOv2 model from {MODEL_URL}")
            model_path = downloader.download(
                url=MODEL_URL,
                filename=MODEL_FILENAME,
                expected_sha256=MODEL_SHA256,
            )
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading DINOv2 model from {model_path}")

        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Get model input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(
            f"DINOv2 model loaded. Input: {self.input_name}, Output: {self.output_name}"
        )

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for DINOv2 inference.

        Args:
            image: PIL Image

        Returns:
            Preprocessed array in NCHW format with ImageNet normalization
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to model input size
        image_resized = image.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.Resampling.BILINEAR)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image_resized, dtype=np.float32) / 255.0

        # Apply ImageNet normalization (per-channel)
        img_array = (img_array - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)

        # Convert HWC to CHW (channels first)
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension: CHW -> NCHW
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def postprocess(self, embedding: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Post-process embedding (L2 normalization, CLS token extraction).

        Args:
            embedding: Raw embedding from model
            normalize: Whether to L2-normalize the embedding

        Returns:
            Processed embedding (CLS token, optionally normalized)
        """
        # Remove batch dimension if present
        if embedding.ndim > 1:
            embedding = embedding.squeeze()

        # For DINOv2, the output is typically the CLS token directly (384D for vits14)
        # If model outputs sequence, take first token (CLS)
        if embedding.ndim > 1:
            embedding = embedding[0]  # Take CLS token

        if normalize:
            # L2 normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    def embed(
        self, image_path: str | Path, normalize: bool = True
    ) -> np.ndarray:
        """Generate DINOv2 embedding for an image.

        Args:
            image_path: Path to input image
            normalize: Whether to L2-normalize the embedding

        Returns:
            1D numpy array (384D for DINOv2 ViT-S/14)
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image
        image = Image.open(image_path)

        # Preprocess
        input_array = self.preprocess(image)

        # Run inference
        embedding = self.session.run([self.output_name], {self.input_name: input_array})[0]

        # Post-process (CLS token extraction and L2 normalization)
        embedding = self.postprocess(embedding, normalize=normalize)

        logger.info(
            f"Generated DINOv2 embedding for {image_path}: dim={len(embedding)}"
        )

        return embedding
