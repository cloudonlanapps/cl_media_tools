"""MobileCLIP embedding using ONNX model.

Model Source: Apple MobileCLIP (https://github.com/apple/ml-mobileclip)
ONNX Conversion: Required from PyTorch checkpoint
Input: 256x256 RGB images with CLIP normalization
Output: 512-dimensional image embedding (MobileCLIP-S2)
"""

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from ....utils.model_downloader import get_model_downloader

logger = logging.getLogger(__name__)

# Model configuration
# NOTE: This URL is a placeholder - MobileCLIP requires ONNX conversion from PyTorch
# Alternative: Use a pre-converted ONNX model from Hugging Face if available
MODEL_URL = "https://huggingface.co/apple/MobileCLIP-S2-onnx/resolve/main/image_encoder.onnx"
MODEL_FILENAME = "mobileclip_s2_image_encoder.onnx"
MODEL_SHA256 = None  # TODO: Add SHA256 hash for verification

# Input configuration (MobileCLIP-S2)
INPUT_SIZE = (256, 256)  # (height, width)
# CLIP normalization (different from ImageNet)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class ClipEmbedder:
    """ONNX-based MobileCLIP image embedding generator."""

    def __init__(self, model_path: str | Path | None = None):
        """Initialize MobileCLIP embedder.

        Args:
            model_path: Path to ONNX model file. If None, downloads from Hugging Face.

        Raises:
            FileNotFoundError: If model_path is provided but doesn't exist
            RuntimeError: If model loading fails
        """
        if model_path is None:
            # Download model using model_downloader
            downloader = get_model_downloader()
            logger.info(f"Downloading MobileCLIP model from {MODEL_URL}")
            try:
                model_path = downloader.download(
                    url=MODEL_URL,
                    filename=MODEL_FILENAME,
                    expected_sha256=MODEL_SHA256,
                )
            except Exception as e:
                logger.error(
                    f"Failed to download MobileCLIP model. "
                    "You may need to convert the model from PyTorch to ONNX manually. "
                    f"Error: {e}"
                )
                raise
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading MobileCLIP model from {model_path}")

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
            f"MobileCLIP model loaded. Input: {self.input_name}, Output: {self.output_name}"
        )

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for MobileCLIP inference.

        Args:
            image: PIL Image

        Returns:
            Preprocessed array in NCHW format with CLIP normalization
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to model input size
        image_resized = image.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.Resampling.BICUBIC)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image_resized, dtype=np.float32) / 255.0

        # Apply CLIP normalization (per-channel)
        img_array = (img_array - CLIP_MEAN.reshape(1, 1, 3)) / CLIP_STD.reshape(1, 1, 3)

        # Convert HWC to CHW (channels first)
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension: CHW -> NCHW
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def postprocess(self, embedding: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Post-process embedding (L2 normalization).

        Args:
            embedding: Raw embedding from model
            normalize: Whether to L2-normalize the embedding

        Returns:
            Processed embedding (optionally normalized)
        """
        # Remove batch dimension if present
        if embedding.ndim > 1:
            embedding = embedding.squeeze()

        # MobileCLIP outputs image features directly (512D for S2)
        # Ensure 1D vector
        if embedding.ndim > 1:
            embedding = embedding.flatten()

        if normalize:
            # L2 normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    def embed(
        self, image_path: str | Path, normalize: bool = True
    ) -> np.ndarray:
        """Generate MobileCLIP embedding for an image.

        Args:
            image_path: Path to input image
            normalize: Whether to L2-normalize the embedding

        Returns:
            1D numpy array (512D for MobileCLIP-S2)

        Raises:
            FileNotFoundError: If image file doesn't exist
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

        # Post-process (L2 normalization)
        embedding = self.postprocess(embedding, normalize=normalize)

        logger.info(
            f"Generated MobileCLIP embedding for {image_path}: dim={len(embedding)}"
        )

        return embedding
