"""Face embedding using ONNX model (ArcFace).

Model Source: https://huggingface.co/garavv/arcface-onnx
Input: 112x112 RGB face images
Output: 512-dimensional embedding
"""

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from ....utils.model_downloader import get_model_downloader

logger = logging.getLogger(__name__)

# Model configuration
MODEL_URL = "https://huggingface.co/garavv/arcface-onnx/resolve/main/arcface.onnx"
MODEL_FILENAME = "arcface_face_embedding.onnx"
MODEL_SHA256 = None  # TODO: Add SHA256 hash for verification

# Expected input shape for ArcFace
INPUT_SIZE = (112, 112)  # (height, width)


class FaceEmbedder:
    """ONNX-based face embedding generator using ArcFace model."""

    def __init__(self, model_path: str | Path | None = None):
        """Initialize face embedder.

        Args:
            model_path: Path to ONNX model file. If None, downloads from Hugging Face.
        """
        if model_path is None:
            # Download model using model_downloader
            downloader = get_model_downloader()
            logger.info(f"Downloading face embedding model from {MODEL_URL}")
            model_path = downloader.download(
                url=MODEL_URL,
                filename=MODEL_FILENAME,
                expected_sha256=MODEL_SHA256,
            )
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading face embedding model from {model_path}")

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
        self.output_name = self.session.get_outputs()[0].name

        logger.info(
            f"Model loaded. Input: {self.input_name}, Output: {self.output_name}"
        )

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess face image for embedding extraction.

        Args:
            image: PIL Image of a cropped face

        Returns:
            Preprocessed array in NCHW format, float32, range [0, 1] or normalized
        """
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

        if normalize:
            # L2 normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    def compute_quality_score(self, image: Image.Image) -> float:
        """Compute quality score for a face image based on sharpness/blur.

        Args:
            image: PIL Image of a face

        Returns:
            Quality score in [0.0, 1.0] range
        """
        # Simple Laplacian variance for blur detection
        # Higher variance = sharper image = better quality
        img_gray = image.convert("L")
        img_array = np.array(img_gray, dtype=np.float32)

        # Compute Laplacian
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        from scipy import signal

        lap_img = signal.convolve2d(img_array, laplacian, mode="same", boundary="symm")

        # Compute variance
        variance = np.var(lap_img)

        # Normalize to [0, 1] range (empirical threshold)
        # Variance > 100 is considered sharp
        quality = min(variance / 100.0, 1.0)

        return float(quality)

    def embed(
        self, image_path: str | Path, normalize: bool = True, compute_quality: bool = True
    ) -> tuple[np.ndarray, float | None]:
        """Generate embedding for a face image.

        Args:
            image_path: Path to input face image (should be cropped face)
            normalize: Whether to L2-normalize the embedding
            compute_quality: Whether to compute quality score

        Returns:
            Tuple of (embedding, quality_score)
                - embedding: 1D numpy array (512D for ArcFace)
                - quality_score: Quality score in [0.0, 1.0] or None
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

        # Compute quality score if requested
        quality_score = None
        if compute_quality:
            try:
                quality_score = self.compute_quality_score(image)
            except ImportError:
                logger.warning(
                    "scipy not available, skipping quality score computation. "
                    "Install scipy for quality scores: pip install scipy"
                )
            except Exception as e:
                logger.warning(f"Failed to compute quality score: {e}")

        logger.info(
            f"Generated embedding for {image_path}: "
            f"dim={len(embedding)}, quality={quality_score}"
        )

        return embedding, quality_score
