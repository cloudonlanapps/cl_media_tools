"""Comprehensive test suite for MobileCLIP embedding plugin."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from cl_ml_tools.common.schemas import Job
from cl_ml_tools.plugins.clip_embedding.schema import (
    ClipEmbedding,
    ClipEmbeddingParams,
    ClipEmbeddingResult,
)
from cl_ml_tools.plugins.clip_embedding.task import ClipEmbeddingTask

# ============================================================================
# Test Fixtures
# ============================================================================


class MockProgressCallback:
    """Mock progress callback for testing."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, progress: int) -> None:
        self.calls.append(progress)


@pytest.fixture
def sample_image(tmp_path: Path) -> str:
    """Create a test image."""
    image_path = tmp_path / "test_image.jpg"

    # Create a simple image (256x256 to match MobileCLIP input)
    img = Image.new("RGB", (256, 256), color=(150, 200, 100))
    img.save(str(image_path), "JPEG")

    return str(image_path)


@pytest.fixture
def clip_embedding_task() -> ClipEmbeddingTask:
    """Create ClipEmbeddingTask instance."""
    return ClipEmbeddingTask()


@pytest.fixture
def mock_progress_callback() -> MockProgressCallback:
    """Create mock progress callback."""
    return MockProgressCallback()


# ============================================================================
# Test Class 1: Schema Validation
# ============================================================================


class TestClipEmbeddingParams:
    """Test ClipEmbeddingParams schema validation."""

    def test_default_params(self) -> None:
        """Test default ClipEmbeddingParams values."""
        params = ClipEmbeddingParams(input_paths=["/test/image.jpg"], output_paths=[])

        assert params.input_paths == ["/test/image.jpg"]
        assert params.output_paths == []
        assert params.normalize is True

    def test_custom_normalize(self) -> None:
        """Test ClipEmbeddingParams with custom normalize setting."""
        params = ClipEmbeddingParams(
            input_paths=["/test/image.jpg"], output_paths=[], normalize=False
        )

        assert params.normalize is False


class TestClipEmbedding:
    """Test ClipEmbedding schema validation."""

    def test_create_from_numpy(self) -> None:
        """Test creating ClipEmbedding from numpy array."""
        embedding_array = np.random.randn(512).astype(np.float32)

        clip_embedding = ClipEmbedding.from_numpy(
            embedding=embedding_array, normalized=True
        )

        assert clip_embedding.embedding_dim == 512
        assert len(clip_embedding.embedding) == 512
        assert clip_embedding.normalized is True

    def test_to_numpy(self) -> None:
        """Test converting ClipEmbedding to numpy array."""
        embedding_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        clip_embedding = ClipEmbedding(
            embedding=embedding_list, embedding_dim=5, normalized=False
        )

        embedding_array = clip_embedding.to_numpy()

        assert isinstance(embedding_array, np.ndarray)
        assert embedding_array.dtype == np.float32
        assert len(embedding_array) == 5
        np.testing.assert_array_almost_equal(embedding_array, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_embedding_dimension_consistency(self) -> None:
        """Test that embedding dimension matches length when using from_numpy."""
        # from_numpy() ensures consistency automatically
        embedding_array = np.random.randn(512).astype(np.float32)
        clip_embedding = ClipEmbedding.from_numpy(
            embedding=embedding_array, normalized=True
        )

        assert len(clip_embedding.embedding) == 512
        assert clip_embedding.embedding_dim == 512

    def test_correct_dimension_512(self) -> None:
        """Test that 512D embeddings are correctly validated."""
        embedding_array = np.random.randn(512).astype(np.float32)
        clip_embedding = ClipEmbedding.from_numpy(
            embedding=embedding_array, normalized=True
        )

        assert clip_embedding.embedding_dim == 512
        assert len(clip_embedding.embedding) == 512


class TestClipEmbeddingResult:
    """Test ClipEmbeddingResult schema validation."""

    def test_success_result(self) -> None:
        """Test successful MobileCLIP embedding result."""
        embedding = ClipEmbedding(
            embedding=[0.1] * 512, embedding_dim=512, normalized=True
        )

        result = ClipEmbeddingResult(
            file_path="/test/image.jpg", embedding=embedding, status="success"
        )

        assert result.file_path == "/test/image.jpg"
        assert result.embedding is not None
        assert result.embedding.embedding_dim == 512
        assert result.embedding.normalized is True
        assert result.status == "success"
        assert result.error is None

    def test_error_result(self) -> None:
        """Test error MobileCLIP embedding result."""
        result = ClipEmbeddingResult(
            file_path="/test/image.jpg",
            embedding=None,
            status="error",
            error="File not found",
        )

        assert result.file_path == "/test/image.jpg"
        assert result.embedding is None
        assert result.status == "error"
        assert result.error == "File not found"


# ============================================================================
# Test Class 2: Task Execution
# ============================================================================


class TestClipEmbeddingTask:
    """Test ClipEmbeddingTask execution."""

    def test_task_type(self, clip_embedding_task: ClipEmbeddingTask) -> None:
        """Test that task type is correctly set."""
        assert clip_embedding_task.task_type == "clip_embedding"

    def test_get_schema(self, clip_embedding_task: ClipEmbeddingTask) -> None:
        """Test that get_schema returns ClipEmbeddingParams."""
        schema = clip_embedding_task.get_schema()
        assert schema == ClipEmbeddingParams

    @pytest.mark.asyncio
    async def test_execute_with_mocked_embedder(
        self,
        clip_embedding_task: ClipEmbeddingTask,
        sample_image: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with mocked MobileCLIP embedder."""
        # Create job
        job = Job(
            job_id="test-job-001",
            task_type="clip_embedding",
            params={
                "input_paths": [sample_image],
                "output_paths": [],
                "normalize": True,
            },
        )

        params = ClipEmbeddingParams(**job.params)

        # Mock the embedder
        mock_embedder = Mock()
        mock_embedding = np.random.randn(512).astype(np.float32)
        # Normalize it
        mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
        mock_embedder.embed.return_value = mock_embedding

        with patch.object(
            clip_embedding_task, "_get_embedder", return_value=mock_embedder
        ):
            result = await clip_embedding_task.execute(job, params, mock_progress_callback)

        # Verify result
        assert result["status"] == "ok"
        assert "task_output" in result

        task_output = result["task_output"]
        assert task_output["total_files"] == 1
        assert task_output["normalize"] is True
        assert len(task_output["files"]) == 1

        file_result = task_output["files"][0]
        assert file_result["file_path"] == sample_image
        assert file_result["status"] == "success"
        assert file_result["embedding"]["embedding_dim"] == 512
        assert file_result["embedding"]["normalized"] is True

        # Verify progress callback was called
        assert len(mock_progress_callback.calls) == 1
        assert mock_progress_callback.calls[0] == 100

    @pytest.mark.asyncio
    async def test_execute_file_not_found(
        self,
        clip_embedding_task: ClipEmbeddingTask,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with non-existent file."""
        job = Job(
            job_id="test-job-002",
            task_type="clip_embedding",
            params={
                "input_paths": ["/nonexistent/image.jpg"],
                "output_paths": [],
                "normalize": True,
            },
        )

        params = ClipEmbeddingParams(**job.params)

        # Mock the embedder to raise FileNotFoundError
        mock_embedder = Mock()
        mock_embedder.embed.side_effect = FileNotFoundError("Image file not found")

        with patch.object(
            clip_embedding_task, "_get_embedder", return_value=mock_embedder
        ):
            result = await clip_embedding_task.execute(job, params, mock_progress_callback)

        # Should return error status
        assert result["status"] == "error"

        task_output = result["task_output"]
        file_result = task_output["files"][0]
        assert file_result["status"] == "error"
        assert "not found" in file_result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_multiple_files(
        self,
        clip_embedding_task: ClipEmbeddingTask,
        sample_image: str,
        tmp_path: Path,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with multiple files."""
        # Create second image
        image_path_2 = tmp_path / "test_image_2.jpg"
        img = Image.new("RGB", (256, 256), color=(80, 120, 220))
        img.save(str(image_path_2), "JPEG")

        job = Job(
            job_id="test-job-003",
            task_type="clip_embedding",
            params={
                "input_paths": [sample_image, str(image_path_2)],
                "output_paths": [],
                "normalize": True,
            },
        )

        params = ClipEmbeddingParams(**job.params)

        # Mock the embedder
        mock_embedder = Mock()

        def mock_embed_side_effect(image_path, normalize):
            embedding = np.random.randn(512).astype(np.float32)
            if normalize:
                embedding = embedding / np.linalg.norm(embedding)
            return embedding

        mock_embedder.embed.side_effect = mock_embed_side_effect

        with patch.object(
            clip_embedding_task, "_get_embedder", return_value=mock_embedder
        ):
            result = await clip_embedding_task.execute(job, params, mock_progress_callback)

        # Verify result
        assert result["status"] == "ok"

        task_output = result["task_output"]
        assert task_output["total_files"] == 2
        assert len(task_output["files"]) == 2

        # Verify both files processed
        assert task_output["files"][0]["status"] == "success"
        assert task_output["files"][1]["status"] == "success"

        # Verify progress callback was called twice (50%, 100%)
        assert len(mock_progress_callback.calls) == 2
        assert mock_progress_callback.calls == [50, 100]

    @pytest.mark.asyncio
    async def test_execute_with_normalization_disabled(
        self,
        clip_embedding_task: ClipEmbeddingTask,
        sample_image: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with normalization disabled."""
        job = Job(
            job_id="test-job-004",
            task_type="clip_embedding",
            params={
                "input_paths": [sample_image],
                "output_paths": [],
                "normalize": False,
            },
        )

        params = ClipEmbeddingParams(**job.params)

        # Mock the embedder
        mock_embedder = Mock()
        mock_embedding = np.random.randn(512).astype(np.float32)
        mock_embedder.embed.return_value = mock_embedding

        with patch.object(
            clip_embedding_task, "_get_embedder", return_value=mock_embedder
        ):
            result = await clip_embedding_task.execute(job, params, mock_progress_callback)

        # Verify embedder was called with normalize=False
        mock_embedder.embed.assert_called_once()
        call_kwargs = mock_embedder.embed.call_args[1]
        assert call_kwargs["normalize"] is False

        # Verify result
        assert result["status"] == "ok"
        assert result["task_output"]["normalize"] is False

    @pytest.mark.asyncio
    async def test_execute_embedder_initialization_failure(
        self,
        clip_embedding_task: ClipEmbeddingTask,
        sample_image: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution when embedder fails to initialize."""
        job = Job(
            job_id="test-job-005",
            task_type="clip_embedding",
            params={
                "input_paths": [sample_image],
                "output_paths": [],
            },
        )

        params = ClipEmbeddingParams(**job.params)

        # Mock embedder initialization failure
        with patch.object(
            clip_embedding_task,
            "_get_embedder",
            side_effect=RuntimeError("MobileCLIP model requires ONNX conversion"),
        ):
            result = await clip_embedding_task.execute(job, params, mock_progress_callback)

        # Should return error
        assert result["status"] == "error"
        assert "MobileCLIP model requires ONNX conversion" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_partial_success(
        self,
        clip_embedding_task: ClipEmbeddingTask,
        sample_image: str,
        tmp_path: Path,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with partial success (some files fail)."""
        # Create second image
        image_path_2 = tmp_path / "test_image_2.jpg"
        img = Image.new("RGB", (256, 256), color=(80, 120, 220))
        img.save(str(image_path_2), "JPEG")

        job = Job(
            job_id="test-job-006",
            task_type="clip_embedding",
            params={
                "input_paths": [sample_image, str(image_path_2)],
                "output_paths": [],
                "normalize": True,
            },
        )

        params = ClipEmbeddingParams(**job.params)

        # Mock the embedder - first succeeds, second fails
        mock_embedder = Mock()
        mock_embedder.embed.side_effect = [
            np.random.randn(512).astype(np.float32),
            RuntimeError("Inference failed"),
        ]

        with patch.object(
            clip_embedding_task, "_get_embedder", return_value=mock_embedder
        ):
            result = await clip_embedding_task.execute(job, params, mock_progress_callback)

        # Should return ok (partial success)
        assert result["status"] == "ok"

        task_output = result["task_output"]
        assert task_output["total_files"] == 2

        # First file succeeded
        assert task_output["files"][0]["status"] == "success"

        # Second file failed
        assert task_output["files"][1]["status"] == "error"
        assert "Inference failed" in task_output["files"][1]["error"]


# ============================================================================
# Test Class 3: Algorithm Unit Tests
# ============================================================================


class TestClipEmbeddingAlgorithm:
    """Test ClipEmbedder algorithm components."""

    def test_l2_normalization_logic(self) -> None:
        """Test L2 normalization produces unit vectors."""
        # Create a random embedding
        embedding = np.random.randn(512).astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        normalized = embedding / norm

        # Verify unit vector (L2 norm = 1)
        assert np.abs(np.linalg.norm(normalized) - 1.0) < 1e-6

    def test_embedding_dimension_consistency(self) -> None:
        """Test that embeddings maintain 512D dimensionality."""
        embedding_array = np.random.randn(512).astype(np.float32)
        clip_embedding = ClipEmbedding.from_numpy(
            embedding=embedding_array, normalized=True
        )

        assert clip_embedding.embedding_dim == 512
        assert len(clip_embedding.embedding) == 512

        # Convert back and verify
        converted = clip_embedding.to_numpy()
        assert len(converted) == 512
        assert converted.dtype == np.float32

    def test_cosine_similarity_computation(self) -> None:
        """Test cosine similarity between normalized embeddings."""
        # Create two random normalized embeddings
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)

        emb2 = np.random.randn(512).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)

        # Cosine similarity = dot product (for normalized vectors)
        cosine_sim = np.dot(emb1, emb2)

        # Should be in range [-1, 1]
        assert -1.0 <= cosine_sim <= 1.0

    def test_identical_embeddings_high_similarity(self) -> None:
        """Test that identical embeddings have similarity ~1.0."""
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # Cosine similarity with itself
        cosine_sim = np.dot(embedding, embedding)

        # Should be very close to 1.0
        assert np.abs(cosine_sim - 1.0) < 1e-6

    def test_text_image_similarity_concept(self) -> None:
        """Test the concept of text-image similarity (for semantic search)."""
        # In actual usage, text and image embeddings from CLIP should be comparable
        # Here we simulate this with two random normalized 512D vectors

        # Simulated image embedding
        image_emb = np.random.randn(512).astype(np.float32)
        image_emb = image_emb / np.linalg.norm(image_emb)

        # Simulated text embedding (from CLIP text encoder)
        text_emb = np.random.randn(512).astype(np.float32)
        text_emb = text_emb / np.linalg.norm(text_emb)

        # Cosine similarity for text-image matching
        similarity = np.dot(image_emb, text_emb)

        # Should be in valid range
        assert -1.0 <= similarity <= 1.0

        # In practice, semantically similar text-image pairs would have
        # similarity > 0.5, and dissimilar pairs < 0.2
        # But with random embeddings, we just verify the computation works

    def test_normalized_flag_consistency(self) -> None:
        """Test that normalized flag is correctly maintained."""
        # Create unnormalized embedding
        embedding_unnorm = np.random.randn(512).astype(np.float32) * 10  # Scale up

        clip_emb_unnorm = ClipEmbedding.from_numpy(
            embedding=embedding_unnorm, normalized=False
        )
        assert clip_emb_unnorm.normalized is False

        # Create normalized embedding
        embedding_norm = embedding_unnorm / np.linalg.norm(embedding_unnorm)

        clip_emb_norm = ClipEmbedding.from_numpy(embedding=embedding_norm, normalized=True)
        assert clip_emb_norm.normalized is True
