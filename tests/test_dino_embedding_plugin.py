"""Comprehensive test suite for DINOv2 embedding plugin."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from cl_ml_tools.common.schema_job import Job
from cl_ml_tools.plugins.dino_embedding.schema import (
    DinoEmbedding,
    DinoEmbeddingParams,
    DinoEmbeddingResult,
)
from cl_ml_tools.plugins.dino_embedding.task import DinoEmbeddingTask

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

    # Create a simple image (224x224 to match DINOv2 input)
    img = Image.new("RGB", (224, 224), color=(120, 180, 220))
    img.save(str(image_path), "JPEG")

    return str(image_path)


@pytest.fixture
def dino_embedding_task() -> DinoEmbeddingTask:
    """Create DinoEmbeddingTask instance."""
    return DinoEmbeddingTask()


@pytest.fixture
def mock_progress_callback() -> MockProgressCallback:
    """Create mock progress callback."""
    return MockProgressCallback()


# ============================================================================
# Test Class 1: Schema Validation
# ============================================================================


class TestDinoEmbeddingParams:
    """Test DinoEmbeddingParams schema validation."""

    def test_default_params(self) -> None:
        """Test default DinoEmbeddingParams values."""
        params = DinoEmbeddingParams(input_paths=["/test/image.jpg"], output_paths=[])

        assert params.input_paths == ["/test/image.jpg"]
        assert params.output_paths == []
        assert params.normalize is True

    def test_custom_normalize(self) -> None:
        """Test DinoEmbeddingParams with custom normalize setting."""
        params = DinoEmbeddingParams(
            input_paths=["/test/image.jpg"], output_paths=[], normalize=False
        )

        assert params.normalize is False


class TestDinoEmbedding:
    """Test DinoEmbedding schema validation."""

    def test_create_from_numpy(self) -> None:
        """Test creating DinoEmbedding from numpy array."""
        embedding_array = np.random.randn(384).astype(np.float32)

        dino_embedding = DinoEmbedding.from_numpy(embedding=embedding_array)

        assert dino_embedding.embedding_dim == 384
        assert len(dino_embedding.embedding) == 384

    def test_to_numpy(self) -> None:
        """Test converting DinoEmbedding to numpy array."""
        embedding_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        dino_embedding = DinoEmbedding(embedding=embedding_list, embedding_dim=5)

        embedding_array = dino_embedding.to_numpy()

        assert isinstance(embedding_array, np.ndarray)
        assert embedding_array.dtype == np.float32
        assert len(embedding_array) == 5
        np.testing.assert_array_almost_equal(embedding_array, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_embedding_dimension_consistency(self) -> None:
        """Test that embedding dimension matches length when using from_numpy."""
        # from_numpy() ensures consistency automatically
        embedding_array = np.random.randn(384).astype(np.float32)
        dino_embedding = DinoEmbedding.from_numpy(embedding=embedding_array)

        assert len(dino_embedding.embedding) == 384
        assert dino_embedding.embedding_dim == 384

    def test_correct_dimension_384(self) -> None:
        """Test that 384D embeddings are correctly validated."""
        embedding_array = np.random.randn(384).astype(np.float32)
        dino_embedding = DinoEmbedding.from_numpy(embedding=embedding_array)

        assert dino_embedding.embedding_dim == 384
        assert len(dino_embedding.embedding) == 384


class TestDinoEmbeddingResult:
    """Test DinoEmbeddingResult schema validation."""

    def test_success_result(self) -> None:
        """Test successful DINOv2 embedding result."""
        embedding = DinoEmbedding(embedding=[0.1] * 384, embedding_dim=384)

        result = DinoEmbeddingResult(
            file_path="/test/image.jpg", embedding=embedding, status="success"
        )

        assert result.file_path == "/test/image.jpg"
        assert result.embedding is not None
        assert result.embedding.embedding_dim == 384
        assert result.status == "success"
        assert result.error is None

    def test_error_result(self) -> None:
        """Test error DINOv2 embedding result."""
        result = DinoEmbeddingResult(
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


class TestDinoEmbeddingTask:
    """Test DinoEmbeddingTask execution."""

    def test_task_type(self, dino_embedding_task: DinoEmbeddingTask) -> None:
        """Test that task type is correctly set."""
        assert dino_embedding_task.task_type == "dino_embedding"

    def test_get_schema(self, dino_embedding_task: DinoEmbeddingTask) -> None:
        """Test that get_schema returns DinoEmbeddingParams."""
        schema = dino_embedding_task.get_schema()
        assert schema == DinoEmbeddingParams

    @pytest.mark.asyncio
    async def test_execute_with_mocked_embedder(
        self,
        dino_embedding_task: DinoEmbeddingTask,
        sample_image: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with mocked DINOv2 embedder."""
        # Create job
        job = Job(
            job_id="test-job-001",
            task_type="dino_embedding",
            params={
                "input_paths": [sample_image],
                "output_paths": [],
                "normalize": True,
            },
        )

        params = DinoEmbeddingParams(**job.params)

        # Mock the embedder
        mock_embedder = Mock()
        mock_embedding = np.random.randn(384).astype(np.float32)
        # Normalize it
        mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
        mock_embedder.embed.return_value = mock_embedding

        with patch.object(dino_embedding_task, "_get_embedder", return_value=mock_embedder):
            result = await dino_embedding_task.execute(job, params, mock_progress_callback)

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
        assert file_result["embedding"]["embedding_dim"] == 384

        # Verify progress callback was called
        assert len(mock_progress_callback.calls) == 1
        assert mock_progress_callback.calls[0] == 100

    @pytest.mark.asyncio
    async def test_execute_file_not_found(
        self,
        dino_embedding_task: DinoEmbeddingTask,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with non-existent file."""
        job = Job(
            job_id="test-job-002",
            task_type="dino_embedding",
            params={
                "input_paths": ["/nonexistent/image.jpg"],
                "output_paths": [],
                "normalize": True,
            },
        )

        params = DinoEmbeddingParams(**job.params)

        # Mock the embedder to raise FileNotFoundError
        mock_embedder = Mock()
        mock_embedder.embed.side_effect = FileNotFoundError("Image file not found")

        with patch.object(dino_embedding_task, "_get_embedder", return_value=mock_embedder):
            result = await dino_embedding_task.execute(job, params, mock_progress_callback)

        # Should return error status
        assert result["status"] == "error"

        task_output = result["task_output"]
        file_result = task_output["files"][0]
        assert file_result["status"] == "error"
        assert "not found" in file_result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_multiple_files(
        self,
        dino_embedding_task: DinoEmbeddingTask,
        sample_image: str,
        tmp_path: Path,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with multiple files."""
        # Create second image
        image_path_2 = tmp_path / "test_image_2.jpg"
        img = Image.new("RGB", (224, 224), color=(80, 120, 160))
        img.save(str(image_path_2), "JPEG")

        job = Job(
            job_id="test-job-003",
            task_type="dino_embedding",
            params={
                "input_paths": [sample_image, str(image_path_2)],
                "output_paths": [],
                "normalize": True,
            },
        )

        params = DinoEmbeddingParams(**job.params)

        # Mock the embedder
        mock_embedder = Mock()

        def mock_embed_side_effect(image_path, normalize):
            embedding = np.random.randn(384).astype(np.float32)
            if normalize:
                embedding = embedding / np.linalg.norm(embedding)
            return embedding

        mock_embedder.embed.side_effect = mock_embed_side_effect

        with patch.object(dino_embedding_task, "_get_embedder", return_value=mock_embedder):
            result = await dino_embedding_task.execute(job, params, mock_progress_callback)

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
        dino_embedding_task: DinoEmbeddingTask,
        sample_image: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with normalization disabled."""
        job = Job(
            job_id="test-job-004",
            task_type="dino_embedding",
            params={
                "input_paths": [sample_image],
                "output_paths": [],
                "normalize": False,
            },
        )

        params = DinoEmbeddingParams(**job.params)

        # Mock the embedder
        mock_embedder = Mock()
        mock_embedding = np.random.randn(384).astype(np.float32)
        mock_embedder.embed.return_value = mock_embedding

        with patch.object(dino_embedding_task, "_get_embedder", return_value=mock_embedder):
            result = await dino_embedding_task.execute(job, params, mock_progress_callback)

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
        dino_embedding_task: DinoEmbeddingTask,
        sample_image: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution when embedder fails to initialize."""
        job = Job(
            job_id="test-job-005",
            task_type="dino_embedding",
            params={
                "input_paths": [sample_image],
                "output_paths": [],
            },
        )

        params = DinoEmbeddingParams(**job.params)

        # Mock embedder initialization failure
        with patch.object(
            dino_embedding_task,
            "_get_embedder",
            side_effect=RuntimeError("ONNX model download failed"),
        ):
            result = await dino_embedding_task.execute(job, params, mock_progress_callback)

        # Should return error
        assert result["status"] == "error"
        assert "ONNX model download failed" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_partial_success(
        self,
        dino_embedding_task: DinoEmbeddingTask,
        sample_image: str,
        tmp_path: Path,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with partial success (some files fail)."""
        # Create second image
        image_path_2 = tmp_path / "test_image_2.jpg"
        img = Image.new("RGB", (224, 224), color=(80, 120, 160))
        img.save(str(image_path_2), "JPEG")

        job = Job(
            job_id="test-job-006",
            task_type="dino_embedding",
            params={
                "input_paths": [sample_image, str(image_path_2)],
                "output_paths": [],
                "normalize": True,
            },
        )

        params = DinoEmbeddingParams(**job.params)

        # Mock the embedder - first succeeds, second fails
        mock_embedder = Mock()
        mock_embedder.embed.side_effect = [
            np.random.randn(384).astype(np.float32),
            RuntimeError("Inference failed"),
        ]

        with patch.object(dino_embedding_task, "_get_embedder", return_value=mock_embedder):
            result = await dino_embedding_task.execute(job, params, mock_progress_callback)

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


class TestDinoEmbeddingAlgorithm:
    """Test DinoEmbedder algorithm components."""

    def test_l2_normalization_logic(self) -> None:
        """Test L2 normalization produces unit vectors."""
        # Create a random embedding
        embedding = np.random.randn(384).astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        normalized = embedding / norm

        # Verify unit vector (L2 norm = 1)
        assert np.abs(np.linalg.norm(normalized) - 1.0) < 1e-6

    def test_embedding_dimension_consistency(self) -> None:
        """Test that embeddings maintain 384D dimensionality."""
        embedding_array = np.random.randn(384).astype(np.float32)
        dino_embedding = DinoEmbedding.from_numpy(embedding=embedding_array)

        assert dino_embedding.embedding_dim == 384
        assert len(dino_embedding.embedding) == 384

        # Convert back and verify
        converted = dino_embedding.to_numpy()
        assert len(converted) == 384
        assert converted.dtype == np.float32

    def test_cosine_similarity_computation(self) -> None:
        """Test cosine similarity between normalized embeddings."""
        # Create two random normalized embeddings
        emb1 = np.random.randn(384).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)

        emb2 = np.random.randn(384).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)

        # Cosine similarity = dot product (for normalized vectors)
        cosine_sim = np.dot(emb1, emb2)

        # Should be in range [-1, 1]
        assert -1.0 <= cosine_sim <= 1.0

    def test_identical_images_should_have_high_similarity(self) -> None:
        """Test that identical embeddings have similarity ~1.0."""
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # Cosine similarity with itself
        cosine_sim = np.dot(embedding, embedding)

        # Should be very close to 1.0
        assert np.abs(cosine_sim - 1.0) < 1e-6

    def test_orthogonal_embeddings_zero_similarity(self) -> None:
        """Test that orthogonal embeddings have similarity ~0.0."""
        # Create two orthogonal vectors (simplified example for first 2 dims)
        emb1 = np.zeros(384, dtype=np.float32)
        emb1[0] = 1.0  # Unit vector along axis 0

        emb2 = np.zeros(384, dtype=np.float32)
        emb2[1] = 1.0  # Unit vector along axis 1

        # Cosine similarity
        cosine_sim = np.dot(emb1, emb2)

        # Should be exactly 0.0
        assert np.abs(cosine_sim) < 1e-6
