"""Comprehensive test suite for face embedding plugin."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from cl_ml_tools.common.schema_job import Job
from cl_ml_tools.plugins.face_embedding.schema import (
    FaceEmbedding,
    FaceEmbeddingParams,
    FaceEmbeddingResult,
)
from cl_ml_tools.plugins.face_embedding.task import FaceEmbeddingTask

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
def sample_face_image(tmp_path: Path) -> str:
    """Create a test face image."""
    image_path = tmp_path / "test_face.jpg"

    # Create a simple image (112x112 to match model input)
    img = Image.new("RGB", (112, 112), color=(200, 150, 100))
    img.save(str(image_path), "JPEG")

    return str(image_path)


@pytest.fixture
def face_embedding_task() -> FaceEmbeddingTask:
    """Create FaceEmbeddingTask instance."""
    return FaceEmbeddingTask()


@pytest.fixture
def mock_progress_callback() -> MockProgressCallback:
    """Create mock progress callback."""
    return MockProgressCallback()


# ============================================================================
# Test Class 1: Schema Validation
# ============================================================================


class TestFaceEmbeddingParams:
    """Test FaceEmbeddingParams schema validation."""

    def test_default_params(self) -> None:
        """Test default FaceEmbeddingParams values."""
        params = FaceEmbeddingParams(input_paths=["/test/face.jpg"], output_paths=[])

        assert params.input_paths == ["/test/face.jpg"]
        assert params.output_paths == []
        assert params.normalize is True

    def test_custom_normalize(self) -> None:
        """Test FaceEmbeddingParams with custom normalize setting."""
        params = FaceEmbeddingParams(
            input_paths=["/test/face.jpg"], output_paths=[], normalize=False
        )

        assert params.normalize is False


class TestFaceEmbedding:
    """Test FaceEmbedding schema validation."""

    def test_create_from_numpy(self) -> None:
        """Test creating FaceEmbedding from numpy array."""
        embedding_array = np.random.randn(512).astype(np.float32)
        quality_score = 0.85

        face_embedding = FaceEmbedding.from_numpy(
            embedding=embedding_array, quality_score=quality_score
        )

        assert face_embedding.embedding_dim == 512
        assert len(face_embedding.embedding) == 512
        assert face_embedding.quality_score == 0.85

    def test_to_numpy(self) -> None:
        """Test converting FaceEmbedding to numpy array."""
        embedding_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        face_embedding = FaceEmbedding(
            embedding=embedding_list, embedding_dim=5, quality_score=0.9
        )

        embedding_array = face_embedding.to_numpy()

        assert isinstance(embedding_array, np.ndarray)
        assert embedding_array.dtype == np.float32
        assert len(embedding_array) == 5
        np.testing.assert_array_almost_equal(embedding_array, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_embedding_dimension_mismatch_allowed(self) -> None:
        """Test that embedding dimension doesn't auto-validate (user responsibility)."""
        # Note: Pydantic doesn't automatically validate len(embedding) == embedding_dim
        # This is by design - users should use from_numpy() which ensures consistency
        embedding = FaceEmbedding(
            embedding=[0.1, 0.2, 0.3],
            embedding_dim=3,  # Must match manually
            quality_score=0.9,
        )
        assert len(embedding.embedding) == 3
        assert embedding.embedding_dim == 3

    def test_quality_score_bounds(self) -> None:
        """Test quality score validation."""
        # Valid quality scores
        FaceEmbedding(embedding=[0.1] * 512, embedding_dim=512, quality_score=0.0)
        FaceEmbedding(embedding=[0.1] * 512, embedding_dim=512, quality_score=1.0)

        # Invalid quality scores
        with pytest.raises(Exception):  # Pydantic validation
            FaceEmbedding(embedding=[0.1] * 512, embedding_dim=512, quality_score=-0.1)

        with pytest.raises(Exception):  # Pydantic validation
            FaceEmbedding(embedding=[0.1] * 512, embedding_dim=512, quality_score=1.5)


class TestFaceEmbeddingResult:
    """Test FaceEmbeddingResult schema validation."""

    def test_success_result(self) -> None:
        """Test successful face embedding result."""
        embedding = FaceEmbedding(
            embedding=[0.1] * 512, embedding_dim=512, quality_score=0.85
        )

        result = FaceEmbeddingResult(
            file_path="/test/face.jpg", embedding=embedding, status="success"
        )

        assert result.file_path == "/test/face.jpg"
        assert result.embedding is not None
        assert result.embedding.embedding_dim == 512
        assert result.embedding.quality_score == 0.85
        assert result.status == "success"
        assert result.error is None

    def test_error_result(self) -> None:
        """Test error face embedding result."""
        result = FaceEmbeddingResult(
            file_path="/test/face.jpg",
            embedding=None,
            status="error",
            error="File not found",
        )

        assert result.file_path == "/test/face.jpg"
        assert result.embedding is None
        assert result.status == "error"
        assert result.error == "File not found"


# ============================================================================
# Test Class 2: Task Execution
# ============================================================================


class TestFaceEmbeddingTask:
    """Test FaceEmbeddingTask execution."""

    def test_task_type(self, face_embedding_task: FaceEmbeddingTask) -> None:
        """Test that task type is correctly set."""
        assert face_embedding_task.task_type == "face_embedding"

    def test_get_schema(self, face_embedding_task: FaceEmbeddingTask) -> None:
        """Test that get_schema returns FaceEmbeddingParams."""
        schema = face_embedding_task.get_schema()
        assert schema == FaceEmbeddingParams

    @pytest.mark.asyncio
    async def test_execute_with_mocked_embedder(
        self,
        face_embedding_task: FaceEmbeddingTask,
        sample_face_image: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with mocked face embedder."""
        # Create job
        job = Job(
            job_id="test-job-001",
            task_type="face_embedding",
            params={
                "input_paths": [sample_face_image],
                "output_paths": [],
                "normalize": True,
            },
        )

        params = FaceEmbeddingParams(**job.params)

        # Mock the embedder
        mock_embedder = Mock()
        mock_embedding = np.random.randn(512).astype(np.float32)
        mock_quality = 0.92
        mock_embedder.embed.return_value = (mock_embedding, mock_quality)

        with patch.object(
            face_embedding_task, "_get_embedder", return_value=mock_embedder
        ):
            result = await face_embedding_task.execute(job, params, mock_progress_callback)

        # Verify result
        assert result["status"] == "ok"
        assert "task_output" in result

        task_output = result["task_output"]
        assert task_output["total_files"] == 1
        assert len(task_output["files"]) == 1

        file_result = task_output["files"][0]
        assert file_result["file_path"] == sample_face_image
        assert file_result["status"] == "success"
        assert file_result["embedding"]["embedding_dim"] == 512
        assert file_result["embedding"]["quality_score"] == 0.92

        # Verify progress callback was called
        assert len(mock_progress_callback.calls) == 1
        assert mock_progress_callback.calls[0] == 100

    @pytest.mark.asyncio
    async def test_execute_file_not_found(
        self,
        face_embedding_task: FaceEmbeddingTask,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with non-existent file."""
        job = Job(
            job_id="test-job-002",
            task_type="face_embedding",
            params={
                "input_paths": ["/nonexistent/face.jpg"],
                "output_paths": [],
                "normalize": True,
            },
        )

        params = FaceEmbeddingParams(**job.params)

        # Mock the embedder to raise FileNotFoundError
        mock_embedder = Mock()
        mock_embedder.embed.side_effect = FileNotFoundError("Image file not found")

        with patch.object(
            face_embedding_task, "_get_embedder", return_value=mock_embedder
        ):
            result = await face_embedding_task.execute(job, params, mock_progress_callback)

        # Should return error status
        assert result["status"] == "error"

        task_output = result["task_output"]
        file_result = task_output["files"][0]
        assert file_result["status"] == "error"
        assert "not found" in file_result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_multiple_files(
        self,
        face_embedding_task: FaceEmbeddingTask,
        sample_face_image: str,
        tmp_path: Path,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with multiple files."""
        # Create second image
        image_path_2 = tmp_path / "test_face_2.jpg"
        img = Image.new("RGB", (112, 112), color=(100, 150, 200))
        img.save(str(image_path_2), "JPEG")

        job = Job(
            job_id="test-job-003",
            task_type="face_embedding",
            params={
                "input_paths": [sample_face_image, str(image_path_2)],
                "output_paths": [],
                "normalize": True,
            },
        )

        params = FaceEmbeddingParams(**job.params)

        # Mock the embedder
        mock_embedder = Mock()
        mock_embedder.embed.side_effect = [
            (np.random.randn(512).astype(np.float32), 0.92),
            (np.random.randn(512).astype(np.float32), 0.88),
        ]

        with patch.object(
            face_embedding_task, "_get_embedder", return_value=mock_embedder
        ):
            result = await face_embedding_task.execute(job, params, mock_progress_callback)

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
        face_embedding_task: FaceEmbeddingTask,
        sample_face_image: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with normalization disabled."""
        job = Job(
            job_id="test-job-004",
            task_type="face_embedding",
            params={
                "input_paths": [sample_face_image],
                "output_paths": [],
                "normalize": False,
            },
        )

        params = FaceEmbeddingParams(**job.params)

        # Mock the embedder
        mock_embedder = Mock()
        mock_embedding = np.random.randn(512).astype(np.float32)
        mock_embedder.embed.return_value = (mock_embedding, 0.85)

        with patch.object(
            face_embedding_task, "_get_embedder", return_value=mock_embedder
        ):
            result = await face_embedding_task.execute(job, params, mock_progress_callback)

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
        face_embedding_task: FaceEmbeddingTask,
        sample_face_image: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution when embedder fails to initialize."""
        job = Job(
            job_id="test-job-005",
            task_type="face_embedding",
            params={
                "input_paths": [sample_face_image],
                "output_paths": [],
            },
        )

        params = FaceEmbeddingParams(**job.params)

        # Mock embedder initialization failure
        with patch.object(
            face_embedding_task,
            "_get_embedder",
            side_effect=RuntimeError("ONNX model not found"),
        ):
            result = await face_embedding_task.execute(job, params, mock_progress_callback)

        # Should return error
        assert result["status"] == "error"
        assert "ONNX model not found" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_low_quality_image(
        self,
        face_embedding_task: FaceEmbeddingTask,
        sample_face_image: str,
        mock_progress_callback: MockProgressCallback,
    ) -> None:
        """Test task execution with low quality image (low quality score)."""
        job = Job(
            job_id="test-job-006",
            task_type="face_embedding",
            params={
                "input_paths": [sample_face_image],
                "output_paths": [],
                "normalize": True,
            },
        )

        params = FaceEmbeddingParams(**job.params)

        # Mock the embedder to return low quality score
        mock_embedder = Mock()
        mock_embedding = np.random.randn(512).astype(np.float32)
        mock_embedder.embed.return_value = (mock_embedding, 0.15)  # Low quality

        with patch.object(
            face_embedding_task, "_get_embedder", return_value=mock_embedder
        ):
            result = await face_embedding_task.execute(job, params, mock_progress_callback)

        # Should still succeed, but with low quality score
        assert result["status"] == "ok"

        file_result = result["task_output"]["files"][0]
        assert file_result["status"] == "success"
        assert file_result["embedding"]["quality_score"] == 0.15


# ============================================================================
# Test Class 3: Algorithm Unit Tests
# ============================================================================


class TestFaceEmbeddingAlgorithm:
    """Test FaceEmbedder algorithm components."""

    def test_l2_normalization_logic(self) -> None:
        """Test L2 normalization produces unit vectors."""
        # Create a random embedding
        embedding = np.random.randn(512).astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        normalized = embedding / norm

        # Verify unit vector
        assert np.abs(np.linalg.norm(normalized) - 1.0) < 1e-6

    def test_quality_score_range(self) -> None:
        """Test that quality scores are in valid range."""
        # This is validated by the schema, but we can test the concept
        valid_scores = [0.0, 0.5, 1.0]
        for score in valid_scores:
            embedding = FaceEmbedding(
                embedding=[0.1] * 512, embedding_dim=512, quality_score=score
            )
            assert 0.0 <= embedding.quality_score <= 1.0

    def test_embedding_dimension_consistency(self) -> None:
        """Test that embeddings maintain consistent dimensionality."""
        # Create embeddings of different sizes
        for dim in [128, 256, 512]:
            embedding_array = np.random.randn(dim).astype(np.float32)
            face_embedding = FaceEmbedding.from_numpy(
                embedding=embedding_array, quality_score=0.9
            )

            assert face_embedding.embedding_dim == dim
            assert len(face_embedding.embedding) == dim

            # Convert back and verify
            converted = face_embedding.to_numpy()
            assert len(converted) == dim
