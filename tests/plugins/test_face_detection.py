"""Unit and integration tests for face detection plugin.

Tests schema validation, bounding box detection, confidence thresholds, task execution, routes, and full job lifecycle.
Requires ML models downloaded.
"""

import json
from pathlib import Path

import pytest

from cl_ml_tools.plugins.face_detection.schema import (
    BoundingBox,
    FaceDetectionOutput,
    FaceDetectionParams,
)
from cl_ml_tools.plugins.face_detection.task import FaceDetectionTask


# ============================================================================
# SCHEMA TESTS
# ============================================================================


def test_face_detection_params_schema_validation():
    """Test FaceDetectionParams schema validates correctly."""
    params = FaceDetectionParams(
        input_path="/path/to/input.jpg",
        output_path="output/faces.json",
        confidence_threshold=0.8,
        nms_threshold=0.5,
    )

    assert params.input_path == "/path/to/input.jpg"
    assert params.output_path == "output/faces.json"
    assert params.confidence_threshold == 0.8
    assert params.nms_threshold == 0.5


def test_face_detection_params_defaults():
    """Test FaceDetectionParams has correct default values."""
    params = FaceDetectionParams(
        input_path="/path/to/input.jpg",
        output_path="output/faces.json",
    )

    assert params.confidence_threshold == 0.7
    assert params.nms_threshold == 0.4


def test_face_detection_params_threshold_validation():
    """Test FaceDetectionParams validates threshold ranges."""
    # Valid thresholds
    params = FaceDetectionParams(
        input_path="/path/to/input.jpg",
        output_path="output/faces.json",
        confidence_threshold=0.5,
        nms_threshold=0.3,
    )
    assert params.confidence_threshold == 0.5

    # Invalid confidence (too high)
    with pytest.raises(ValueError):
        _ = FaceDetectionParams(
            input_path="/path/to/input.jpg",
            output_path="output/faces.json",
            confidence_threshold=1.5,
        )

    # Invalid nms (negative)
    with pytest.raises(ValueError):
        _ = FaceDetectionParams(
            input_path="/path/to/input.jpg",
            output_path="output/faces.json",
            nms_threshold=-0.1,
        )


def test_bounding_box_schema_validation():
    """Test BoundingBox schema validates correctly."""
    bbox = BoundingBox(
        x1=0.1,
        y1=0.2,
        x2=0.8,
        y2=0.9,
        confidence=0.95,
    )

    assert bbox.x1 == 0.1
    assert bbox.y1 == 0.2
    assert bbox.x2 == 0.8
    assert bbox.y2 == 0.9
    assert bbox.confidence == 0.95


def test_bounding_box_to_absolute():
    """Test BoundingBox.to_absolute converts coordinates."""
    bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9, confidence=0.9)

    absolute = bbox.to_absolute(image_width=1000, image_height=800)

    assert absolute["x1"] == 100
    assert absolute["y1"] == 160
    assert absolute["x2"] == 800
    assert absolute["y2"] == 720


def test_face_detection_output_schema_validation():
    """Test FaceDetectionOutput schema validates correctly."""
    bbox = BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5, confidence=0.9)

    output = FaceDetectionOutput(
        faces=[bbox],
        num_faces=1,
        image_width=800,
        image_height=600,
    )

    assert len(output.faces) == 1
    assert output.num_faces == 1
    assert output.image_width == 800
    assert output.image_height == 600


# ============================================================================
# ALGORITHM TESTS
# ============================================================================


@pytest.mark.requires_models
def test_face_detection_algo_basic(sample_image_path: Path):
    """Test basic face detection."""
    from cl_ml_tools.plugins.face_detection.algo.face_detector import FaceDetector

    detector = FaceDetector()
    faces = detector.detect(str(sample_image_path), confidence_threshold=0.5)

    assert isinstance(faces, list)
    # May or may not contain faces depending on test image


@pytest.mark.requires_models
def test_face_detection_algo_returns_bounding_boxes(sample_image_path: Path):
    """Test face detection returns proper bounding box format."""
    from cl_ml_tools.plugins.face_detection.algo.face_detector import FaceDetector

    detector = FaceDetector()
    faces = detector.detect(str(sample_image_path), confidence_threshold=0.3)

    for face in faces:
        # Each face should have bbox coordinates and confidence
        assert "x1" in face or "bbox" in face or hasattr(face, "x1")


@pytest.mark.requires_models
def test_face_detection_algo_confidence_threshold(sample_image_path: Path):
    """Test confidence threshold filtering."""
    from cl_ml_tools.plugins.face_detection.algo.face_detector import FaceDetector

    detector = FaceDetector()

    # Low threshold - more detections
    faces_low = detector.detect(str(sample_image_path), confidence_threshold=0.3)

    # High threshold - fewer detections
    faces_high = detector.detect(str(sample_image_path), confidence_threshold=0.9)

    # High threshold should have same or fewer detections
    assert len(faces_high) <= len(faces_low)


@pytest.mark.requires_models
def test_face_detection_algo_error_handling_invalid_file(tmp_path: Path):
    """Test face detection handles invalid image files."""
    from cl_ml_tools.plugins.face_detection.algo.face_detector import FaceDetector

    invalid_file = tmp_path / "invalid.jpg"
    invalid_file.write_text("not an image")

    detector = FaceDetector()

    with pytest.raises(Exception):
        detector.detect(str(invalid_file), confidence_threshold=0.5)


# ============================================================================
# TASK TESTS
# ============================================================================


@pytest.mark.requires_models
@pytest.mark.asyncio
async def test_face_detection_task_run_success(sample_image_path: Path, tmp_path: Path):
    """Test FaceDetectionTask execution success."""
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(sample_image_path.read_bytes())

    params = FaceDetectionParams(
        input_path=str(input_path),
        output_path="output/faces.json",
        confidence_threshold=0.5,
    )

    task = FaceDetectionTask()
    task.setup()
    job_id = "test-job-123"

    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / job_id / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            output_path = tmp_path / "output" / "faces.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return str(output_path)

    storage = MockStorage()

    output = await task.run(job_id, params, storage)

    assert isinstance(output, FaceDetectionOutput)
    assert output.num_faces >= 0
    assert output.image_width > 0
    assert output.image_height > 0

    # Verify output file
    output_file = tmp_path / "output" / "faces.json"
    assert output_file.exists()

    with open(output_file) as f:
        result = json.load(f)

    assert "num_faces" in result
    assert "faces" in result


@pytest.mark.requires_models
@pytest.mark.asyncio
async def test_face_detection_task_run_file_not_found(tmp_path: Path):
    """Test FaceDetectionTask raises FileNotFoundError for missing input."""
    params = FaceDetectionParams(
        input_path="/nonexistent/file.jpg",
        output_path="output/faces.json",
    )

    task = FaceDetectionTask()
    task.setup()
    job_id = "test-job-789"

    class MockStorage:
        def resolve_path(self, job_id: str, relative_path: str) -> Path:
            return tmp_path / job_id / relative_path

        def allocate_path(self, job_id: str, relative_path: str) -> str:
            return str(tmp_path / "output" / "faces.json")

    storage = MockStorage()

    with pytest.raises(FileNotFoundError):
        await task.run(job_id, params, storage)


# ============================================================================
# ROUTE TESTS
# ============================================================================


def test_face_detection_route_creation(api_client):
    """Test face_detection route is registered."""
    response = api_client.get("/openapi.json")
    assert response.status_code == 200

    openapi = response.json()
    assert "/jobs/face_detection" in openapi["paths"]


@pytest.mark.requires_models
def test_face_detection_route_job_submission(api_client, sample_image_path: Path):
    """Test job submission via face_detection route."""
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/face_detection",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"confidence_threshold": 0.7, "priority": 5},
        )

    assert response.status_code == 200
    data = response.json()

    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["task_type"] == "face_detection"


# ============================================================================
# INTEGRATION TEST (API → Worker)
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_models
async def test_face_detection_full_job_lifecycle(
    api_client, worker, job_repository, sample_image_path: Path, file_storage
):
    """Test complete flow: API → Repository → Worker → Output."""
    # 1. Submit job via API
    with open(sample_image_path, "rb") as f:
        response = api_client.post(
            "/jobs/face_detection",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"confidence_threshold": 0.5, "priority": 5},
        )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    # 2. Worker consumes job
    processed = await worker.run_once()
    assert processed == 1

    # 3. Verify completion
    job = job_repository.get(job_id)
    assert job is not None
    assert job.status == "completed"

    # 4. Validate output
    assert job.output is not None
    output = FaceDetectionOutput.model_validate(job.output)
    assert output.num_faces >= 0
    assert len(output.faces) == output.num_faces
