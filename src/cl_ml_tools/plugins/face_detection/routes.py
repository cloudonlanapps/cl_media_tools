"""Face detection route factory."""

from typing import Annotated, Callable, Literal, Protocol, TypedDict
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...common.file_storage import JobStorage
from ...common.job_repository import JobRepository
from ...common.schemas import Job


class UserLike(Protocol):
    id: str | None


class JobCreatedResponse(TypedDict):
    job_id: str
    status: Literal["queued"]


def create_router(
    repository: JobRepository,
    file_storage: JobStorage,
    get_current_user: Callable[[], UserLike | None],
) -> APIRouter:
    """Create FastAPI router for face detection endpoints.

    Args:
        repository: Job repository for persistence
        file_storage: File storage for managing uploaded files
        get_current_user: Dependency for getting current user

    Returns:
        Configured APIRouter with face detection endpoint
    """
    router = APIRouter()

    @router.post("/jobs/face_detection", response_model=JobCreatedResponse)
    async def create_face_detection_job(
        file: Annotated[UploadFile, File(description="Image file to detect faces in")],
        confidence_threshold: Annotated[
            float, Form(ge=0.0, le=1.0, description="Minimum confidence threshold (0.0-1.0)")
        ] = 0.7,
        nms_threshold: Annotated[
            float, Form(ge=0.0, le=1.0, description="NMS threshold for overlapping boxes (0.0-1.0)")
        ] = 0.4,
        priority: Annotated[int, Form(ge=0, le=10, description="Job priority (0-10)")] = 5,
        user: Annotated[UserLike | None, Depends(get_current_user)] = None,
    ) -> JobCreatedResponse:
        """Create a new face detection job.

        Args:
            file: Uploaded image file
            confidence_threshold: Minimum confidence for face detections
            nms_threshold: Non-maximum suppression threshold
            priority: Job priority (0-10, higher is more priority)
            user: Current user (injected by dependency)

        Returns:
            JobCreatedResponse with job_id and status
        """
        job_id = str(uuid4())

        # Create job directory
        _ = file_storage.create_job_directory(job_id)

        if not file.filename:
            raise ValueError("Uploaded file has no filename")

        filename: str = file.filename

        # Save uploaded file
        file_info = await file_storage.save_input_file(job_id, filename, file)
        input_path = file_info["path"]

        # Create job (no output_paths needed for face detection)
        job = Job(
            job_id=job_id,
            task_type="face_detection",
            params={
                "input_paths": [input_path],
                "output_paths": [],  # Face detection doesn't produce output files
                "confidence_threshold": confidence_threshold,
                "nms_threshold": nms_threshold,
            },
        )

        created_by = user.id if user is not None else None
        _ = repository.add_job(job, created_by=created_by, priority=priority)

        return {
            "job_id": job_id,
            "status": "queued",
        }

    # Mark function as used (accessed via FastAPI decorator)
    _ = create_face_detection_job

    return router
