"""Face embedding route factory."""

from typing import Annotated, Callable, Literal, Protocol, TypedDict
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...common.file_storage import FileStorage
from ...common.job_repository import JobRepository
from ...common.schemas import Job


class UserLike(Protocol):
    id: str | None


class JobCreatedResponse(TypedDict):
    job_id: str
    status: Literal["queued"]


def create_router(
    repository: JobRepository,
    file_storage: FileStorage,
    get_current_user: Callable[[], UserLike | None],
) -> APIRouter:
    """Create FastAPI router for face embedding endpoints.

    Args:
        repository: Job repository for persistence
        file_storage: File storage for managing uploaded files
        get_current_user: Dependency for getting current user

    Returns:
        Configured APIRouter with face embedding endpoint
    """
    router = APIRouter()

    @router.post("/jobs/face_embedding", response_model=JobCreatedResponse)
    async def create_face_embedding_job(
        file: Annotated[UploadFile, File(description="Cropped face image file")],
        normalize: Annotated[
            bool, Form(description="Whether to L2-normalize the embedding")
        ] = True,
        priority: Annotated[int, Form(ge=0, le=10, description="Job priority (0-10)")] = 5,
        user: Annotated[UserLike | None, Depends(get_current_user)] = None,
    ) -> JobCreatedResponse:
        """Create a new face embedding job.

        Args:
            file: Uploaded face image file (should be cropped to face)
            normalize: Whether to L2-normalize the embedding vector
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

        # Create job (no output_paths needed for face embedding)
        job = Job(
            job_id=job_id,
            task_type="face_embedding",
            params={
                "input_paths": [input_path],
                "output_paths": [],  # Face embedding doesn't produce output files
                "normalize": normalize,
            },
        )

        created_by = user.id if user is not None else None
        _ = repository.add_job(job, created_by=created_by, priority=priority)

        return {
            "job_id": job_id,
            "status": "queued",
        }

    # Mark function as used (accessed via FastAPI decorator)
    _ = create_face_embedding_job

    return router
