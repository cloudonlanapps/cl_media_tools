"""Hash computation route factory."""

from typing import Annotated, Callable, Protocol
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...common.file_storage import FileStorage
from ...common.job_repository import JobRepository
from ...common.schemas import Job


class UserLike(Protocol):
    """Protocol for user objects returned by authentication."""

    id: str | None


def create_router(
    repository: JobRepository,
    file_storage: FileStorage,
    get_current_user: Callable[[], UserLike | None],
) -> APIRouter:
    """Create router with injected dependencies.

    Args:
        repository: JobRepository implementation
        file_storage: FileStorage implementation
        get_current_user: Callable that returns current user (for auth)

    Returns:
        Configured APIRouter with hash computation endpoint
    """
    router = APIRouter()

    @router.post("/jobs/hash")
    async def create_hash_job(
        file: Annotated[UploadFile, File(description="File to hash")],
        algorithm: Annotated[str, Form(description="Hash algorithm (sha512 or md5)")] = "sha512",
        priority: Annotated[int, Form(ge=0, le=10, description="Job priority (0-10)")] = 5,
        user: Annotated[UserLike | None, Depends(get_current_user)] = None,
    ):
        """Create a hash computation job.

        Upload a file and specify hash algorithm. The job will be queued
        for processing by a worker.

        Returns:
            job_id: Unique identifier for the created job
            status: Initial job status ("queued")
        """
        job_id = str(uuid4())

        if not file.filename:
            raise ValueError("Uploaded file has no filename")

        filename: str = file.filename

        # Create job directory and save uploaded file
        _ = file_storage.create_job_directory(job_id)
        file_info = await file_storage.save_input_file(job_id, filename, file)

        # Get input path (no output paths for compute-only)
        input_path = file_info["path"]

        # Create job
        job = Job(
            job_id=job_id,
            task_type="hash",
            params={
                "input_paths": [input_path],
                "output_paths": [],  # Compute only, no output files
                "algorithm": algorithm,
            },
        )

        # Save to repository
        created_by = user.id if user is not None else None
        _ = repository.add_job(job, created_by=created_by, priority=priority)

        return {"job_id": job_id, "status": "queued"}

    # Mark function as used (accessed via FastAPI decorator)
    _ = create_hash_job

    return router
