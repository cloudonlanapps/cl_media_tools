"""Image conversion route factory."""

from pathlib import Path
from typing import Callable, Literal, Protocol, TypedDict
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
    router = APIRouter()

    # pyright: reportCallInDefaultInitializer=false
    @router.post("/jobs/image_conversion", response_model=JobCreatedResponse)
    async def create_conversion_job(  # pyright: reportUnusedFunction=false
        file: UploadFile = File(..., description="Image file to convert"),
        format: Literal["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"] = Form(
            ..., description="Target format"
        ),
        quality: int = Form(85, ge=1, le=100, description="Output quality (1-100)"),
        priority: int = Form(5, ge=0, le=10, description="Job priority (0-10)"),
        user: UserLike | None = Depends(get_current_user),
    ) -> JobCreatedResponse:
        job_id = str(uuid4())

        _ = file_storage.create_job_directory(job_id)

        if not file.filename:
            raise ValueError("Uploaded file has no filename")

        filename: str = file.filename

        file_info = await file_storage.save_input_file(job_id, filename, file)

        input_path = file_info["path"]
        original_stem = Path(filename).stem
        output_ext = "jpg" if format == "jpeg" else format
        output_filename = f"converted_{original_stem}.{output_ext}"
        output_path = str(file_storage.get_output_path(job_id) / output_filename)

        job = Job(
            job_id=job_id,
            task_type="image_conversion",
            params={
                "input_paths": [input_path],
                "output_paths": [output_path],
                "format": format,
                "quality": quality,
            },
        )

        created_by = user.id if user is not None else None
        _ = repository.add_job(job, created_by=created_by, priority=priority)

        return {
            "job_id": job_id,
            "status": "queued",
        }

    return router
