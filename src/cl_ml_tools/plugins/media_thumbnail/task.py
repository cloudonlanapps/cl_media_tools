"""Media thumbnail task implementation."""

from io import BytesIO
from pathlib import Path
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from ...utils.media_types import MediaType, determine_mime
from .algo.image_thumbnail import image_thumbnail
from .algo.video_thumbnail import video_thumbnail
from .schema import MediaThumbnailParams


class MediaThumbnailTask(ComputeModule[MediaThumbnailParams]):
    """Compute module for generating a thumbnail for an image or video."""

    @property
    @override
    def task_type(self) -> str:
        return "media_thumbnail"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return MediaThumbnailParams

    @override
    async def execute(
        self,
        job: Job,
        params: MediaThumbnailParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Generate a thumbnail and store it on disk."""

        input_path = Path(params.input_path)

        # Phase 1: validate input
        if not input_path.exists():
            return TaskResult(
                status="error",
                error=f"Input file not found: {params.input_path}",
            )

        # Phase 2: detect media type
        try:
            with input_path.open("rb") as f:
                media_type = determine_mime(BytesIO(f.read()))
        except ImportError as exc:
            return TaskResult(
                status="error",
                error=f"Required library not installed: {exc}",
            )

        # Phase 3: generate thumbnail
        try:
            if media_type == MediaType.IMAGE:
                image_thumbnail(
                    input_path=params.input_path,
                    output_path=params.output_path,
                    width=params.width,
                    height=params.height,
                    maintain_aspect_ratio=params.maintain_aspect_ratio,
                )
                media_type_str = "image"

            elif media_type == MediaType.VIDEO:
                video_thumbnail(
                    input_path=params.input_path,
                    output_path=params.output_path,
                    width=params.width,
                    height=params.height,
                )
                media_type_str = "video"

            else:
                return TaskResult(
                    status="error",
                    error=(
                        f"Unsupported media type: {media_type}. Only image and video are supported."
                    ),
                )

            if progress_callback:
                progress_callback(100)

            return TaskResult(
                status="ok",
                task_output={
                    "media_type": media_type_str,
                },
            )

        except Exception as exc:
            return TaskResult(
                status="error",
                error=str(exc),
            )
