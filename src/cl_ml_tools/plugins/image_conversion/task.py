"""Image conversion task implementation."""

from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.image_convert import image_convert
from .schema import ImageConversionParams


class ImageConversionTask(ComputeModule[ImageConversionParams]):
    """Compute module for converting an image between formats."""

    @property
    @override
    def task_type(self) -> str:
        return "image_conversion"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return ImageConversionParams

    @override
    async def execute(
        self,
        job: Job,
        params: ImageConversionParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Convert an image and store it on disk."""

        try:
            _ = image_convert(
                input_path=params.input_path,
                output_path=params.output_path,
                format=params.format,
                quality=params.quality,
            )

            if progress_callback:
                progress_callback(100)

            return TaskResult(
                status="ok",
                task_output={},
            )

        except ImportError:
            return TaskResult(
                status="error",
                error=("Pillow is not installed. Install with: pip install cl_ml_tools[compute]"),
            )

        except FileNotFoundError as exc:
            return TaskResult(
                status="error",
                error=f"Input file not found: {exc}",
            )

        except Exception as exc:  # noqa: BLE001
            return TaskResult(
                status="error",
                error=str(exc),
            )
