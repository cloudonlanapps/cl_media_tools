"""ComputeModule - Abstract base class for compute tasks."""

from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

from .schemas import BaseJobParams, Job

R = TypeVar("R")


class ComputeModule(ABC, Generic[R]):
    """Abstract base class for all compute tasks.

    All task plugins must extend this class and implement the required methods.

    Example:
        class ImageResizeTask(ComputeModule[ImageResizeResult]):

            @property
            def task_type(self) -> str:
                return "image_resize"

            def get_schema(self) -> type[BaseJobParams]:
                return ImageResizeParams

            async def execute(self, job, params, progress_callback=None):
                return {
                    "status": "ok",
                    "task_output": {...},
                }
    """

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return task type identifier."""
        ...

    @abstractmethod
    def get_schema(self) -> type[BaseJobParams]:
        """Return the Pydantic params class for this task."""
        ...

    @abstractmethod
    async def execute(
        self,
        job: Job,
        params: BaseJobParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> R:
        """Execute the task.

        Args:
            job: The Job object (job_id, task_type, etc.)
            params: Validated parameters (subclass of BaseJobParams)
            progress_callback: Optional callback to report progress (0–100)

        Returns:
            Task-specific result type `R`
        """
        ...
