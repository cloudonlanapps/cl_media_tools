"""ComputeModule - Abstract base class for compute tasks."""

from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

from .schemas import BaseJobParams, Job, TaskResult

P = TypeVar("P", bound=BaseJobParams)


class ComputeModule(ABC, Generic[P]):
    """Abstract base class for all compute tasks.

    All task plugins must extend this class and implement the required methods.

    Example:
        class ImageThumbnailTask(ComputeModule):

            @property
            def task_type(self) -> str:
                return "image_thumbnail"

            def get_schema(self) -> type[BaseJobParams]:
                return ImageThumbnailParams

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
        params: P,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Execute the task.

        Args:
            job: The Job object (job_id, task_type, etc.)
            params: Validated parameters (subclass of BaseJobParams)
            progress_callback: Optional callback to report progress (0–100)

        Returns:
            TaskResult dict with status and optional task_output/error
        """
        ...
