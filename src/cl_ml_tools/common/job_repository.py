"""JobRepository Protocol - interface for job persistence."""

from collections.abc import Sequence
from typing import Protocol, TypedDict, runtime_checkable

from .schemas import Job

# ─────────────────────────────────────────────────────────────
# Typed update payload (avoids **kwargs: Any)
# ─────────────────────────────────────────────────────────────


class JobUpdate(TypedDict, total=False):
    status: str  # "queued" | "processing" | "completed" | "error"
    progress: int  # 0–100
    task_output: dict[str, object]
    error_message: str
    priority: int


# ─────────────────────────────────────────────────────────────
# JobRepository protocol
# ─────────────────────────────────────────────────────────────


@runtime_checkable
class JobRepository(Protocol):
    """Protocol for job persistence operations.

    Applications must implement this protocol to provide job storage
    functionality. Implementations must ensure atomic job claiming.
    """

    def add_job(
        self,
        job: Job,
        created_by: str | None = None,
        priority: int | None = None,
    ) -> str:
        """Save job to database.

        Returns:
            The job_id of the saved job
        """
        ...

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        ...

    def update_job(
        self,
        job_id: str,
        updates: JobUpdate,
    ) -> bool:
        """Update job fields.

        Args:
            job_id: Unique job identifier
            updates: Typed job update payload

        Returns:
            True if job was updated, False if job not found
        """
        ...

    def fetch_next_job(
        self,
        task_types: Sequence[str],
    ) -> Job | None:
        """Atomically find and claim the next queued job.

        Implementations MUST ensure this operation is atomic.
        """
        ...

    def delete_job(self, job_id: str) -> bool:
        """Delete job from database."""
        ...
