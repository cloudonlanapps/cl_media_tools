"""cl_ml_tools - Tools for master-worker media processing / ML ."""

from cl_ml_tools.common.schemas import Job, BaseJobParams
from cl_ml_tools.common.compute_module import ComputeModule
from cl_ml_tools.common.job_repository import JobRepository
from cl_ml_tools.common.file_storage import FileStorage
from cl_ml_tools.worker import Worker
from .utils.mqtt import (
    get_broadcaster,
    shutdown_broadcaster,
    MQTTBroadcaster,
    NoOpBroadcaster,
)
from cl_ml_tools.master import create_master_router

__version__ = "0.1.0"

__all__ = [
    "Job",
    "BaseJobParams",
    "ComputeModule",
    "JobRepository",
    "FileStorage",
    "__version__",
    "Worker",
    "MQTTBroadcaster",
    "NoOpBroadcaster",
    "create_master_router",
    "get_broadcaster",
    "shutdown_broadcaster",
]
