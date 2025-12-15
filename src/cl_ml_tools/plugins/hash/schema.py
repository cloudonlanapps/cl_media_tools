"""Hash computation parameters schema."""

from typing import Literal

from pydantic import Field

from ...common.schemas import BaseJobParams


class HashParams(BaseJobParams):
    """Parameters for hash computation task.

    Attributes:
        input_paths: List of absolute paths to input files
        output_paths: List of absolute paths for output files (empty for compute-only)
        algorithm: Hash algorithm to use (sha512 or md5)
    """

    algorithm: Literal["sha512", "md5"] = Field(
        default="sha512",
        description="Hash algorithm to use (sha512 or md5)",
    )
