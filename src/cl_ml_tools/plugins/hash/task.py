"""Hash computation task implementation."""

import json
import time
from io import BytesIO
from pathlib import Path
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from ...utils.media_types import MediaType, determine_media_type
from .algo.generic import sha512hash_generic
from .algo.image import sha512hash_image
from .algo.md5 import get_md5_hexdigest
from .algo.video import sha512hash_video2
from .schema import HashParams


class HashTask(ComputeModule[HashParams]):
    """Compute module for file hashing with media-type detection."""

    @property
    @override
    def task_type(self) -> str:
        return "hash"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return HashParams

    @override
    async def execute(
        self,
        job: Job,
        params: HashParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Compute hash for a single file and persist result to disk."""

        input_path = Path(params.input_path)

        # Phase 1: validate input
        if not input_path.exists():
            return TaskResult(
                status="error",
                error=f"Input file not found: {params.input_path}",
            )

        try:
            # Read file once
            file_bytes = input_path.read_bytes()
            bytes_io = BytesIO(file_bytes)

            # Detect media type
            import magic  # lazy import

            mime = magic.Magic(mime=True)
            file_type = mime.from_buffer(file_bytes)
            media_type = determine_media_type(bytes_io, file_type)

            _ = bytes_io.seek(0)
            start = time.time()
            # Compute hash
            if params.algorithm == "md5":
                hash_value = get_md5_hexdigest(bytes_io)
                process_time = 0.0
                algorithm_used = "md5"

            elif media_type == MediaType.IMAGE:
                hash_value, process_time = sha512hash_image(bytes_io)
                algorithm_used = "sha512_image"

            elif media_type == MediaType.VIDEO:
                hash_bytes = sha512hash_video2(bytes_io)
                hash_value = hash_bytes.hex()
                algorithm_used = "sha512_video"
            else:
                hash_value, process_time = sha512hash_generic(bytes_io)
                algorithm_used = "sha512_generic"
            process_time = time.time() - start

            output = {
                "hash_value": hash_value,
                "algorithm": algorithm_used,
                "media_type": media_type.value,
                "process_time": process_time,
            }

            # Persist hash result
            with open(params.output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)

            if progress_callback:
                progress_callback(100)

            return TaskResult(
                status="ok",
                task_output=output,
            )

        except ImportError as exc:
            return TaskResult(
                status="error",
                error=(
                    f"Missing dependency: {exc}. Install with: pip install cl_ml_tools[compute]"
                ),
            )

        except Exception as exc:  # noqa: BLE001
            return TaskResult(
                status="error",
                error=f"Hash computation failed: {exc}",
            )
