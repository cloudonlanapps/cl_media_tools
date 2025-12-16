"""EXIF metadata extraction task implementation."""

import json
import logging
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.exif_tool_wrapper import MetadataExtractor
from .schema import ExifMetadata, ExifParams

logger = logging.getLogger(__name__)


class ExifTask(ComputeModule[ExifParams]):
    """Compute module for extracting EXIF metadata from a media file."""

    @property
    @override
    def task_type(self) -> str:
        return "exif"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return ExifParams

    @override
    async def execute(
        self,
        job: Job,
        params: ExifParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Extract EXIF metadata and store it as JSON."""

        # Phase 1: initialize extractor
        try:
            extractor = MetadataExtractor()
        except RuntimeError as exc:
            logger.error("ExifTool initialization failed", exc_info=exc)
            return TaskResult(
                status="error",
                error=(
                    "ExifTool is not installed or not found in PATH. "
                    "Please install ExifTool: https://exiftool.org/"
                ),
            )

        # Phase 2: extract metadata
        try:
            if params.tags:
                raw_metadata = extractor.extract_metadata(
                    params.input_path,
                    tags=params.tags,
                )
            else:
                raw_metadata = extractor.extract_metadata_all(params.input_path)

            if raw_metadata:
                metadata = ExifMetadata.from_raw_metadata(raw_metadata)
            else:
                logger.warning("No EXIF metadata found for %s", params.input_path)
                metadata = ExifMetadata()

            output = {
                "metadata": metadata.model_dump(),
                "tags_requested": params.tags if params.tags else "all",
            }

            # Persist EXIF metadata as JSON
            with open(params.output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)

            if progress_callback:
                progress_callback(100)

            return TaskResult(
                status="ok",
                task_output=output,
            )

        except FileNotFoundError:
            logger.error("File not found: %s", params.input_path)
            return TaskResult(
                status="error",
                error="Input file not found",
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to extract EXIF metadata from %s",
                params.input_path,
                exc_info=exc,
            )
            return TaskResult(
                status="error",
                error=str(exc),
            )
