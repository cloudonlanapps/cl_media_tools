"""HLS streaming conversion task implementation."""

from pathlib import Path
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.schemas import BaseJobParams, Job, TaskResult
from .algo.hls_stream_generator import HLSStreamGenerator, HLSVariant
from .algo.hls_validator import HLSValidator
from .schema import HLSStreamingParams


class HLSStreamingTask(ComputeModule[HLSStreamingParams]):
    """Compute module for HLS streaming conversion."""

    @property
    @override
    def task_type(self) -> str:
        return "hls_streaming"

    @override
    def get_schema(self) -> type[BaseJobParams]:
        return HLSStreamingParams

    @override
    async def execute(
        self,
        job: Job,
        params: HLSStreamingParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> TaskResult:
        """Convert a media file to HLS and store outputs in a directory."""

        input_path = params.input_path
        output_dir = Path(params.output_path)

        # Phase 1: validate input
        if not Path(input_path).exists():
            return TaskResult(
                status="error",
                error=f"Input file not found: {input_path}",
            )

        # Phase 2: prepare output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 3: generate HLS streams
        try:
            generator = HLSStreamGenerator(
                input_file=input_path,
                output_dir=str(output_dir),
            )

            requested_variants = [
                HLSVariant(resolution=v.resolution, bitrate=v.bitrate) for v in params.variants
            ]

            _ = generator.addVariants(requested_variants)

            if params.include_original:
                _ = generator.addOriginal()

            master_playlist = output_dir / "adaptive.m3u8"

            # Validate output
            validator = HLSValidator(str(master_playlist))
            validation = validator.validate()

            if not validation.is_valid:
                return TaskResult(
                    status="error",
                    error=f"HLS validation failed: {', '.join(validation.errors)}",
                )

            if progress_callback:
                progress_callback(100)

            return TaskResult(
                status="ok",
                task_output={
                    "master_playlist": str(master_playlist),
                    "variants_generated": len(requested_variants),
                    "total_segments": validation.total_segments,
                    "include_original": params.include_original,
                },
            )

        except Exception as exc:  # noqa: BLE001
            return TaskResult(
                status="error",
                error=f"HLS conversion failed: {exc}",
            )
