"""HLS streaming conversion task implementation."""

from pathlib import Path
from typing import Callable, override

from ...common.compute_module import ComputeModule
from ...common.file_storage import JobStorage
from .algo.hls_stream_generator import HLSStreamGenerator, HLSVariant
from .algo.hls_validator import HLSValidator
from .schema import HLSStreamingOutput, HLSStreamingParams


class HLSStreamingTask(ComputeModule[HLSStreamingParams, HLSStreamingOutput]):
    """Compute module for HLS streaming conversion."""

    schema: type[HLSStreamingParams] = HLSStreamingParams

    @property
    @override
    def task_type(self) -> str:
        return "hls_streaming"

    @override
    async def run(
        self,
        job_id: str,
        params: HLSStreamingParams,
        storage: JobStorage,
        progress_callback: Callable[[int], None] | None = None,
    ) -> HLSStreamingOutput:
        input_path = Path(params.input_path)
        if not input_path.exists():
            raise FileNotFoundError("Input file not found: " + params.input_path)

        output_dir = Path(
            storage.allocate_path(
                job_id=job_id,
                relative_path=params.output_path,
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = HLSStreamGenerator(
            input_file=str(input_path),
            output_dir=str(output_dir),
        )

        requested_variants = [
            HLSVariant(resolution=v.resolution, bitrate=v.bitrate) for v in params.variants
        ]

        _ = generator.addVariants(requested_variants)

        if params.include_original:
            _ = generator.addOriginal()

        master_playlist = output_dir / "adaptive.m3u8"

        validator = HLSValidator(str(master_playlist))
        validation = validator.validate()

        if not validation.is_valid:
            raise RuntimeError("HLS validation failed: " + ", ".join(validation.errors))

        if progress_callback:
            progress_callback(100)

        return HLSStreamingOutput(
            master_playlist=str(master_playlist),
            variants_generated=len(requested_variants),
            total_segments=validation.total_segments,
            include_original=params.include_original,
        )
