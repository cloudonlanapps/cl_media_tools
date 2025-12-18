import csv
import hashlib
import subprocess
from io import BytesIO, StringIO
from typing import TextIO


class UnsupportedMediaType(Exception):
    """Raised when media format or metadata is invalid."""


def validate_csv(csv_output: TextIO, video_size: int) -> bool:
    """Validate that all rows in the CSV data have exactly three columns,
    correct types, and are within video boundaries.
    """
    _ = csv_output.seek(0)
    reader = csv.reader(csv_output)

    for row in reader:
        if len(row) != 3:
            return False

        try:
            offset = int(row[0])
            size = int(row[1])
        except ValueError:
            return False

        if len(row[2]) != 1:
            return False

        if offset < 0 or size <= 0 or offset + size > video_size:
            return False

    return True


def sha512hash_video2(video_stream: BytesIO) -> bytes:
    """Compute SHA-512 hash using I-frames only."""

    _ = video_stream.seek(0)
    video_bytes: bytes = video_stream.getvalue()
    video_size: int = len(video_bytes)

    command: list[str] = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pkt_pos,pkt_size,pict_type",
        "-of",
        "csv=p=0",
        "-",
    ]

    try:
        process = subprocess.run(
            command,
            input=video_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise UnsupportedMediaType("ffprobe command timed out.") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise UnsupportedMediaType(f"An error occurred while running ffprobe: {exc}") from exc

    if process.returncode != 0:
        raise UnsupportedMediaType(f"ffprobe error: {process.stderr.decode(errors='ignore')}")

    csv_output_data: str = process.stdout.decode("utf-8", errors="ignore")

    if not csv_output_data.strip():
        raise UnsupportedMediaType("No data returned from ffprobe.")

    csv_output = StringIO(csv_output_data)

    if not validate_csv(csv_output, video_size):
        raise UnsupportedMediaType("CSV data is invalid.")

    _ = csv_output.seek(0)
    reader = csv.reader(csv_output)

    cumulative_hash = hashlib.sha512()

    try:
        for row in reader:
            if len(row) != 3:
                raise UnsupportedMediaType(f"Invalid CSV row format: expected 3 columns, got {len(row)}")

            offset_str, size_str, frame_type = row

            try:
                offset = int(offset_str)
                size = int(size_str)
            except ValueError as exc:
                raise UnsupportedMediaType(f"Invalid offset or size in CSV: {exc}") from exc

            # Validate bounds
            if offset < 0 or size <= 0 or offset + size > video_size:
                raise UnsupportedMediaType(
                    f"Frame data out of bounds: offset={offset}, size={size}, video_size={video_size}"
                )

            if frame_type == "I":
                _ = video_stream.seek(offset)
                frame_data = video_stream.read(size)
                if len(frame_data) != size:
                    raise UnsupportedMediaType(
                        f"Failed to read complete frame: expected {size} bytes, got {len(frame_data)}"
                    )
                cumulative_hash.update(frame_data)

    except csv.Error as exc:
        raise UnsupportedMediaType(f"Error parsing CSV data: {exc}") from exc
    except IOError as exc:
        raise UnsupportedMediaType(f"Error processing video data: {exc}") from exc

    return cumulative_hash.digest()
