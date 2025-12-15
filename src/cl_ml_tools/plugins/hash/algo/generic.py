"""Generic SHA-512 hash computation."""

import hashlib
import time
from io import BytesIO


def sha512hash_generic(bytes_io: BytesIO) -> tuple[str, float]:
    """Compute SHA-512 hash for generic file.

    Uses chunked reading for memory efficiency with large files.

    Args:
        bytes_io: File content as BytesIO stream

    Returns:
        Tuple of (hash_hexdigest, process_time_seconds)
    """
    start_time = time.time()

    _ = bytes_io.seek(0)
    hash_obj = hashlib.sha512()

    # Chunk reading for memory efficiency
    for chunk in iter(lambda: bytes_io.read(4096), b""):
        hash_obj.update(chunk)

    hash_value = hash_obj.hexdigest()
    end_time = time.time()
    process_time = end_time - start_time

    return hash_value, process_time
