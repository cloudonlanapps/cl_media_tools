# Hash Plugin

Computes cryptographic hashes (SHA-512 or MD5) for uploaded files with media-type-aware processing.

## Features

- **Media Type Detection**: Automatically detects file type (image, video, text, audio, etc.)
- **Optimized Algorithms**:
  - **Images**: SHA-512 hash of decoded pixel data (via PIL)
  - **Videos**: SHA-512 hash of I-frames only (via ffprobe)
  - **Other files**: Generic SHA-512 or MD5 of raw bytes
- **Processing Metrics**: Returns hash value and computation time

## Algorithm Details

The plugin organizes hash algorithms in the `algo/` directory:

- **`algo/image.py`**: `sha512hash_image()` - Uses PIL to decode image and hash pixel data
- **`algo/video.py`**: `sha512hash_video2()` - Extracts I-frames with ffprobe, hashes frame data
- **`algo/md5.py`**: `get_md5_hexdigest()` - Standard MD5 hash of file bytes
- **`algo/generic.py`**: `sha512hash_generic()` - Generic SHA-512 for text, audio, and other files

### Media Type Routing

| Media Type | SHA-512 Algorithm | MD5 Algorithm |
|------------|------------------|---------------|
| IMAGE      | `sha512hash_image` (pixel-based) | `get_md5_hexdigest` |
| VIDEO      | `sha512hash_video2` (I-frames) | `get_md5_hexdigest` |
| TEXT       | `sha512hash_generic` | `get_md5_hexdigest` |
| AUDIO      | `sha512hash_generic` | `get_md5_hexdigest` |
| FILE       | `sha512hash_generic` | `get_md5_hexdigest` |
| URL        | `sha512hash_generic` | `get_md5_hexdigest` |

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | UploadFile | Yes | - | File to hash |
| `algorithm` | str | No | `sha512` | Hash algorithm (`sha512` or `md5`) |
| `priority` | int | No | `5` | Job priority (0-10, higher = more urgent) |

## API Endpoint

```
POST /api/jobs/hash
```

### Example Request

```bash
curl -X POST "http://localhost:8000/api/jobs/hash" \
  -F "file=@document.pdf" \
  -F "algorithm=sha512"
```

### Example Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

## Task Output

On successful completion, the job's `task_output` contains:

```json
{
  "files": [
    {
      "file_path": "/path/to/input/document.pdf",
      "media_type": "file",
      "hash_value": "a7f3d2e1b9c4f8a3d6e5c7b2a1f9e8d7c6b5a4f3e2d1c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1",
      "algorithm_used": "sha512_generic",
      "process_time": 0.0234
    }
  ],
  "total_files": 1
}
```

### Output Fields

- **`file_path`**: Absolute path to input file
- **`media_type`**: Detected media type (image, video, text, audio, file, url)
- **`hash_value`**: Computed hash (hexadecimal string)
- **`algorithm_used`**: Specific algorithm variant used (e.g., `sha512_image`, `md5`)
- **`process_time`**: Time taken to compute hash (seconds)

## Dependencies

Requires:
- `Pillow` - For image hashing
- `opencv-python-headless` - For video hashing (ffprobe)
- `python-magic` - For MIME type detection

Install with:

```bash
pip install cl_ml_tools[compute]
```

## Development Setup

This project uses `uv` for dependency management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Run type checking
basedpyright

# Run linting
ruff check

# Run tests
pytest
```
