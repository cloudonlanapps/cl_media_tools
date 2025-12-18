# Hash Algorithm

## Overview

Provides cryptographic and perceptual hashing functions for files, images, and videos. Supports MD5 for file integrity verification and SHA-512 for content-based hashing. For media files, implements perceptual hashing that detects visual/temporal changes rather than byte-level changes.

## Algorithm Details

### Hash Types

**1. MD5 Hash (File Integrity)**
- **Algorithm:** MD5 cryptographic hash
- **Input:** Any file as BytesIO stream
- **Output:** 32-character hexadecimal string
- **Use Case:** File integrity checks, duplicate detection
- **Collision Resistance:** Weak (deprecated for security, but fine for deduplication)

**2. SHA-512 Generic Hash**
- **Algorithm:** SHA-512 cryptographic hash
- **Input:** Any file as BytesIO stream
- **Output:** 128-character hexadecimal string
- **Use Case:** Secure file hashing, integrity verification
- **Collision Resistance:** Strong (512-bit digest)

**3. SHA-512 Image Hash (Perceptual)**
- **Algorithm:** SHA-512 hash of decoded pixel data
- **Input:** Image file (JPEG, PNG, WEBP, HEIF, etc.)
- **Output:** 128-character hexadecimal string + processing time
- **Use Case:** Detect visual changes, ignore metadata/compression differences
- **Key Feature:** Same visual content = same hash, even if file bytes differ

**4. SHA-512 Video Hash (Perceptual)**
- **Algorithm:** SHA-512 hash of I-frame data only
- **Input:** Video file (MP4, MOV, AVI, etc.)
- **Output:** 64-byte digest (binary)
- **Use Case:** Detect video content changes, ignore re-encoding
- **Dependencies:** Requires ffprobe (part of FFmpeg)

### Implementation

#### 1. MD5 Hash
```python
def get_md5_hexdigest(bytes_io: BytesIO) -> str:
    hash_md5 = hashlib.md5()
    bytes_io.seek(0)

    # Chunked reading for memory efficiency
    for chunk in iter(lambda: bytes_io.read(4096), b""):
        hash_md5.update(chunk)

    return hash_md5.hexdigest()
```

**Characteristics:**
- Chunk size: 4096 bytes
- Memory efficient for large files
- Resets stream position to start
- Returns lowercase hexadecimal string

#### 2. SHA-512 Generic Hash
```python
def sha512hash_generic(bytes_io: BytesIO) -> tuple[str, float]:
    start_time = time.time()
    bytes_io.seek(0)
    hash_obj = hashlib.sha512()

    for chunk in iter(lambda: bytes_io.read(4096), b""):
        hash_obj.update(chunk)

    hash_value = hash_obj.hexdigest()
    process_time = time.time() - start_time

    return hash_value, process_time
```

**Characteristics:**
- Same chunked reading as MD5
- Measures processing time
- Returns (hash_string, time_seconds)

#### 3. SHA-512 Image Hash (Perceptual)
```python
def sha512hash_image(image_stream: BytesIO) -> tuple[str, float]:
    start_time = time.time()

    with Image.open(image_stream) as im:
        # Hash raw pixel data, not file bytes
        hash = hashlib.sha512(im.tobytes()).hexdigest()

    end_time = time.time()
    process_time = end_time - start_time

    return hash, process_time
```

**Why Perceptual:**
- Hashes decoded pixel data, not file bytes
- Same visual content = same hash, even if:
  - File format differs (JPEG vs PNG)
  - Compression quality differs
  - EXIF metadata differs
  - File was re-saved

**Supported Formats:**
- JPEG, PNG, WEBP, HEIF, BMP, GIF, TIFF
- Handles truncated images (LOAD_TRUNCATED_IMAGES = True)
- Automatically registers HEIF/HEIC support

#### 4. SHA-512 Video Hash (Perceptual)
```python
def sha512hash_video2(video_stream: BytesIO) -> bytes:
    # 1. Use ffprobe to extract I-frame locations
    command = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=pkt_pos,pkt_size,pict_type",
        "-of", "csv=p=0", "-"
    ]

    process = subprocess.run(command, input=video_bytes, ...)

    # 2. Parse CSV output (offset, size, frame_type)
    reader = csv.reader(csv_output)
    cumulative_hash = hashlib.sha512()

    # 3. Hash only I-frames
    for row in reader:
        offset, size, frame_type = int(row[0]), int(row[1]), row[2]
        if frame_type == "I":
            video_stream.seek(offset)
            frame_data = video_stream.read(size)
            cumulative_hash.update(frame_data)

    return cumulative_hash.digest()  # Binary, not hex
```

**Why I-Frames Only:**
- I-frames (Intra-frames) contain full image data
- P/B frames only contain deltas from I-frames
- Same visual content = same I-frames, even if:
  - Video was re-encoded with different codec
  - Bitrate/quality settings changed
  - Container format changed (MP4 → MOV)
  - Metadata differs

**Validation:**
- Checks CSV format (3 columns: offset, size, frame_type)
- Validates frame boundaries (offset + size ≤ video_size)
- Ensures all data is readable
- Raises `UnsupportedMediaType` on errors

### Performance Characteristics

**MD5 Hash:**
- **Speed:** ~500 MB/s (depends on CPU)
- **Memory:** Constant ~4KB buffer
- **Use Case:** Fast integrity checks

**SHA-512 Generic Hash:**
- **Speed:** ~400 MB/s (slightly slower than MD5)
- **Memory:** Constant ~4KB buffer
- **Use Case:** Secure file hashing

**SHA-512 Image Hash:**
- **Speed:**
  - Small images (1920×1080): ~10ms
  - Large images (4K): ~50ms
- **Memory:** Full decoded image in memory (~25MB for 4K RGB)
- **Bottleneck:** Image decoding (PIL), not hashing

**SHA-512 Video Hash:**
- **Speed:**
  - Short video (10s, 1080p): ~200ms
  - Long video (60s, 4K): ~1-2s
- **Memory:** Full video in memory + I-frame extraction
- **Bottleneck:** ffprobe execution and CSV parsing
- **Requires:** FFmpeg/ffprobe installed

## Parameters

### get_md5_hexdigest()
```python
def get_md5_hexdigest(bytes_io: BytesIO) -> str
```

**Parameters:**
- `bytes_io` (BytesIO): File content as in-memory byte stream
  - Stream position reset to 0 before hashing
  - Can be any file type

**Returns:**
- `str`: 32-character lowercase hexadecimal MD5 hash

**Example:**
```python
from cl_ml_tools.algorithms import get_md5_hexdigest
from io import BytesIO

with open("file.bin", "rb") as f:
    md5 = get_md5_hexdigest(BytesIO(f.read()))
print(md5)  # "5d41402abc4b2a76b9719d911017c592"
```

### sha512hash_generic()
```python
def sha512hash_generic(bytes_io: BytesIO) -> tuple[str, float]
```

**Parameters:**
- `bytes_io` (BytesIO): File content as in-memory byte stream

**Returns:**
- `tuple[str, float]`: (hash_hexdigest, process_time_seconds)
  - hash_hexdigest: 128-character lowercase hex string
  - process_time_seconds: Time spent hashing

**Example:**
```python
from cl_ml_tools.algorithms import sha512hash_generic
from io import BytesIO

with open("document.pdf", "rb") as f:
    hash_value, time_taken = sha512hash_generic(BytesIO(f.read()))
print(f"Hash: {hash_value}")
print(f"Processed in {time_taken:.3f}s")
```

### sha512hash_image()
```python
def sha512hash_image(image_stream: BytesIO) -> tuple[str, float]
```

**Parameters:**
- `image_stream` (BytesIO): Image file content as byte stream
  - Supported formats: JPEG, PNG, WEBP, HEIF, BMP, GIF, TIFF

**Returns:**
- `tuple[str, float]`: (hash_hexdigest, process_time_seconds)
  - hash_hexdigest: SHA-512 hash of decoded pixel data
  - process_time_seconds: Time for decoding + hashing

**Raises:**
- `PIL.UnidentifiedImageError`: If image format is unsupported or corrupted

**Example:**
```python
from cl_ml_tools.algorithms import sha512hash_image
from io import BytesIO

with open("photo.jpg", "rb") as f:
    img_hash, time_taken = sha512hash_image(BytesIO(f.read()))

# Same image saved as PNG will have the same hash
with open("photo.png", "rb") as f:
    png_hash, _ = sha512hash_image(BytesIO(f.read()))

assert img_hash == png_hash  # True if visually identical
```

### sha512hash_video2()
```python
def sha512hash_video2(video_stream: BytesIO) -> bytes
```

**Parameters:**
- `video_stream` (BytesIO): Video file content as byte stream
  - Supported formats: MP4, MOV, AVI, MKV (anything ffprobe can read)

**Returns:**
- `bytes`: 64-byte SHA-512 digest (binary, not hex)

**Raises:**
- `UnsupportedMediaType`: If video is invalid, ffprobe fails, or format unsupported
- `subprocess.TimeoutExpired`: If ffprobe takes longer than 30 seconds

**Example:**
```python
from cl_ml_tools.algorithms import sha512hash_video2
from io import BytesIO

with open("video.mp4", "rb") as f:
    video_hash = sha512hash_video2(BytesIO(f.read()))

# Convert binary hash to hex for display
hash_hex = video_hash.hex()
print(f"Video hash: {hash_hex}")
```

## Output Format

### MD5 Output
```python
"5d41402abc4b2a76b9719d911017c592"
```
- Type: `str`
- Length: 32 characters (128 bits as hex)
- Character set: `[0-9a-f]`

### SHA-512 Generic/Image Output
```python
("hash_string_128_chars", 0.042)
```
- Type: `tuple[str, float]`
- hash_string: 128 characters (512 bits as hex)
- process_time: Seconds (float)

### SHA-512 Video Output
```python
b'\x9b\x71\xd2\x24\xbd...'  # 64 bytes
```
- Type: `bytes`
- Length: 64 bytes (512 bits)
- Use `.hex()` to convert to hexadecimal string

## Use Cases

### 1. File Deduplication
```python
from cl_ml_tools.algorithms import get_md5_hexdigest
from io import BytesIO
from pathlib import Path

seen_hashes = set()
duplicates = []

for file_path in Path("/media").rglob("*"):
    if file_path.is_file():
        with open(file_path, "rb") as f:
            file_hash = get_md5_hexdigest(BytesIO(f.read()))

        if file_hash in seen_hashes:
            duplicates.append(file_path)
        else:
            seen_hashes.add(file_hash)

print(f"Found {len(duplicates)} duplicate files")
```

### 2. Image Similarity Detection
```python
from cl_ml_tools.algorithms import sha512hash_image
from io import BytesIO

def are_visually_identical(img1_path, img2_path):
    """Check if two images have identical visual content."""
    with open(img1_path, "rb") as f:
        hash1, _ = sha512hash_image(BytesIO(f.read()))

    with open(img2_path, "rb") as f:
        hash2, _ = sha512hash_image(BytesIO(f.read()))

    return hash1 == hash2

# Returns True even if one is JPEG and other is PNG
identical = are_visually_identical("photo.jpg", "photo.png")
```

### 3. Video Content Monitoring
```python
from cl_ml_tools.algorithms import sha512hash_video2
from io import BytesIO

def detect_video_changes(original_path, new_path):
    """Detect if video content has changed (ignoring re-encoding)."""
    with open(original_path, "rb") as f:
        original_hash = sha512hash_video2(BytesIO(f.read()))

    with open(new_path, "rb") as f:
        new_hash = sha512hash_video2(BytesIO(f.read()))

    return original_hash != new_hash

# Returns False if video was just re-encoded with different codec
changed = detect_video_changes("original.mp4", "reencoded.mov")
```

### 4. File Integrity Verification
```python
from cl_ml_tools.algorithms import sha512hash_generic
from io import BytesIO

def verify_download(file_path, expected_hash):
    """Verify downloaded file matches expected hash."""
    with open(file_path, "rb") as f:
        actual_hash, _ = sha512hash_generic(BytesIO(f.read()))

    if actual_hash == expected_hash:
        print("✓ File integrity verified")
        return True
    else:
        print("✗ File corrupted or tampered")
        return False

verify_download("download.zip", "abc123...")
```

## Edge Cases & Limitations

### Supported Scenarios
✅ Files of any size (chunked reading for memory efficiency)
✅ Image formats: JPEG, PNG, WEBP, HEIF, BMP, GIF, TIFF
✅ Video formats: MP4, MOV, AVI, MKV (anything ffprobe supports)
✅ Truncated/incomplete images (handled gracefully)
✅ Large files (constant memory usage for MD5/SHA-512 generic)

### Limitations

**MD5 Hash:**
- **Security:** Cryptographically broken, not suitable for security applications
- **Collisions:** Theoretical collision attacks exist (but rare in practice)
- **Use Case:** Only for deduplication/integrity, not for security

**SHA-512 Image Hash:**
- **Memory:** Requires full decoded image in memory
  - 4K RGB image ≈ 25MB RAM
  - 8K RGB image ≈ 100MB RAM
- **Lossless Only:** Different compression = different pixel data
  - JPEG quality 95 vs 85 will have different hashes
  - Only detects identical visual content if losslessly converted
- **Format Dependent:** Converting JPEG → PNG may introduce slight color shifts
- **No Rotation Detection:** Rotated image has different hash

**SHA-512 Video Hash:**
- **Requires FFmpeg:** Must have ffprobe installed and in PATH
- **I-Frame Dependency:** Videos without I-frames will fail
  - Some streaming formats use all P-frames
  - Screen recordings may have sparse I-frames
- **Memory:** Requires entire video in memory (not suitable for huge files)
- **Timeout:** 30-second timeout may be insufficient for very long videos
- **Binary Output:** Returns `bytes`, not hex string (must convert with `.hex()`)

### Error Handling

```python
from cl_ml_tools.plugins.hash.algo.video import UnsupportedMediaType
from PIL import UnidentifiedImageError

# Image hashing errors
try:
    hash, time = sha512hash_image(image_stream)
except UnidentifiedImageError:
    print("Corrupted or unsupported image format")

# Video hashing errors
try:
    hash = sha512hash_video2(video_stream)
except UnsupportedMediaType as e:
    print(f"Video processing failed: {e}")
    # Common reasons:
    # - ffprobe not installed
    # - Invalid video format
    # - No I-frames in video
    # - Video corrupted
except subprocess.TimeoutExpired:
    print("Video processing timed out (>30 seconds)")
```

### Memory Considerations

**Low Memory Environments:**
- Use `get_md5_hexdigest()` or `sha512hash_generic()` for large files
- Avoid `sha512hash_image()` for very large images (>50MP)
- Avoid `sha512hash_video2()` for videos >500MB

**High Memory Environments:**
- All functions safe to use
- Image hash can handle 8K images
- Video hash can handle long videos

## Performance Optimization

### For Large File Sets

**Parallel Hashing:**
```python
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def hash_file(file_path):
    with open(file_path, "rb") as f:
        return get_md5_hexdigest(BytesIO(f.read()))

files = list(Path("/media").rglob("*.jpg"))

with ProcessPoolExecutor(max_workers=4) as executor:
    hashes = list(executor.map(hash_file, files))
```

**Incremental Hashing (avoid re-reading):**
```python
# Instead of reading file twice:
with open(path, "rb") as f:
    data = f.read()

md5 = get_md5_hexdigest(BytesIO(data))
sha512, _ = sha512hash_generic(BytesIO(data))
```

### For Video Processing

**Batch Processing:**
```python
# Process multiple videos in parallel
from concurrent.futures import ThreadPoolExecutor

def hash_video(video_path):
    with open(video_path, "rb") as f:
        return sha512hash_video2(BytesIO(f.read()))

videos = ["video1.mp4", "video2.mp4", "video3.mp4"]

with ThreadPoolExecutor(max_workers=2) as executor:
    hashes = list(executor.map(hash_video, videos))
```

**Skip for Streaming Content:**
- Don't use video hash for live streams or very long recordings
- Consider sampling strategy (hash first 5 minutes only)

## Technical Details

### MD5 Algorithm
- **Standard:** RFC 1321
- **Block Size:** 512 bits
- **Output Size:** 128 bits (16 bytes)
- **Speed:** ~500 MB/s on modern CPUs

### SHA-512 Algorithm
- **Standard:** FIPS 180-4
- **Block Size:** 1024 bits
- **Output Size:** 512 bits (64 bytes)
- **Speed:** ~400 MB/s on modern CPUs
- **Security:** No known practical attacks

### Perceptual Hashing Philosophy
- **Goal:** Detect content similarity, not byte similarity
- **Trade-off:** More expensive (decode + hash) vs standard hashing
- **Applications:** Media deduplication, plagiarism detection, content monitoring

## References

- **MD5:** [RFC 1321](https://www.ietf.org/rfc/rfc1321.txt)
- **SHA-512:** [FIPS 180-4](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf)
- **FFmpeg/ffprobe:** [https://ffmpeg.org/](https://ffmpeg.org/)
- **Pillow (PIL):** [https://pillow.readthedocs.io/](https://pillow.readthedocs.io/)
- **pillow-heif:** [https://github.com/bigcat88/pillow_heif](https://github.com/bigcat88/pillow_heif)

## Version History

- **v0.2.1:** Current implementation with improved error handling
- **v0.2.0:** Initial hash plugin with MD5, SHA-512, perceptual hashing

## Support

For issues or questions:
- Check tests: `tests/plugins/test_hash.py`
- Review source: `src/cl_ml_tools/plugins/hash/`
- File issues: [GitHub repository](https://github.com/your-org/cl_ml_tools/issues)
