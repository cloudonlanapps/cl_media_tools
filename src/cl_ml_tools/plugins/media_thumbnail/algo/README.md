# Media Thumbnail Algorithm

## Overview

Generates thumbnails for images and videos using PIL/Pillow for images and FFmpeg for videos. Image thumbnails support aspect ratio preservation and high-quality resampling. Video thumbnails create a 4×4 grid of keyframes for visual preview of video content.

## Algorithm Details

### Libraries

**For Images:**
- **Library:** Pillow (PIL Fork)
- **Resampling:** LANCZOS (high-quality downsampling)
- **License:** HPND

**For Videos:**
- **Tool:** FFmpeg
- **Approach:** Extract keyframes and tile into grid
- **License:** GPL/LGPL
- **Installation:**
  - macOS: `brew install ffmpeg`
  - Linux: `apt-get install ffmpeg`

### Supported Formats

**Images:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WEBP (.webp)
- GIF (.gif) - first frame only
- BMP (.bmp)
- TIFF (.tiff)
- HEIF/HEIC (.heif, .heic) - requires pillow-heif
- All PIL-supported formats

**Videos:**
- MP4 (.mp4)
- MOV (.mov)
- AVI (.avi)
- MKV (.mkv)
- WEBM (.webm)
- FLV (.flv)
- All FFmpeg-supported formats

## Image Thumbnail Implementation

### Algorithm

```python
def image_thumbnail(
    *,
    input_path: Path,
    output_path: Path,
    width: int | None = None,
    height: int | None = None,
    maintain_aspect_ratio: bool = True,
) -> str:
    """Create image thumbnail."""

    # 1. Determine target dimensions
    if width is None and height is None:
        w = h = 256  # Default square
    elif width is None:
        w = h = height  # Square from height
    elif height is None:
        w = h = width  # Square from width
    else:
        w = width
        h = height

    # 2. Open and resize
    with Image.open(input_path) as img:
        if maintain_aspect_ratio:
            # PIL's thumbnail() maintains aspect ratio
            img.thumbnail((w, h), Image.Resampling.LANCZOS)
            thumbnail = img
        else:
            # resize() forces exact dimensions
            thumbnail = img.resize((w, h), Image.Resampling.LANCZOS)

        # 3. Save thumbnail
        thumbnail.save(output_path)

    return str(output_path)
```

### Resampling Quality

**LANCZOS (Lanczos3):**
- **Quality:** Highest quality downsampling
- **Speed:** Slower than BILINEAR/BICUBIC
- **Use Case:** Thumbnails, high-quality previews
- **Trade-off:** 3×3 kernel, more computation but sharper results

**Comparison (downsampling 4K → 256px):**
- NEAREST: Fast, blocky, aliasing artifacts
- BILINEAR: Medium, soft, some aliasing
- BICUBIC: Good, smooth, balanced
- LANCZOS: Best, sharp, minimal artifacts (used here)

### Aspect Ratio Behavior

**maintain_aspect_ratio=True (default):**
```python
# Input: 1920×1080 (16:9)
# Target: 256×256 (square)
# Output: 256×144 (maintains 16:9, fits within box)

# Input: 800×1200 (portrait, 2:3)
# Target: 256×256 (square)
# Output: 171×256 (maintains 2:3, fits within box)
```

**maintain_aspect_ratio=False:**
```python
# Input: 1920×1080 (16:9)
# Target: 256×256 (square)
# Output: 256×256 (stretched/squashed)
```

### Performance Characteristics

**Image Thumbnails:**
- **Speed:**
  - 1920×1080 → 256×256: ~20-50ms
  - 4K → 256×256: ~100-200ms
  - 12MP photo → 256×256: ~150-300ms
- **Memory:** Full decoded image + thumbnail (~minimal)
- **Bottleneck:** Image decoding, not resizing

## Video Thumbnail Implementation

### Algorithm

Video thumbnails use FFmpeg to create a 4×4 grid of keyframes:

```python
def video_thumbnail(
    *,
    input_path: Path,
    output_path: Path,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Create video thumbnail as 4×4 keyframe grid."""

    # 1. Determine dimensions (default 256×256)
    if width is None and height is None:
        width = 256
        height = 256

    # 2. Build scale filter
    if width is not None and height is not None:
        # Fit within box, maintain aspect ratio
        scale_filter = (
            f"scale='min(iw,{width})':'min(ih,{height})'"
            f":force_original_aspect_ratio=decrease"
        )
    elif width is not None:
        scale_filter = f"scale={width}:-1"  # Width fixed, height auto
    else:
        scale_filter = f"scale=-1:{height}"  # Height fixed, width auto

    # 3. FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-loglevel", "panic",   # Suppress output
        "-y",                    # Overwrite output file
        "-skip_frame", "nokey",  # Process only keyframes (I-frames)
        "-i", str(input_path),
        "-frames", "1",          # Output 1 frame (the tiled grid)
        "-q:v", "1",             # Highest quality
        "-vf", f"tile=4x4,loop=16:1,{scale_filter}",
        str(output_path)
    ]

    result = subprocess.run(ffmpeg_command, ...)
```

### FFmpeg Filter Explanation

**Filter Chain:**
```
-vf "tile=4x4,loop=16:1,scale=..."
```

1. **tile=4x4:** Arrange frames into 4×4 grid (16 frames total)
2. **loop=16:1:** Loop input to ensure 16 frames (for short videos)
3. **scale=...:** Resize final grid to target dimensions

**Keyframe Selection:**
- `-skip_frame nokey`: Only process keyframes (I-frames)
- Skips P-frames (predicted) and B-frames (bidirectional)
- Results in evenly distributed preview frames

**Example Grid Layout:**
```
┌─────┬─────┬─────┬─────┐
│ F1  │ F2  │ F3  │ F4  │  Frame 1-4 (keyframes)
├─────┼─────┼─────┼─────┤
│ F5  │ F6  │ F7  │ F8  │  Frame 5-8
├─────┼─────┼─────┼─────┤
│ F9  │ F10 │ F11 │ F12 │  Frame 9-12
├─────┼─────┼─────┼─────┤
│ F13 │ F14 │ F15 │ F16 │  Frame 13-16
└─────┴─────┴─────┴─────┘
```

### Performance Characteristics

**Video Thumbnails:**
- **Speed:**
  - Short video (10s, 1080p): ~500ms-1s
  - Long video (60s, 4K): ~2-5s
  - Very long video (2hr, 1080p): ~5-10s
- **Memory:** FFmpeg processes in streaming mode (minimal memory)
- **Bottleneck:** Seeking to keyframes, decoding 16 frames
- **Dependency:** Requires FFmpeg installed

## Parameters

### image_thumbnail()
```python
def image_thumbnail(
    *,
    input_path: str | Path,
    output_path: str | Path,
    width: int | None = None,
    height: int | None = None,
    maintain_aspect_ratio: bool = True,
) -> str
```

**Parameters:**
- `input_path` (str | Path): Path to source image
  - Must exist, raises FileNotFoundError if not found
- `output_path` (str | Path): Path for thumbnail
  - Parent directory created automatically
  - Format determined by extension
- `width` (int | None): Target width in pixels (default: None)
  - If both width and height are None: defaults to 256×256
  - If only width specified: square thumbnail (width × width)
- `height` (int | None): Target height in pixels (default: None)
  - If only height specified: square thumbnail (height × height)
- `maintain_aspect_ratio` (bool): Preserve aspect ratio (default: True)
  - True: Fit within width×height box, maintain aspect
  - False: Force exact width×height, may distort

**Returns:**
- `str`: Absolute path to output thumbnail

**Raises:**
- `FileNotFoundError`: If input file doesn't exist
- `OSError`: If Pillow fails to read/write image

**Example:**
```python
from cl_ml_tools.algorithms import image_thumbnail

# Square thumbnail (256×256 or smaller if aspect ratio preserved)
image_thumbnail(
    input_path="photo.jpg",
    output_path="thumb.jpg",
    width=256,
    height=256
)

# Fixed width, auto height (maintains aspect ratio)
image_thumbnail(
    input_path="photo.jpg",
    output_path="thumb.jpg",
    width=512
)  # e.g., 1920×1080 → 512×288

# Force exact dimensions (may distort)
image_thumbnail(
    input_path="photo.jpg",
    output_path="thumb.jpg",
    width=200,
    height=200,
    maintain_aspect_ratio=False
)
```

### video_thumbnail()
```python
def video_thumbnail(
    *,
    input_path: str | Path,
    output_path: str | Path,
    width: int | None = None,
    height: int | None = None,
) -> str
```

**Parameters:**
- `input_path` (str | Path): Path to source video
  - Must exist, raises FileNotFoundError if not found
- `output_path` (str | Path): Path for thumbnail (typically .jpg)
  - Parent directory created automatically
- `width` (int | None): Target width in pixels (default: None)
  - If both None: defaults to 256×256
- `height` (int | None): Target height in pixels (default: None)

**Returns:**
- `str`: Absolute path to output thumbnail

**Raises:**
- `FileNotFoundError`: If input file doesn't exist
- `RuntimeError`: If FFmpeg command fails

**Example:**
```python
from cl_ml_tools.algorithms import video_thumbnail

# Default 256×256 grid
video_thumbnail(
    input_path="movie.mp4",
    output_path="preview.jpg"
)

# Larger preview (512×512 grid)
video_thumbnail(
    input_path="movie.mp4",
    output_path="preview_large.jpg",
    width=512,
    height=512
)

# Fixed width, auto height
video_thumbnail(
    input_path="movie.mp4",
    output_path="preview.jpg",
    width=640
)
```

## Output Format

### Image Thumbnails

**Characteristics:**
- Format: Same as input (or based on output_path extension)
- Quality: High (LANCZOS resampling)
- File Size: Typically 5-50 KB for 256px thumbnails

**Dimensions (maintain_aspect_ratio=True):**
```
Input: 1920×1080, Target: 256×256
  → Output: 256×144 (fits within box)

Input: 800×1200, Target: 256×256
  → Output: 171×256 (fits within box)

Input: 1000×1000, Target: 256×256
  → Output: 256×256 (exact fit)
```

**Dimensions (maintain_aspect_ratio=False):**
```
Input: 1920×1080, Target: 256×256
  → Output: 256×256 (stretched)

Input: 800×1200, Target: 256×256
  → Output: 256×256 (squashed)
```

### Video Thumbnails

**Characteristics:**
- Format: JPEG (typically)
- Layout: 4×4 grid of keyframes (16 frames)
- Quality: Highest (-q:v 1)
- File Size: 50-200 KB for 256px grid

**Example Output:**
- Input: 10-second 1080p video
- Output: 256×144 grid (4×4 layout, each cell 64×36)
- Shows frames at: ~0s, ~0.6s, ~1.2s, ... ~9.4s

## Use Cases

### 1. Image Gallery Thumbnails
```python
from pathlib import Path
from cl_ml_tools.algorithms import image_thumbnail

def create_gallery_thumbs(image_dir, thumb_dir, size=256):
    """Generate square thumbnails for image gallery."""
    Path(thumb_dir).mkdir(parents=True, exist_ok=True)

    for img_path in Path(image_dir).glob("*.jpg"):
        thumb_path = Path(thumb_dir) / f"thumb_{img_path.name}"

        image_thumbnail(
            input_path=img_path,
            output_path=thumb_path,
            width=size,
            height=size,
            maintain_aspect_ratio=True
        )

        print(f"Created: {thumb_path.name}")

create_gallery_thumbs("/photos", "/thumbs")
```

### 2. Video Preview Grid
```python
from cl_ml_tools.algorithms import video_thumbnail

def create_video_previews(video_dir, preview_dir):
    """Generate preview grids for all videos."""
    Path(preview_dir).mkdir(parents=True, exist_ok=True)

    for video_path in Path(video_dir).glob("*.mp4"):
        preview_path = Path(preview_dir) / f"{video_path.stem}_preview.jpg"

        video_thumbnail(
            input_path=video_path,
            output_path=preview_path,
            width=512,
            height=512
        )

        print(f"Created preview: {preview_path.name}")

create_video_previews("/videos", "/previews")
```

### 3. Responsive Image Sizes
```python
from cl_ml_tools.algorithms import image_thumbnail

def create_responsive_sizes(image_path, output_dir):
    """Generate multiple thumbnail sizes for responsive web."""
    sizes = {
        "small": 128,
        "medium": 256,
        "large": 512,
        "xlarge": 1024
    }

    for name, size in sizes.items():
        output_path = Path(output_dir) / f"{image_path.stem}_{name}.jpg"

        image_thumbnail(
            input_path=image_path,
            output_path=output_path,
            width=size,
            height=size,
            maintain_aspect_ratio=True
        )

        print(f"{name}: {output_path}")

create_responsive_sizes("photo.jpg", "/responsive")
```

### 4. Batch Processing
```python
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def thumbnail_single(args):
    input_path, output_path = args
    return image_thumbnail(
        input_path=input_path,
        output_path=output_path,
        width=256,
        height=256
    )

def batch_thumbnail(input_dir, output_dir, workers=4):
    """Process many images in parallel."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tasks = []
    for img in Path(input_dir).glob("*.jpg"):
        thumb = Path(output_dir) / f"thumb_{img.name}"
        tasks.append((img, thumb))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(thumbnail_single, tasks))

    print(f"Created {len(results)} thumbnails")

batch_thumbnail("/photos", "/thumbs")
```

### 5. Auto-Detect Media Type
```python
from cl_ml_tools.algorithms import image_thumbnail, video_thumbnail
from cl_ml_tools.utils import determine_media_type, MediaType

def create_thumbnail_auto(media_path, output_path):
    """Auto-detect if image or video and create thumbnail."""
    media_type = determine_media_type(media_path)

    if media_type == MediaType.IMAGE:
        return image_thumbnail(
            input_path=media_path,
            output_path=output_path,
            width=256,
            height=256
        )
    elif media_type == MediaType.VIDEO:
        return video_thumbnail(
            input_path=media_path,
            output_path=output_path,
            width=256,
            height=256
        )
    else:
        raise ValueError(f"Unsupported media type: {media_type}")

# Works for both images and videos
create_thumbnail_auto("photo.jpg", "thumb.jpg")
create_thumbnail_auto("video.mp4", "thumb.jpg")
```

## Edge Cases & Limitations

### Supported Scenarios
✅ All PIL-supported image formats
✅ All FFmpeg-supported video formats
✅ Images with/without transparency (RGBA, RGB, grayscale)
✅ Portrait, landscape, and square images
✅ Very short videos (<1 second)
✅ Very long videos (hours)

### Limitations

**Image Thumbnails:**
- **Animated GIFs:** Only first frame used
- **Multi-page TIFFs:** Only first page used
- **Memory:** Full image loaded into memory
- **EXIF Orientation:** Not automatically applied (may appear rotated)

**Video Thumbnails:**
- **Requires FFmpeg:** Must be installed on system
- **Keyframe Dependency:** Sparse keyframes → sparse grid
  - Some videos have keyframes only every 2-10 seconds
  - Screen recordings may have very sparse keyframes
- **Audio-Only:** Fails (no video stream)
- **No Animation:** Static image output only
- **Timeout:** Very long videos (>1 hour) may be slow

### Error Handling

```python
from PIL import UnidentifiedImageError

# Image thumbnail errors
try:
    image_thumbnail(
        input_path="photo.jpg",
        output_path="thumb.jpg",
        width=256,
        height=256
    )
except FileNotFoundError:
    print("Image not found")
except UnidentifiedImageError:
    print("Corrupted or unsupported image")
except OSError as e:
    print(f"Pillow error: {e}")

# Video thumbnail errors
try:
    video_thumbnail(
        input_path="video.mp4",
        output_path="preview.jpg"
    )
except FileNotFoundError:
    print("Video not found")
except RuntimeError as e:
    print(f"FFmpeg error: {e}")
    # Common causes:
    # - FFmpeg not installed
    # - Corrupted video file
    # - Unsupported codec
```

### Memory Considerations

**Image Thumbnails:**
- 1920×1080 RGB: ~6 MB memory
- 4K RGB: ~25 MB memory
- 8K RGB: ~100 MB memory

**Video Thumbnails:**
- Minimal memory (FFmpeg streams)
- ~50-100 MB peak during grid creation

## Performance Optimization

### Parallel Image Processing
```python
from concurrent.futures import ProcessPoolExecutor

def thumbnail_batch_parallel(image_paths, output_dir, workers=4):
    """Process multiple images in parallel."""
    def process_one(img_path):
        thumb_path = Path(output_dir) / f"thumb_{img_path.name}"
        return image_thumbnail(
            input_path=img_path,
            output_path=thumb_path,
            width=256,
            height=256
        )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_one, image_paths))

    return results
```

### Cache Thumbnails
```python
from pathlib import Path

def get_or_create_thumbnail(image_path, cache_dir, size=256):
    """Get cached thumbnail or create if not exists."""
    cache_path = Path(cache_dir) / f"{image_path.stem}_{size}.jpg"

    if cache_path.exists():
        return str(cache_path)

    return image_thumbnail(
        input_path=image_path,
        output_path=cache_path,
        width=size,
        height=size
    )
```

## References

- **Pillow Documentation:** [https://pillow.readthedocs.io/](https://pillow.readthedocs.io/)
- **PIL Image.thumbnail():** [https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail)
- **FFmpeg Documentation:** [https://ffmpeg.org/documentation.html](https://ffmpeg.org/documentation.html)
- **FFmpeg tile filter:** [https://ffmpeg.org/ffmpeg-filters.html#tile](https://ffmpeg.org/ffmpeg-filters.html#tile)
- **LANCZOS Resampling:** [https://en.wikipedia.org/wiki/Lanczos_resampling](https://en.wikipedia.org/wiki/Lanczos_resampling)

## Version History

- **v0.2.1:** Current implementation with improved error handling
- **v0.2.0:** Initial thumbnail support for images and videos

## Support

For issues or questions:
- Check tests: `tests/plugins/test_media_thumbnail.py`
- Review source: `src/cl_ml_tools/plugins/media_thumbnail/`
- File issues: [GitHub repository](https://github.com/your-org/cl_ml_tools/issues)
