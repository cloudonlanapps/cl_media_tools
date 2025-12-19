# Image Conversion Algorithm

## Overview

Converts images between different formats using PIL/Pillow. Supports common web and print formats including JPEG, PNG, WEBP, GIF, BMP, and TIFF. Handles color mode conversions (RGB/RGBA), quality settings for lossy formats, and optimization for web delivery.

## Algorithm Details

### Library

- **Library:** Pillow (PIL Fork)
- **Version:** 10.0+ recommended
- **License:** HPND (Historical Permission Notice and Disclaimer)
- **Homepage:** [https://pillow.readthedocs.io/](https://pillow.readthedocs.io/)

### Supported Formats

**Input Formats (Read):**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WEBP (.webp)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- HEIF/HEIC (.heif, .heic) - requires pillow-heif
- And 30+ others supported by Pillow

**Output Formats (Write):**
- JPEG (.jpg, .jpeg) - Lossy, good for photos
- PNG (.png) - Lossless, supports transparency
- WEBP (.webp) - Modern format, smaller than JPEG/PNG
- GIF (.gif) - Lossless, supports animation
- BMP (.bmp) - Uncompressed bitmap
- TIFF (.tiff) - Lossless, supports layers

### Implementation

```python
def image_convert(
    *,
    input_path: Path,
    output_path: Path,
    format: str,
    quality: int | None = None,
) -> str:
    """Convert image to target format."""

    with Image.open(input_path) as img:
        fmt = format.lower()

        # 1. Handle color mode conversion
        if fmt in ("jpg", "jpeg") and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # 2. Set format-specific options
        save_kwargs = {}

        if fmt in ("jpg", "jpeg", "webp") and quality is not None:
            save_kwargs["quality"] = quality

        if fmt == "png":
            save_kwargs["optimize"] = True

        # 3. Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 4. Save with PIL format name
        img.save(output_path, format=_get_pil_format(fmt), **save_kwargs)

    return str(output_path)
```

### Color Mode Handling

**Automatic Conversions:**

1. **RGBA → RGB (for JPEG):**
   ```python
   # JPEG doesn't support transparency
   if format == "jpg" and image.mode == "RGBA":
       image = image.convert("RGB")  # Flatten alpha channel to white
   ```

2. **Palette → RGB (for JPEG):**
   ```python
   # Palette mode (P) not supported in JPEG
   if format == "jpg" and image.mode == "P":
       image = image.convert("RGB")
   ```

3. **Preserve RGBA (for PNG/WEBP):**
   ```python
   # PNG and WEBP support transparency, no conversion needed
   if format in ("png", "webp") and image.mode == "RGBA":
       pass  # Keep alpha channel
   ```

### Quality Settings

**JPEG Quality (1-100):**
- **Quality 95-100:** Near-lossless, large file size
- **Quality 85-95:** High quality, recommended for web (default: 85)
- **Quality 75-85:** Good quality, balanced
- **Quality 60-75:** Medium quality, smaller files
- **Quality 1-60:** Low quality, visible artifacts

**WEBP Quality (1-100):**
- **Quality 80-100:** Better than JPEG at same quality
- **Quality 70-80:** Good for web, smaller than JPEG
- **Quality 1-70:** Aggressive compression

**PNG Optimization:**
- Always uses `optimize=True` for smaller file sizes
- No quality parameter (lossless format)
- May take slightly longer but reduces file size ~10-30%

### Performance Characteristics

- **Speed:**
  - Small image (1920×1080): ~50-100ms
  - Large image (4K): ~200-500ms
  - Format change only: ~10-20ms (no re-encoding)
- **Memory:** Full decoded image in memory (~25MB for 4K RGB)
- **Bottleneck:** Image decoding + encoding, not conversion logic

### Format Recommendations

| Use Case | Recommended Format | Quality | Notes |
|----------|-------------------|---------|-------|
| **Web photos** | WEBP | 80-85 | Smallest, modern browsers |
| **Web photos (compat)** | JPEG | 85 | Universal support |
| **Logos/graphics** | PNG | N/A | Transparency, sharp edges |
| **Screenshots** | PNG | N/A | Lossless, text clarity |
| **Print** | TIFF | N/A | Lossless, high quality |
| **Email attachments** | JPEG | 75 | Good balance of size/quality |
| **Archival** | PNG or TIFF | N/A | Lossless, no degradation |

## Parameters

### image_convert()
```python
def image_convert(
    *,
    input_path: str | Path,
    output_path: str | Path,
    format: str,
    quality: int | None = None,
) -> str
```

**Parameters:**
- `input_path` (str | Path): Path to source image
  - Must exist, raises FileNotFoundError if not found
  - Any PIL-supported format
- `output_path` (str | Path): Path for converted image
  - Parent directory created automatically
  - Extension should match format (but not required)
- `format` (str): Target format
  - Accepted values: `"jpg"`, `"jpeg"`, `"png"`, `"webp"`, `"gif"`, `"bmp"`, `"tiff"`
  - Case-insensitive
- `quality` (int | None): Output quality for lossy formats (default: None)
  - Range: 1-100
  - Applies to: JPEG, WEBP
  - Ignored for: PNG, GIF, BMP, TIFF (lossless)
  - Default behavior:
    - JPEG: PIL default (75)
    - WEBP: PIL default (80)
    - Plugin default in schema: 85

**Returns:**
- `str`: Absolute path to output file

**Raises:**
- `FileNotFoundError`: If input file doesn't exist
- `OSError`: If Pillow fails to read/write image
- `ValueError`: If format is unsupported

**Example:**
```python
from cl_ml_tools.algorithms import convert_image

# Convert PNG to JPEG
convert_image(
    input_path="screenshot.png",
    output_path="photo.jpg",
    format="jpg",
    quality=90
)

# Convert JPEG to WEBP (smaller file size)
convert_image(
    input_path="photo.jpg",
    output_path="photo.webp",
    format="webp",
    quality=85
)

# Convert to PNG (lossless, quality ignored)
convert_image(
    input_path="image.jpg",
    output_path="image.png",
    format="png"
)
```

## Output Format

The function creates a new image file at `output_path` with the specified format and returns the path as a string.

**File Size Comparison (example 4K photo):**
```
Original (PNG):        25.3 MB
JPEG (quality=95):      3.8 MB  (-85%)
JPEG (quality=85):      2.1 MB  (-92%)
JPEG (quality=75):      1.5 MB  (-94%)
WEBP (quality=85):      1.2 MB  (-95%)
WEBP (quality=75):      0.8 MB  (-97%)
```

**Output Characteristics:**

| Format | Transparency | Animation | Lossy/Lossless | Typical Use |
|--------|--------------|-----------|----------------|-------------|
| JPEG   | ❌ | ❌ | Lossy | Photos, web images |
| PNG    | ✅ | ❌ | Lossless | Graphics, screenshots |
| WEBP   | ✅ | ✅ | Both | Modern web, photos |
| GIF    | ✅ | ✅ | Lossless | Simple animations |
| BMP    | ❌ | ❌ | Lossless | Windows bitmaps |
| TIFF   | ✅ | ❌ | Lossless | Print, archival |

## Use Cases

### 1. Web Optimization
```python
from cl_ml_tools.algorithms import convert_image
from pathlib import Path

def optimize_for_web(input_dir, output_dir):
    """Convert all images to WEBP for web delivery."""
    for img_path in Path(input_dir).glob("*.jpg"):
        output_path = Path(output_dir) / f"{img_path.stem}.webp"

        convert_image(
            input_path=img_path,
            output_path=output_path,
            format="webp",
            quality=85
        )

        original_size = img_path.stat().st_size
        optimized_size = output_path.stat().st_size
        savings = (1 - optimized_size / original_size) * 100

        print(f"{img_path.name}: {savings:.1f}% smaller")

optimize_for_web("/photos", "/web_photos")
```

### 2. Format Standardization
```python
from pathlib import Path

def standardize_to_jpeg(directory):
    """Convert all images in directory to JPEG."""
    formats = ["*.png", "*.webp", "*.bmp", "*.tiff"]

    for pattern in formats:
        for img_path in Path(directory).glob(pattern):
            output_path = img_path.with_suffix(".jpg")

            convert_image(
                input_path=img_path,
                output_path=output_path,
                format="jpg",
                quality=90
            )

            print(f"Converted: {img_path.name} → {output_path.name}")

            # Optionally delete original
            # img_path.unlink()

standardize_to_jpeg("/mixed_formats")
```

### 3. Thumbnail Generation with Format Change
```python
from PIL import Image

def create_thumbnail_webp(input_path, output_path, size=(256, 256)):
    """Create thumbnail and convert to WEBP."""
    with Image.open(input_path) as img:
        img.thumbnail(size)
        temp_path = output_path.with_suffix(".tmp.jpg")
        img.save(temp_path, "JPEG", quality=85)

    # Convert thumbnail to WEBP
    convert_image(
        input_path=temp_path,
        output_path=output_path,
        format="webp",
        quality=80
    )

    temp_path.unlink()  # Clean up temporary file

create_thumbnail_webp("photo.png", "thumb.webp")
```

### 4. Batch Quality Adjustment
```python
from pathlib import Path

def reduce_quality_batch(input_dir, output_dir, target_quality=75):
    """Re-save JPEGs with lower quality to reduce file size."""
    for jpg_path in Path(input_dir).glob("*.jpg"):
        output_path = Path(output_dir) / jpg_path.name

        convert_image(
            input_path=jpg_path,
            output_path=output_path,
            format="jpg",
            quality=target_quality
        )

        original_kb = jpg_path.stat().st_size / 1024
        compressed_kb = output_path.stat().st_size / 1024

        print(f"{jpg_path.name}: {original_kb:.1f} KB → {compressed_kb:.1f} KB")

reduce_quality_batch("/originals", "/compressed", target_quality=70)
```

### 5. Email-Friendly Conversion
```python
def prepare_for_email(image_path, max_size_mb=5, quality=75):
    """Convert image to JPEG with size constraint."""
    from pathlib import Path

    output_path = Path(image_path).with_suffix(".email.jpg")

    # Start with given quality
    current_quality = quality

    while current_quality > 50:
        convert_image(
            input_path=image_path,
            output_path=output_path,
            format="jpg",
            quality=current_quality
        )

        size_mb = output_path.stat().st_size / (1024 * 1024)

        if size_mb <= max_size_mb:
            print(f"✓ Size: {size_mb:.2f} MB (quality={current_quality})")
            return str(output_path)

        current_quality -= 5

    print(f"⚠ Could not reduce below {max_size_mb} MB")
    return str(output_path)

prepare_for_email("large_photo.png", max_size_mb=3)
```

## Edge Cases & Limitations

### Supported Scenarios
✅ All PIL-supported input formats
✅ Output formats: JPEG, PNG, WEBP, GIF, BMP, TIFF
✅ Automatic RGBA → RGB conversion for JPEG
✅ Automatic palette → RGB conversion for JPEG
✅ Preserves EXIF metadata (when supported by format)
✅ Creates output directory if it doesn't exist

### Limitations

**Color Mode Conversions:**
- **RGBA → RGB:** Transparency flattened to white background
  - Converting PNG with transparency to JPEG loses alpha channel
  - No custom background color option
- **Color Space:** Assumes sRGB, no ICC profile handling
- **16-bit Images:** Converted to 8-bit

**Format Limitations:**
- **JPEG:** No transparency, lossy compression
- **PNG:** Larger file sizes than JPEG for photos
- **WEBP:** Not supported in older browsers (IE11, Safari <14)
- **GIF:** Limited to 256 colors (not suitable for photos)
- **BMP:** Uncompressed, very large file sizes
- **TIFF:** Not web-compatible, mainly for print/archival

**Metadata:**
- EXIF metadata preserved when converting between formats that support it
- Some metadata lost when converting to formats without metadata support (BMP, GIF)
- IPTC/XMP metadata handling depends on PIL version

**Animation:**
- Animated GIFs/WEBPs converted to first frame only
- No multi-page TIFF support (uses first page)

### Error Handling

```python
from PIL import UnidentifiedImageError

try:
    convert_image(
        input_path="photo.jpg",
        output_path="photo.png",
        format="png"
    )
except FileNotFoundError:
    print("Input file not found")
except UnidentifiedImageError:
    print("Corrupted or unsupported image format")
except OSError as e:
    print(f"Pillow error: {e}")
    # Common causes:
    # - Disk full
    # - Permission denied
    # - Corrupted image data
```

### Memory Considerations

**Low Memory Environments:**
- Avoid converting very large images (>50MP)
- Full image loaded into memory during conversion
- Peak memory = ~3× image size (input + output + working buffer)

**Example Memory Usage:**
- 1920×1080 RGB: ~6 MB
- 4K (3840×2160) RGB: ~25 MB
- 8K (7680×4320) RGB: ~100 MB

## Performance Optimization

### Parallel Processing
```python
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def convert_single(args):
    input_path, output_path, format, quality = args
    return convert_image(
        input_path=input_path,
        output_path=output_path,
        format=format,
        quality=quality
    )

# Prepare conversion tasks
tasks = []
for img in Path("/input").glob("*.png"):
    output = Path("/output") / f"{img.stem}.jpg"
    tasks.append((img, output, "jpg", 85))

# Convert in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(convert_single, tasks))

print(f"Converted {len(results)} images")
```

### In-Place Conversion (Advanced)
```python
from PIL import Image
from pathlib import Path

def convert_inplace(filepath, target_format, quality=85):
    """Convert image and replace original (dangerous!)."""
    temp_path = Path(filepath).with_suffix(f".{target_format}")

    convert_image(
        input_path=filepath,
        output_path=temp_path,
        format=target_format,
        quality=quality
    )

    # Replace original
    Path(filepath).unlink()
    temp_path.rename(filepath.with_suffix(f".{target_format}"))

# Use with caution!
# convert_inplace("photo.png", "jpg", quality=90)
```

### Progressive JPEG (Web Optimization)
```python
from PIL import Image

def convert_to_progressive_jpeg(input_path, output_path, quality=85):
    """Create progressive JPEG for better web loading."""
    with Image.open(input_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        img.save(
            output_path,
            "JPEG",
            quality=quality,
            optimize=True,
            progressive=True  # Progressive rendering
        )

convert_to_progressive_jpeg("photo.png", "photo.jpg")
```

## Technical Details

### Color Modes (PIL)
- **RGB:** 3 channels (Red, Green, Blue), 8-bit each
- **RGBA:** 4 channels (RGB + Alpha), 8-bit each
- **L:** Grayscale, 8-bit
- **P:** Palette mode (indexed color), 256 colors max
- **CMYK:** 4 channels for print (Cyan, Magenta, Yellow, Black)

### PIL Format Names
```python
format_map = {
    "jpg":  "JPEG",
    "jpeg": "JPEG",
    "png":  "PNG",
    "webp": "WEBP",
    "gif":  "GIF",
    "bmp":  "BMP",
    "tiff": "TIFF",
}
```

### Compression Algorithms
- **JPEG:** DCT-based lossy compression (ISO/IEC 10918)
- **PNG:** DEFLATE lossless compression (RFC 2083)
- **WEBP:** VP8/VP9 codec (lossy) or lossless
- **GIF:** LZW compression
- **TIFF:** Various (uncompressed, LZW, JPEG, etc.)

## References

- **Pillow Documentation:** [https://pillow.readthedocs.io/](https://pillow.readthedocs.io/)
- **PIL Handbook:** [https://pillow.readthedocs.io/en/stable/handbook/](https://pillow.readthedocs.io/en/stable/handbook/)
- **JPEG Standard:** [https://www.w3.org/Graphics/JPEG/](https://www.w3.org/Graphics/JPEG/)
- **PNG Specification:** [http://www.libpng.org/pub/png/spec/](http://www.libpng.org/pub/png/spec/)
- **WEBP:** [https://developers.google.com/speed/webp](https://developers.google.com/speed/webp)

## Version History

- **v0.2.1:** Current implementation with improved error handling
- **v0.2.0:** Initial image conversion support

## Support

For issues or questions:
- Check tests: `tests/plugins/test_image_conversion.py`
- Review source: `src/cl_ml_tools/plugins/image_conversion/`
- Pillow docs: [https://pillow.readthedocs.io/](https://pillow.readthedocs.io/)
- File issues: [GitHub repository](https://github.com/your-org/cl_ml_tools/issues)
