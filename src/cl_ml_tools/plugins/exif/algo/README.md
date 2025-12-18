# EXIF Metadata Extraction Algorithm

## Overview

Extracts EXIF and other metadata from images, videos, and documents using Phil Harvey's ExifTool utility. Provides structured access to camera settings, GPS coordinates, timestamps, and thousands of other metadata fields. Supports selective tag extraction or complete metadata dumps.

## Algorithm Details

### External Dependency

- **Tool:** ExifTool by Phil Harvey
- **Version:** 12.0+ recommended
- **License:** GPL-1.0-or-later / Artistic-2.0
- **Homepage:** [https://exiftool.org/](https://exiftool.org/)
- **Installation:**
  - macOS: `brew install exiftool`
  - Linux: `apt-get install libimage-exiftool-perl`
  - Windows: Download from exiftool.org

### Supported Formats

ExifTool supports **600+ file formats**, including:

**Images:**
- JPEG, PNG, WEBP, HEIF/HEIC, TIFF, BMP, GIF
- RAW formats: CR2, NEF, ARW, DNG, RAF, ORF
- Adobe: PSD, AI

**Videos:**
- MP4, MOV, AVI, MKV, FLV, WEBM
- 3GP, M4V, MPG

**Documents:**
- PDF, DOCX, XLSX, PPTX
- HTML, XML, JSON

**Audio:**
- MP3, M4A, FLAC, WAV, OGG

**Others:**
- ZIP, EXE, DLL (for embedded metadata)

### Implementation

The plugin wraps ExifTool's command-line interface via Python `subprocess`:

#### 1. Selective Tag Extraction
```python
def extract_metadata(filepath: Path, tags: list[str]) -> MetadataDict:
    # Build command with specific tags
    tag_args = [f"-{tag}" for tag in tags]

    result = subprocess.run(
        ["exiftool", "-n", "-j", *tag_args, str(filepath)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30
    )

    parsed = json.loads(result.stdout)
    return parsed[0] if parsed else {}
```

**Flags Explained:**
- `-n`: Numeric output (GPS coordinates as decimals, not degrees/minutes/seconds)
- `-j`: JSON output format
- `-<tag>`: Extract specific tag (e.g., `-Make`, `-Model`, `-GPSLatitude`)

**Example Command:**
```bash
exiftool -n -j -Make -Model -ISO -GPSLatitude -GPSLongitude photo.jpg
```

**Output:**
```json
[{
  "Make": "Canon",
  "Model": "Canon EOS R5",
  "ISO": 400,
  "GPSLatitude": 37.7749,
  "GPSLongitude": -122.4194
}]
```

#### 2. Complete Metadata Extraction
```python
def extract_metadata_all(filepath: Path) -> MetadataDict:
    result = subprocess.run(
        ["exiftool", "-G", "-n", "-j", str(filepath)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30
    )

    parsed = json.loads(result.stdout)
    return parsed[0] if parsed else {}
```

**Flags Explained:**
- `-G`: Include group names (EXIF, XMP, IPTC, etc.)
- `-n`: Numeric output
- `-j`: JSON output

**Example Output:**
```json
[{
  "EXIF:Make": "Canon",
  "EXIF:Model": "Canon EOS R5",
  "EXIF:ISO": 400,
  "EXIF:FNumber": 2.8,
  "EXIF:ExposureTime": "1/100",
  "GPS:GPSLatitude": 37.7749,
  "GPS:GPSLongitude": -122.4194,
  "XMP:Title": "Sunset Photo",
  "IPTC:Keywords": ["nature", "sunset"]
}]
```

### Structured Output Schema

The plugin provides a typed output schema for common EXIF fields:

```python
class ExifMetadataOutput(TaskOutput):
    # Camera Information
    make: str | None              # Camera manufacturer
    model: str | None             # Camera model

    # Timestamps
    date_time_original: str | None  # Original capture time
    create_date: str | None         # File creation date

    # Image Dimensions
    image_width: int | None
    image_height: int | None
    orientation: int | None       # 1-8 (rotation/flip)

    # Camera Settings
    iso: int | None               # ISO speed
    f_number: float | None        # Aperture (e.g., 2.8)
    exposure_time: str | None     # Shutter speed (e.g., "1/100")
    focal_length: float | None    # Lens focal length (mm)

    # GPS Coordinates
    gps_latitude: float | None    # Decimal degrees
    gps_longitude: float | None   # Decimal degrees
    gps_altitude: float | None    # Meters above sea level

    # Software
    software: str | None          # Editing software used

    # Raw JSON
    raw_metadata: dict[str, JSONValue]  # All extracted tags
```

### Performance Characteristics

- **Speed:**
  - Small image (5MB): ~50-100ms
  - Large RAW file (50MB): ~200-500ms
  - Video file (500MB): ~1-2s
- **Memory:** Minimal (~5-10MB for ExifTool process)
- **Bottleneck:** Subprocess spawning and JSON parsing
- **Timeout:** 30 seconds per file (configurable)

## Parameters

### MetadataExtractor.__init__()
```python
def __init__(self) -> None
```

**Raises:**
- `RuntimeError`: If ExifTool is not installed or not found in PATH

**Example:**
```python
from cl_ml_tools.algorithms import ExifToolWrapper

try:
    extractor = ExifToolWrapper()
except RuntimeError as e:
    print("ExifTool not installed. Install with: brew install exiftool")
```

### MetadataExtractor.extract_metadata()
```python
def extract_metadata(
    filepath: str | Path,
    tags: list[str]
) -> MetadataDict
```

**Parameters:**
- `filepath` (str | Path): Path to media file
  - Must exist, raises FileNotFoundError if not found
- `tags` (list[str]): List of EXIF tags to extract
  - Tag names are case-insensitive
  - Examples: `["Make", "Model", "ISO", "GPSLatitude"]`
  - Empty list returns empty dict

**Returns:**
- `MetadataDict`: Dictionary mapping tag names to values
  - Type: `dict[str, str | int | float | bool | None | list | dict]`
  - Numeric values returned as numbers (not strings)
  - Missing tags not included in output

**Raises:**
- `FileNotFoundError`: If file doesn't exist

**Error Handling:**
- ExifTool failures return empty dict (logged as warning)
- Timeout returns empty dict (logged as warning)
- JSON parse errors return empty dict (logged as warning)

**Example:**
```python
from cl_ml_tools.algorithms import ExifToolWrapper
from pathlib import Path

extractor = ExifToolWrapper()

# Extract specific tags
metadata = extractor.extract_metadata(
    filepath="photo.jpg",
    tags=["Make", "Model", "ISO", "FNumber", "ExposureTime"]
)

print(metadata)
# {
#   "Make": "Canon",
#   "Model": "Canon EOS R5",
#   "ISO": 400,
#   "FNumber": 2.8,
#   "ExposureTime": "1/100"
# }
```

### MetadataExtractor.extract_metadata_all()
```python
def extract_metadata_all(
    filepath: str | Path
) -> MetadataDict
```

**Parameters:**
- `filepath` (str | Path): Path to media file

**Returns:**
- `MetadataDict`: Dictionary with ALL extractable metadata
  - Includes group prefixes (e.g., "EXIF:Make", "GPS:Latitude")
  - Can contain hundreds of fields
  - Numeric values (-n flag applied)

**Raises:**
- `FileNotFoundError`: If file doesn't exist

**Example:**
```python
extractor = ExifToolWrapper()

# Extract all metadata
all_metadata = extractor.extract_metadata_all("photo.jpg")

print(f"Found {len(all_metadata)} metadata fields")
print(all_metadata.keys())
# dict_keys(['EXIF:Make', 'EXIF:Model', 'EXIF:ISO', ...])
```

## Output Format

### Selective Extraction
```json
{
  "Make": "Sony",
  "Model": "ILCE-7M4",
  "ISO": 800,
  "FNumber": 1.4,
  "ExposureTime": "1/200",
  "GPSLatitude": 40.7128,
  "GPSLongitude": -74.0060,
  "GPSAltitude": 10.5
}
```

### Complete Extraction (with groups)
```json
{
  "SourceFile": "/path/to/photo.jpg",
  "File:FileSize": 5242880,
  "File:FileType": "JPEG",
  "EXIF:Make": "Sony",
  "EXIF:Model": "ILCE-7M4",
  "EXIF:Orientation": 1,
  "EXIF:XResolution": 72,
  "EXIF:YResolution": 72,
  "EXIF:ISO": 800,
  "EXIF:DateTimeOriginal": "2024:06:20 16:45:30",
  "EXIF:FNumber": 1.4,
  "EXIF:ExposureTime": "1/200",
  "EXIF:FocalLength": 50,
  "GPS:GPSLatitude": 40.7128,
  "GPS:GPSLongitude": -74.0060,
  "GPS:GPSAltitude": 10.5,
  "XMP:Title": "City Skyline",
  "XMP:Creator": "John Doe",
  "IPTC:Keywords": ["architecture", "urban"]
}
```

### Structured Output
```python
from cl_ml_tools.plugins.exif.schema import ExifMetadataOutput

output = ExifMetadataOutput.from_raw_metadata(raw_metadata)

# Access typed fields
print(output.make)              # "Sony"
print(output.iso)               # 800
print(output.gps_latitude)      # 40.7128
print(output.gps_longitude)     # -74.0060
print(output.raw_metadata)      # Full dict
```

## Common EXIF Tags

### Camera Information
- `Make` - Camera manufacturer (e.g., "Canon", "Sony", "Nikon")
- `Model` - Camera model (e.g., "Canon EOS R5", "ILCE-7M4")
- `LensModel` - Lens used (e.g., "RF 50mm F1.8 STM")
- `SerialNumber` - Camera serial number

### Timestamps
- `DateTimeOriginal` - Original capture datetime (EXIF format)
- `CreateDate` - File creation date
- `ModifyDate` - Last modification date
- `GPSDateTime` - GPS timestamp (UTC)

### Image Properties
- `ImageWidth` - Image width in pixels
- `ImageHeight` - Image height in pixels
- `Orientation` - Rotation/flip (1-8)
- `ColorSpace` - Color space (1=sRGB, 65535=Uncalibrated)
- `BitsPerSample` - Bit depth per channel

### Camera Settings
- `ISO` - ISO sensitivity (e.g., 100, 400, 3200)
- `FNumber` - Aperture (e.g., 2.8, 5.6, 11)
- `ExposureTime` - Shutter speed (e.g., "1/100", "1/4000")
- `FocalLength` - Focal length in mm (e.g., 50, 85, 200)
- `ExposureProgram` - Shooting mode (1=Manual, 2=Program, etc.)
- `MeteringMode` - Metering mode (1=Average, 2=Center-weighted, etc.)
- `Flash` - Flash status (0=No flash, 1=Flash fired)
- `WhiteBalance` - White balance mode

### GPS Location
- `GPSLatitude` - Latitude in decimal degrees (with -n flag)
- `GPSLongitude` - Longitude in decimal degrees (with -n flag)
- `GPSAltitude` - Altitude in meters
- `GPSLatitudeRef` - N or S (without -n flag)
- `GPSLongitudeRef` - E or W (without -n flag)

### XMP/IPTC Metadata
- `Title` - Image title
- `Description` - Image description
- `Creator` - Photographer/creator name
- `Keywords` - Keyword tags (list)
- `Copyright` - Copyright notice
- `Rating` - Star rating (0-5)

## Use Cases

### 1. Camera Settings Analysis
```python
from cl_ml_tools.algorithms import ExifToolWrapper

extractor = ExifToolWrapper()

# Extract camera settings
settings = extractor.extract_metadata(
    "photo.jpg",
    tags=["Make", "Model", "ISO", "FNumber", "ExposureTime", "FocalLength"]
)

print(f"Camera: {settings.get('Make')} {settings.get('Model')}")
print(f"Settings: ISO {settings.get('ISO')}, "
      f"f/{settings.get('FNumber')}, "
      f"{settings.get('ExposureTime')}s, "
      f"{settings.get('FocalLength')}mm")

# Output:
# Camera: Canon Canon EOS R5
# Settings: ISO 400, f/2.8, 1/100s, 50mm
```

### 2. GPS Location Extraction
```python
def get_photo_location(filepath):
    """Extract GPS coordinates from photo."""
    extractor = ExifToolWrapper()

    gps_data = extractor.extract_metadata(
        filepath,
        tags=["GPSLatitude", "GPSLongitude", "GPSAltitude"]
    )

    if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
        lat = gps_data["GPSLatitude"]
        lon = gps_data["GPSLongitude"]
        alt = gps_data.get("GPSAltitude", 0)

        return {
            "latitude": lat,
            "longitude": lon,
            "altitude": alt,
            "maps_url": f"https://www.google.com/maps?q={lat},{lon}"
        }

    return None

location = get_photo_location("vacation.jpg")
if location:
    print(f"Photo taken at: {location['latitude']}, {location['longitude']}")
    print(f"View on map: {location['maps_url']}")
```

### 3. Batch Metadata Extraction
```python
from pathlib import Path
import json

def extract_metadata_batch(image_dir, output_file):
    """Extract metadata from all images in directory."""
    extractor = ExifToolWrapper()
    results = {}

    for image_path in Path(image_dir).glob("*.jpg"):
        metadata = extractor.extract_metadata(
            image_path,
            tags=["Make", "Model", "DateTimeOriginal", "GPSLatitude", "GPSLongitude"]
        )
        results[str(image_path)] = metadata

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Extracted metadata from {len(results)} images")

extract_metadata_batch("/photos", "metadata.json")
```

### 4. Photo Organization by Date
```python
from datetime import datetime
from pathlib import Path
import shutil

def organize_photos_by_date(source_dir, dest_dir):
    """Organize photos into YYYY/MM folders based on capture date."""
    extractor = ExifToolWrapper()

    for photo in Path(source_dir).glob("*.jpg"):
        metadata = extractor.extract_metadata(
            photo,
            tags=["DateTimeOriginal"]
        )

        if "DateTimeOriginal" in metadata:
            # Parse EXIF datetime (format: "2024:06:20 16:45:30")
            dt_str = metadata["DateTimeOriginal"]
            dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")

            # Create YYYY/MM folder structure
            year_folder = Path(dest_dir) / str(dt.year)
            month_folder = year_folder / f"{dt.month:02d}"
            month_folder.mkdir(parents=True, exist_ok=True)

            # Copy photo to organized location
            dest_path = month_folder / photo.name
            shutil.copy2(photo, dest_path)
            print(f"Moved {photo.name} to {month_folder}")

organize_photos_by_date("/unsorted_photos", "/organized_photos")
```

### 5. Video Metadata Extraction
```python
def analyze_video_metadata(video_path):
    """Extract technical metadata from video file."""
    extractor = ExifToolWrapper()

    video_meta = extractor.extract_metadata(
        video_path,
        tags=[
            "Duration", "ImageWidth", "ImageHeight",
            "VideoFrameRate", "VideoCodec", "AudioCodec",
            "CreateDate", "Make", "Model"
        ]
    )

    print(f"Duration: {video_meta.get('Duration')}s")
    print(f"Resolution: {video_meta.get('ImageWidth')}x{video_meta.get('ImageHeight')}")
    print(f"Frame Rate: {video_meta.get('VideoFrameRate')} fps")
    print(f"Video Codec: {video_meta.get('VideoCodec')}")
    print(f"Audio Codec: {video_meta.get('AudioCodec')}")
    print(f"Recorded: {video_meta.get('CreateDate')}")
    print(f"Camera: {video_meta.get('Make')} {video_meta.get('Model')}")

analyze_video_metadata("vacation_clip.mp4")
```

## Edge Cases & Limitations

### Supported Scenarios
✅ 600+ file formats supported by ExifTool
✅ Images with no EXIF (returns empty dict, no error)
✅ RAW camera formats (CR2, NEF, ARW, DNG, etc.)
✅ Videos with embedded metadata
✅ PDFs with document metadata
✅ Case-insensitive tag names

### Limitations

**Dependency Required:**
- Requires ExifTool installed on system
- Not available as Python package (external binary)
- Must be in PATH or specify full path

**Performance:**
- Subprocess overhead (~10-50ms per invocation)
- Not suitable for real-time processing
- Consider batch processing for large datasets

**Output Format:**
- All values returned as JSON primitives (str, int, float, bool, None)
- Complex structures flattened
- Some tags may have platform-specific formatting

**Error Handling:**
- Non-existent files raise FileNotFoundError
- Corrupt files return empty dict (no exception)
- Missing tags silently omitted from output
- Timeout errors return empty dict

### Common Issues

**ExifTool Not Found:**
```python
RuntimeError: ExifTool is not installed or not found in PATH.
```
**Solution:** Install ExifTool via package manager

**Empty Results:**
```python
{}  # Empty dictionary
```
**Causes:**
- File has no metadata
- Requested tags don't exist
- ExifTool failed to parse file
- Timeout occurred

**Missing GPS Coordinates:**
- Many cameras don't record GPS by default
- Smartphone photos usually have GPS
- Can be stripped by social media platforms

## Performance Optimization

### Batch Processing
```python
# Instead of spawning ExifTool for each file:
for photo in photos:
    metadata = extractor.extract_metadata(photo, tags)  # Slow

# Consider using ExifTool's batch mode directly:
import subprocess
import json

result = subprocess.run(
    ["exiftool", "-n", "-j", *photos],
    capture_output=True,
    text=True
)
all_metadata = json.loads(result.stdout)  # List of dicts
```

### Parallel Processing
```python
from concurrent.futures import ProcessPoolExecutor

def extract_single(filepath):
    extractor = ExifToolWrapper()
    return extractor.extract_metadata(filepath, tags=["Make", "Model", "ISO"])

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(extract_single, photo_paths))
```

### Cache Results
```python
import json
from pathlib import Path

CACHE_FILE = "metadata_cache.json"

def get_metadata_cached(filepath, tags):
    """Extract metadata with file-based caching."""
    cache = {}
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)

    file_key = str(filepath)
    if file_key in cache:
        return cache[file_key]

    extractor = ExifToolWrapper()
    metadata = extractor.extract_metadata(filepath, tags)

    cache[file_key] = metadata
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

    return metadata
```

## References

- **ExifTool Homepage:** [https://exiftool.org/](https://exiftool.org/)
- **ExifTool Tag Names:** [https://exiftool.org/TagNames/](https://exiftool.org/TagNames/)
- **EXIF Standard:** [https://www.cipa.jp/std/documents/e/DC-008-2012_E.pdf](https://www.cipa.jp/std/documents/e/DC-008-2012_E.pdf)
- **GPS EXIF Tags:** [https://exiftool.org/TagNames/GPS.html](https://exiftool.org/TagNames/GPS.html)
- **XMP Specification:** [https://www.adobe.com/devnet/xmp.html](https://www.adobe.com/devnet/xmp.html)

## Version History

- **v0.2.1:** Current implementation with improved error handling
- **v0.2.0:** Initial EXIF extraction support with ExifTool wrapper

## Support

For issues or questions:
- Check tests: `tests/plugins/test_exif.py`
- Review source: `src/cl_ml_tools/plugins/exif/`
- ExifTool docs: [https://exiftool.org/exiftool_pod.html](https://exiftool.org/exiftool_pod.html)
- File issues: [GitHub repository](https://github.com/your-org/cl_ml_tools/issues)
