# HLS Streaming Algorithm

## Overview

Converts video files to HTTP Live Streaming (HLS) format using FFmpeg. Generates adaptive bitrate streaming with multiple quality variants (720p, 480p, 240p, etc.), master playlist (M3U8), and transport stream segments (.ts). Enables smooth video playback with automatic quality switching based on network conditions.

## Algorithm Details

### External Dependency

- **Tool:** FFmpeg
- **Version:** 4.0+ recommended
- **License:** GPL/LGPL
- **Installation:**
  - macOS: `brew install ffmpeg`
  - Linux: `apt-get install ffmpeg`
  - Requires H.264 video codec support

### HLS Format

**HTTP Live Streaming (HLS):**
- Developed by Apple for adaptive bitrate streaming
- Widely supported (iOS, Android, web browsers, smart TVs)
- Uses HTTP protocol (works through firewalls/CDNs)
- Splits video into small segments (typically 6-10 seconds)

**Components:**
1. **Master Playlist (.m3u8):** Lists all quality variants
2. **Variant Playlists (.m3u8):** List segments for each quality
3. **Transport Segments (.ts):** Actual video chunks

### Adaptive Bitrate Streaming

**How it works:**
1. Client requests master playlist
2. Client selects appropriate quality based on network speed
3. Client downloads video segments sequentially
4. Client switches quality dynamically if network changes

**Benefits:**
- Smooth playback without buffering
- Automatic quality adaptation
- Fast startup (progressive download)
- Seek-friendly (segment-based)

### Implementation

#### 1. Generate Master Playlist
```python
# Create master.m3u8 listing all variants
#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=3500000,RESOLUTION=1280x720
adaptive-720p-3500.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=1500000,RESOLUTION=854x480
adaptive-480p-1500.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=426x240
adaptive-240p-800.m3u8
```

#### 2. Transcode Video to Variants
```python
# For each variant (720p, 480p, 240p):
ffmpeg_command = [
    "ffmpeg",
    "-i", input_video,
    "-vf", "scale=-2:720",  # Scale to 720p height, maintain aspect ratio
    "-c:v", "libx264",       # H.264 codec
    "-b:v", "3500k",         # Target bitrate
    "-c:a", "aac",          # AAC audio codec
    "-b:a", "128k",         # Audio bitrate
    "-hls_time", "6",       # 6-second segments
    "-hls_list_size", "0",  # Include all segments in playlist
    "-hls_segment_filename", "segment_%03d.ts",
    "adaptive-720p-3500.m3u8"
]
```

#### 3. Generate Segments
- Video split into 6-second chunks (configurable)
- Each segment: independent, seekable, decodable
- Naming: `segment_000.ts`, `segment_001.ts`, etc.

### Performance Characteristics

**Processing Time:**
- Depends on video length and number of variants
- Example: 60-second 1080p video → 3 variants:
  - 720p: ~30 seconds
  - 480p: ~20 seconds
  - 240p: ~15 seconds
  - Total: ~65 seconds (~1.1× video length)

**Disk Usage:**
- Original: 100 MB
- 720p variant: ~35 MB
- 480p variant: ~18 MB
- 240p variant: ~9 MB
- Total: ~162 MB (original + variants)

**Segment Size:**
- 6-second segments at 720p/3500kbps: ~2.6 MB per segment
- 10-minute video: ~100 segments per variant

## Parameters

### HLSStreamGenerator.__init__()
```python
def __init__(self, input_file: str, output_dir: str)
```

**Parameters:**
- `input_file` (str): Path to input video file
  - Supported formats: MP4, MOV, AVI, MKV, WEBM, FLV
  - Must exist, raises NotFound if not found
- `output_dir` (str): Directory for output HLS files
  - Created automatically if doesn't exist
  - Will contain master playlist, variant playlists, and .ts segments

### Variant Configuration
```python
class HLSVariant:
    def __init__(self, resolution: int | None = None, bitrate: int | None = None)
```

**Parameters:**
- `resolution` (int | None): Target height in pixels
  - Common values: 1080, 720, 480, 360, 240, 144
  - Width calculated automatically to maintain aspect ratio
  - None = original resolution (copy codec, no re-encoding)
- `bitrate` (int | None): Target bitrate in kbps
  - 720p: 2500-4000 kbps
  - 480p: 1000-2000 kbps
  - 240p: 500-1000 kbps
  - None = use FFmpeg default

### Default Variants
```python
variants = [
    HLSVariant(resolution=720, bitrate=3500),   # HD quality
    HLSVariant(resolution=480, bitrate=1500),   # Standard quality
    HLSVariant(resolution=240, bitrate=800),    # Low quality
]
```

## Output Format

### Directory Structure
```
output_dir/
├── adaptive.m3u8               # Master playlist
├── adaptive-720p-3500.m3u8     # 720p variant playlist
├── adaptive-720p-3500_000.ts   # 720p segments
├── adaptive-720p-3500_001.ts
├── ...
├── adaptive-480p-1500.m3u8     # 480p variant playlist
├── adaptive-480p-1500_000.ts   # 480p segments
├── ...
└── adaptive-240p-800.m3u8      # 240p variant playlist
    └── ...
```

### Master Playlist (adaptive.m3u8)
```m3u8
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-STREAM-INF:BANDWIDTH=3500000,RESOLUTION=1280x720,CODECS="avc1.4d401f,mp4a.40.2"
adaptive-720p-3500.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=1500000,RESOLUTION=854x480,CODECS="avc1.4d401f,mp4a.40.2"
adaptive-480p-1500.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=426x240,CODECS="avc1.4d401f,mp4a.40.2"
adaptive-240p-800.m3u8
```

### Variant Playlist (adaptive-720p-3500.m3u8)
```m3u8
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:6
#EXT-X-MEDIA-SEQUENCE:0
#EXTINF:6.000000,
adaptive-720p-3500_000.ts
#EXTINF:6.000000,
adaptive-720p-3500_001.ts
#EXTINF:4.500000,
adaptive-720p-3500_002.ts
#EXT-X-ENDLIST
```

## Use Cases

### 1. Basic HLS Conversion
```python
from cl_ml_tools.algorithms import HLSStreamGenerator, HLSVariant

# Convert video to HLS with default variants (720p, 480p, 240p)
generator = HLSStreamGenerator(
    input_file="movie.mp4",
    output_dir="/hls_output"
)

# Generate all variants
generator.generate_all_variants()

print(f"Master playlist: {generator.master_pl_path}")
print(f"Generated {len(generator.variants)} variants")
```

### 2. Custom Quality Variants
```python
# Create custom variant configuration
variants = [
    HLSVariant(resolution=1080, bitrate=5000),  # Full HD
    HLSVariant(resolution=720, bitrate=2500),   # HD
    HLSVariant(resolution=480, bitrate=1200),   # SD
    HLSVariant(resolution=360, bitrate=800),    # Low SD
]

generator = HLSStreamGenerator("video.mp4", "/output")
for variant in variants:
    generator.generate_variant(variant)

generator.create_master_playlist()
```

### 3. Include Original Quality
```python
# Include original video without re-encoding (fastest)
variants = [
    HLSVariant(resolution=None, bitrate=None),  # Original (no transcode)
    HLSVariant(resolution=480, bitrate=1500),    # SD fallback
]

generator = HLSStreamGenerator("video.mp4", "/output")
generator.generate_all_variants(variants, include_original=True)
```

### 4. Video Platform Integration
```python
def process_upload(video_path, user_id):
    """Process uploaded video for streaming."""
    output_dir = f"/cdn/videos/{user_id}/{video_id}"

    # Generate HLS variants
    generator = HLSStreamGenerator(video_path, output_dir)
    generator.generate_all_variants()

    # Upload to CDN (example)
    cdn_url = upload_directory_to_cdn(output_dir)

    # Return streaming URL
    return f"{cdn_url}/adaptive.m3u8"

# HTML5 video player usage:
# <video>
#   <source src="https://cdn.example.com/videos/123/adaptive.m3u8" type="application/x-mpegURL">
# </video>
```

### 5. Batch Processing
```python
from pathlib import Path

def batch_convert_to_hls(input_dir, output_base_dir):
    """Convert all videos in directory to HLS."""
    for video in Path(input_dir).glob("*.mp4"):
        output_dir = Path(output_base_dir) / video.stem

        try:
            generator = HLSStreamGenerator(str(video), str(output_dir))
            generator.generate_all_variants()
            print(f"✓ Converted: {video.name}")
        except Exception as e:
            print(f"✗ Failed: {video.name} - {e}")

batch_convert_to_hls("/uploads", "/hls_videos")
```

## Edge Cases & Limitations

### Supported Scenarios
✅ All FFmpeg-supported video formats (MP4, MOV, AVI, MKV, WEBM, etc.)
✅ Various aspect ratios (16:9, 4:3, ultrawide, portrait)
✅ Videos with or without audio
✅ Long videos (hours)
✅ Already compressed videos

### Limitations

**FFmpeg Dependency:**
- Requires FFmpeg installed on system
- Must have H.264 (libx264) codec support
- FFmpeg version 4.0+ recommended

**Processing Time:**
- Real-time or slower (typically 0.5-2× video length)
- Multiple variants processed sequentially (not parallel)
- Longer videos take proportionally longer

**Disk Space:**
- Requires significant disk space (original + all variants)
- Example: 1 GB original → 400-600 MB for 3 variants
- Segments not cleaned up on failure

**Codec Support:**
- Output always uses H.264 (libx264) video codec
- Output always uses AAC audio codec
- No support for HEVC/H.265 or VP9

**Network Requirements (Playback):**
- Requires HTTP server or CDN to serve files
- Cannot play from local filesystem in browsers (security)
- CORS headers required for cross-origin playback

### Error Handling

```python
from cl_ml_tools.plugins.hls_streaming.algo.hls_stream_generator import NotFound, InternalServerError

try:
    generator = HLSStreamGenerator("video.mp4", "/output")
    generator.generate_all_variants()
except NotFound:
    print("Input video file not found")
except InternalServerError as e:
    print(f"FFmpeg transcode failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Optimization

### Faster Encoding (Lower Quality)
```python
# Use faster x264 preset (lower quality but faster)
# Note: Requires modifying FFmpeg command in source code
# Presets: ultrafast, superfast, veryfast, faster, fast, medium (default), slow, slower, veryslow
```

### Parallel Variant Generation
```python
from concurrent.futures import ProcessPoolExecutor

def generate_variant_parallel(args):
    generator, variant = args
    generator.generate_variant(variant)

variants = [
    HLSVariant(resolution=720, bitrate=3500),
    HLSVariant(resolution=480, bitrate=1500),
    HLSVariant(resolution=240, bitrate=800),
]

with ProcessPoolExecutor(max_workers=3) as executor:
    generator = HLSStreamGenerator("video.mp4", "/output")
    tasks = [(generator, v) for v in variants]
    executor.map(generate_variant_parallel, tasks)
```

### Skip Re-encoding for Same Resolution
```python
# If source is already 720p H.264, include as original
generator = HLSStreamGenerator("720p_video.mp4", "/output")
variants = [
    HLSVariant(resolution=None, bitrate=None),  # Original (copy, no transcode)
    HLSVariant(resolution=480, bitrate=1500),
    HLSVariant(resolution=240, bitrate=800),
]
generator.generate_all_variants(variants, include_original=True)
```

## Bitrate Recommendations

### By Resolution

| Resolution | Bitrate (kbps) | Use Case |
|------------|---------------|----------|
| **1080p** | 4000-6000 | Full HD, high-quality streaming |
| **720p** | 2500-4000 | HD, standard high-quality |
| **480p** | 1000-2000 | SD, mobile/slower connections |
| **360p** | 600-1200 | Low SD, very slow connections |
| **240p** | 400-800 | Minimal quality, 3G networks |
| **144p** | 200-400 | Emergency fallback |

### By Use Case

| Use Case | Recommended Variants |
|----------|---------------------|
| **YouTube-like** | 1080p (5000), 720p (2500), 480p (1200), 360p (800) |
| **Mobile-first** | 720p (2500), 480p (1200), 240p (600) |
| **Bandwidth-limited** | 480p (1000), 360p (700), 240p (400) |
| **4K streaming** | 2160p (15000), 1080p (5000), 720p (2500), 480p (1200) |

## HTML5 Video Player Integration

### HLS.js (Recommended)
```html
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<video id="video" controls></video>
<script>
  const video = document.getElementById('video');
  const hls = new Hls();
  hls.loadSource('https://cdn.example.com/videos/adaptive.m3u8');
  hls.attachMedia(video);
</script>
```

### Native Support (Safari, iOS)
```html
<video controls>
  <source src="https://cdn.example.com/videos/adaptive.m3u8" type="application/x-mpegURL">
</video>
```

### Video.js with HLS Plugin
```html
<link href="https://vjs.zencdn.net/7.20.3/video-js.css" rel="stylesheet" />
<video id="my-video" class="video-js" controls preload="auto">
  <source src="https://cdn.example.com/videos/adaptive.m3u8" type="application/x-mpegURL">
</video>
<script src="https://vjs.zencdn.net/7.20.3/video.min.js"></script>
<script src="https://unpkg.com/@videojs/http-streaming@2.14.3/dist/videojs-http-streaming.min.js"></script>
```

## Validation

```python
from cl_ml_tools.algorithms import validate_hls_directory

# Validate generated HLS output
is_valid, errors = validate_hls_directory("/hls_output")

if is_valid:
    print("✓ HLS directory is valid")
else:
    print("✗ Validation errors:")
    for error in errors:
        print(f"  - {error}")
```

## References

- **HLS Specification:** [RFC 8216](https://datatracker.ietf.org/doc/html/rfc8216)
- **Apple HLS Authoring:** [https://developer.apple.com/documentation/http_live_streaming](https://developer.apple.com/documentation/http_live_streaming)
- **FFmpeg HLS Documentation:** [https://ffmpeg.org/ffmpeg-formats.html#hls-2](https://ffmpeg.org/ffmpeg-formats.html#hls-2)
- **HLS.js Player:** [https://github.com/video-dev/hls.js/](https://github.com/video-dev/hls.js/)
- **M3U8 Specification:** [https://en.wikipedia.org/wiki/M3U](https://en.wikipedia.org/wiki/M3U)

## Version History

- **v0.2.1:** Current implementation with improved logging
- **v0.2.0:** Initial HLS streaming support

## Support

For issues or questions:
- Check tests: `tests/plugins/test_hls_streaming.py`
- Review source: `src/cl_ml_tools/plugins/hls_streaming/`
- FFmpeg docs: [https://ffmpeg.org/documentation.html](https://ffmpeg.org/documentation.html)
- File issues: [GitHub repository](https://github.com/your-org/cl_ml_tools/issues)
