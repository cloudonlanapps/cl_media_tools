# Face Detection Algorithm

## Overview

Detects faces in images using Google's MediaPipe Face Detection model. Outputs bounding box coordinates and confidence scores for each detected face. Includes Non-Maximum Suppression (NMS) to eliminate duplicate detections and supports configurable confidence thresholds.

## Algorithm Details

### Model

- **Model:** MediaPipe Face Detection (ONNX)
- **Architecture:** Lightweight CNN-based detector optimized for mobile
- **Input Size:** 192×192 RGB
- **Output:** Bounding boxes + confidence scores
- **Source:** [https://huggingface.co/qualcomm/MediaPipe-Face-Detection](https://huggingface.co/qualcomm/MediaPipe-Face-Detection)
- **Original:** Google MediaPipe
- **License:** Apache 2.0
- **Model Size:** ~2.45 MB

### Key Features

**Lightweight & Fast:**
- Optimized for real-time applications
- Small model size (~2.45 MB)
- Fast inference (~20-50ms per image)

**Robust Detection:**
- Works with various face orientations
- Handles partial occlusions
- Detects faces at different scales

**NMS Processing:**
- Removes overlapping duplicate detections
- Configurable IoU threshold
- Keeps highest-confidence detections

### Implementation

#### 1. Preprocessing
```python
# Convert to RGB
if image.mode != "RGB":
    image = image.convert("RGB")

# Resize to 192×192 using BILINEAR resampling
image = image.resize((192, 192), Image.Resampling.BILINEAR)

# Normalize to [0, 1]
img_array = np.asarray(image, dtype=np.float32) / 255.0

# Transpose to CHW format and add batch dimension
img_array = np.transpose(img_array, (2, 0, 1))
img_array = np.expand_dims(img_array, axis=0)
```

#### 2. Inference
- ONNX Runtime with CPU execution provider
- Graph optimization enabled
- Outputs:
  - Bounding boxes: [batch, num_boxes, 4] (normalized coordinates)
  - Confidence scores: [batch, num_boxes] or [batch, num_boxes, 1]

#### 3. Postprocessing
```python
# 1. Extract boxes and scores from model outputs
boxes = outputs[0]  # [num_boxes, 4]
scores = outputs[1]  # [num_boxes]

# 2. Filter by confidence threshold
mask = scores >= confidence_threshold
boxes = boxes[mask]
scores = scores[mask]

# 3. Apply Non-Maximum Suppression
keep_indices = nms(boxes, scores, nms_threshold)
boxes = boxes[keep_indices]
scores = scores[keep_indices]

# 4. Convert from normalized to absolute coordinates
# MediaPipe format: [x_center, y_center, width, height] (normalized)
for box, score in zip(boxes, scores):
    x_center, y_center, width, height = box
    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height
```

### Non-Maximum Suppression (NMS)

**Purpose:** Remove duplicate/overlapping bounding boxes

**Algorithm:**
1. Sort boxes by confidence score (descending)
2. Keep box with highest score
3. Calculate IoU (Intersection over Union) with remaining boxes
4. Remove boxes with IoU > nms_threshold
5. Repeat until all boxes processed

**IoU Calculation:**
```python
intersection_area = overlap_width × overlap_height
union_area = box1_area + box2_area - intersection_area
iou = intersection_area / union_area
```

**Example:**
```
Two overlapping boxes for same face:
- Box A: confidence=0.95, IoU with B=0.6
- Box B: confidence=0.85

With nms_threshold=0.4:
  IoU (0.6) > threshold (0.4) → Remove Box B, keep Box A
```

### Performance Characteristics

- **Memory:** ~10MB (model) + ~3MB (processing buffer)
- **Speed:**
  - Raspberry Pi 5 (4 cores): ~80ms per image
  - Desktop CPU (8 cores): ~20ms per image
  - With Hailo 8 AI Hat+: TBD (future optimization)
- **Model Size:** ~2.45 MB download
- **Disk Cache:** Models cached in `~/.cache/cl_ml_tools/models/`

## Parameters

### FaceDetector.__init__()
```python
def __init__(self, model_path: str | Path | None = None)
```

**Parameters:**
- `model_path` (str | Path | None): Custom model path, or None to auto-download
  - If None: Downloads from HuggingFace to cache directory
  - If specified: Loads model from given path
  - Raises FileNotFoundError if path doesn't exist

### FaceDetector.detect()
```python
def detect(
    self,
    image_path: str | Path,
    confidence_threshold: float = 0.7,
    nms_threshold: float = 0.4,
) -> list[FaceDetection]
```

**Parameters:**
- `image_path` (str | Path): Path to input image
  - Supported formats: JPEG, PNG, WEBP, HEIF, BMP, GIF, TIFF
- `confidence_threshold` (float): Minimum confidence for detections (default: 0.7)
  - Range: 0.0-1.0
  - Higher = fewer but more confident detections
  - Lower = more detections but may include false positives
  - Recommended: 0.5-0.8
- `nms_threshold` (float): IoU threshold for NMS (default: 0.4)
  - Range: 0.0-1.0
  - Higher = keep more overlapping boxes
  - Lower = more aggressive suppression
  - Recommended: 0.3-0.5

**Returns:**
- `list[FaceDetection]`: List of detected faces
  - Each detection is a TypedDict with:
    - `x1` (float): Left coordinate (pixels)
    - `y1` (float): Top coordinate (pixels)
    - `x2` (float): Right coordinate (pixels)
    - `y2` (float): Bottom coordinate (pixels)
    - `confidence` (float): Detection confidence score (0.0-1.0)

**Raises:**
- `FileNotFoundError`: If image_path doesn't exist
- `PIL.UnidentifiedImageError`: If image format unsupported
- `RuntimeError`: If model inference fails

## Output Format

Returns a list of `FaceDetection` dictionaries:

```python
[
    {
        "x1": 120.5,
        "y1": 80.3,
        "x2": 220.7,
        "y2": 180.9,
        "confidence": 0.95
    },
    {
        "x1": 450.2,
        "y1": 150.1,
        "x2": 550.8,
        "y2": 250.6,
        "confidence": 0.88
    }
]
```

**Bounding Box Format:**
- Coordinates in absolute pixels (not normalized)
- (x1, y1): Top-left corner
- (x2, y2): Bottom-right corner
- Width = x2 - x1
- Height = y2 - y1

**Empty List:**
- Returned when no faces detected above confidence threshold

## Use Cases

### 1. Simple Face Detection
```python
from cl_ml_tools.algorithms import FaceDetector

detector = FaceDetector()

# Detect faces with default settings
faces = detector.detect("group_photo.jpg")

print(f"Found {len(faces)} faces")
for i, face in enumerate(faces, 1):
    print(f"Face {i}: confidence={face['confidence']:.2f}, "
          f"bbox=({face['x1']:.0f},{face['y1']:.0f})-({face['x2']:.0f},{face['y2']:.0f})")

# Output:
# Found 3 faces
# Face 1: confidence=0.95, bbox=(120,80)-(220,180)
# Face 2: confidence=0.88, bbox=(450,150)-(550,250)
# Face 3: confidence=0.75, bbox=(680,200)-(780,300)
```

### 2. Face Cropping for Face Recognition
```python
from PIL import Image

def crop_faces(image_path, output_dir):
    """Detect and crop all faces from an image."""
    detector = FaceDetector()
    faces = detector.detect(image_path, confidence_threshold=0.7)

    with Image.open(image_path) as img:
        for i, face in enumerate(faces):
            # Crop face region
            face_img = img.crop((
                int(face['x1']),
                int(face['y1']),
                int(face['x2']),
                int(face['y2'])
            ))

            # Save cropped face
            output_path = Path(output_dir) / f"face_{i+1}.jpg"
            face_img.save(output_path)
            print(f"Saved: {output_path}")

crop_faces("group_photo.jpg", "/faces")
```

### 3. Face Counting and Statistics
```python
from pathlib import Path

def analyze_face_count(image_dir):
    """Count faces in all images in a directory."""
    detector = FaceDetector()
    results = {}

    for img_path in Path(image_dir).glob("*.jpg"):
        faces = detector.detect(img_path, confidence_threshold=0.6)
        results[img_path.name] = {
            "count": len(faces),
            "avg_confidence": sum(f['confidence'] for f in faces) / len(faces) if faces else 0,
            "max_confidence": max((f['confidence'] for f in faces), default=0)
        }

    # Print summary
    for filename, stats in results.items():
        print(f"{filename}: {stats['count']} faces "
              f"(avg conf: {stats['avg_confidence']:.2f})")

    total_faces = sum(r['count'] for r in results.values())
    print(f"\nTotal: {total_faces} faces across {len(results)} images")

analyze_face_count("/photos")
```

### 4. Face Blurring for Privacy
```python
from PIL import Image, ImageFilter

def blur_faces(image_path, output_path, blur_radius=30):
    """Detect and blur all faces in an image."""
    detector = FaceDetector()
    faces = detector.detect(image_path)

    with Image.open(image_path) as img:
        for face in faces:
            # Extract face region
            x1, y1, x2, y2 = int(face['x1']), int(face['y1']), int(face['x2']), int(face['y2'])
            face_region = img.crop((x1, y1, x2, y2))

            # Apply blur
            blurred = face_region.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Paste blurred region back
            img.paste(blurred, (x1, y1))

        img.save(output_path)
        print(f"Blurred {len(faces)} faces, saved to {output_path}")

blur_faces("photo.jpg", "photo_blurred.jpg")
```

### 5. Batch Face Detection with Confidence Filtering
```python
def find_images_with_multiple_faces(image_dir, min_faces=2, confidence=0.7):
    """Find all images containing at least N faces."""
    detector = FaceDetector()
    results = []

    for img_path in Path(image_dir).glob("*.jpg"):
        faces = detector.detect(img_path, confidence_threshold=confidence)

        if len(faces) >= min_faces:
            results.append((img_path, len(faces)))

    # Sort by number of faces (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"Found {len(results)} images with {min_faces}+ faces:")
    for path, count in results:
        print(f"  {path.name}: {count} faces")

    return results

# Find group photos (3+ people)
group_photos = find_images_with_multiple_faces("/photos", min_faces=3)
```

## Edge Cases & Limitations

### Supported Scenarios
✅ Multiple faces per image
✅ Faces at various scales (close-up to distant)
✅ Partial face occlusions (sunglasses, hats)
✅ Various face orientations (±45° rotation)
✅ Different lighting conditions

### Limitations

**Face Orientation:**
- Best with frontal or near-frontal faces (±45°)
- Profile faces (90° rotation) may not be detected
- Upside-down faces may be missed

**Face Size:**
- Very small faces (<20×20 pixels) may not be detected
- Very large faces (>80% of image) may be partially detected

**Occlusions:**
- Heavy occlusions (masks covering >50% of face) may fail
- Hands covering face may reduce detection confidence

**Environmental Factors:**
- Extreme low light may reduce accuracy
- Very high contrast (backlighting) may affect detection
- Blurry/out-of-focus faces may have lower confidence

**Model Limitations:**
- No age, gender, or emotion detection
- No facial landmark detection (eyes, nose, mouth positions)
- No face recognition (identity matching)
- For these, use face_embedding plugin after detection

### Error Handling

```python
from PIL import UnidentifiedImageError

try:
    faces = detector.detect("photo.jpg")
except FileNotFoundError:
    print("Image file not found")
except UnidentifiedImageError:
    print("Corrupted or unsupported image format")
except RuntimeError as e:
    print(f"Model inference error: {e}")

# Check if any faces found
if not faces:
    print("No faces detected (try lowering confidence_threshold)")
```

## Performance Optimization

### Parallel Processing
```python
from concurrent.futures import ProcessPoolExecutor

def detect_single(image_path):
    detector = FaceDetector()
    return detector.detect(image_path)

image_paths = list(Path("/photos").glob("*.jpg"))

with ProcessPoolExecutor(max_workers=4) as executor:
    all_faces = list(executor.map(detect_single, image_paths))

for path, faces in zip(image_paths, all_faces):
    print(f"{path.name}: {len(faces)} faces")
```

### Caching Detections
```python
import json
from pathlib import Path

def get_or_detect_faces(image_path, cache_dir, confidence=0.7):
    """Load cached detections or compute if not exists."""
    cache_file = Path(cache_dir) / f"{Path(image_path).stem}_faces.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    detector = FaceDetector()
    faces = detector.detect(image_path, confidence_threshold=confidence)

    with open(cache_file, "w") as f:
        json.dump(faces, f)

    return faces
```

## Confidence Threshold Recommendations

| Use Case | Recommended Threshold | Notes |
|----------|----------------------|-------|
| **General detection** | 0.7 (default) | Balanced precision/recall |
| **High precision** | 0.8-0.9 | Fewer false positives, may miss some faces |
| **High recall** | 0.5-0.6 | More detections, may include false positives |
| **Crowded scenes** | 0.6-0.7 | Lower threshold to catch distant/small faces |
| **Close-up portraits** | 0.7-0.8 | Higher threshold for cleaner results |
| **Face blurring (privacy)** | 0.5-0.6 | Lower to avoid missing any faces |

## NMS Threshold Recommendations

| Scenario | Recommended NMS Threshold | Effect |
|----------|--------------------------|--------|
| **Overlapping faces** | 0.3-0.4 (default) | Remove most duplicates |
| **Crowded scenes** | 0.2-0.3 | More aggressive, avoid duplicate boxes |
| **Sparse faces** | 0.4-0.5 | Less aggressive, allow some overlap |

## References

- **MediaPipe Face Detection:** [https://google.github.io/mediapipe/solutions/face_detection](https://google.github.io/mediapipe/solutions/face_detection)
- **HuggingFace Model:** [https://huggingface.co/qualcomm/MediaPipe-Face-Detection](https://huggingface.co/qualcomm/MediaPipe-Face-Detection)
- **ONNX Runtime:** [https://onnxruntime.ai/](https://onnxruntime.ai/)
- **Non-Maximum Suppression:** [https://en.wikipedia.org/wiki/Canny_edge_detector#Non-maximum_suppression](https://en.wikipedia.org/wiki/Canny_edge_detector#Non-maximum_suppression)

## Version History

- **v0.2.1:** Current implementation with complete NMS and postprocessing
- **v0.2.0:** Initial face detection support

## Support

For issues or questions:
- Check tests: `tests/plugins/test_face_detection.py`
- Review source: `src/cl_ml_tools/plugins/face_detection/`
- File issues: [GitHub repository](https://github.com/your-org/cl_ml_tools/issues)
