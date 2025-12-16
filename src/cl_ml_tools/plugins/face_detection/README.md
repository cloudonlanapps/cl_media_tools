# Face Detection Plugin

Detects faces in images using ONNX-based MediaPipe Face Detection model.

## Model

**Source**: [MediaPipe Face Detection on Hugging Face](https://huggingface.co/qualcomm/MediaPipe-Face-Detection)
- **Model File**: MediaPipeFaceLandmarkDetector.onnx (2.45 MB)
- **Input**: 192x192 RGB images
- **Output**: Face detections with normalized bounding boxes
- **Auto-download**: Model downloads automatically from Hugging Face on first use

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | UploadFile | Yes | - | Image file to detect faces in |
| `confidence_threshold` | float | No | `0.7` | Minimum confidence score for detections (0.0-1.0) |
| `nms_threshold` | float | No | `0.4` | Non-maximum suppression threshold for overlapping boxes (0.0-1.0) |
| `priority` | int | No | `5` | Job priority (0-10) |

## API Endpoint

```
POST /jobs/face_detection
```

### Example Request

```bash
curl -X POST "http://localhost:8000/api/jobs/face_detection" \
  -F "file=@photo.jpg" \
  -F "confidence_threshold=0.7" \
  -F "nms_threshold=0.4"
```

### Example Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

## Task Output

```json
{
  "files": [
    {
      "file_path": "/path/to/photo.jpg",
      "status": "success",
      "detection": {
        "file_path": "/path/to/photo.jpg",
        "faces": [
          {
            "x1": 0.25,
            "y1": 0.30,
            "x2": 0.60,
            "y2": 0.75,
            "confidence": 0.95
          }
        ],
        "num_faces": 1,
        "image_width": 1920,
        "image_height": 1080
      }
    }
  ],
  "total_files": 1,
  "confidence_threshold": 0.7,
  "nms_threshold": 0.4
}
```

## Bounding Box Coordinates

Coordinates are **normalized** to [0.0, 1.0] range:
- `x1, y1`: Top-left corner (normalized)
- `x2, y2`: Bottom-right corner (normalized)

To convert to absolute pixel coordinates:
```python
abs_x1 = x1 * image_width
abs_y1 = y1 * image_height
abs_x2 = x2 * image_width
abs_y2 = y2 * image_height
```

## Dependencies

- `onnxruntime>=1.16.0`
- `numpy>=1.24.0`
- `Pillow>=10.0`

Installed automatically with `cl_ml_tools`.

## Model Cache Location

Downloaded models are cached at:
```
~/.cache/cl_ml_tools/models/mediapipe_face_detection.onnx
```

## Notes

- Model downloads automatically on first use (~2.5 MB)
- Optimized for CPU inference
- Post-processing implementation may need refinement based on specific ONNX model output format
