# Face Embedding Plugin

Generates face embeddings (feature vectors) using ONNX-based ArcFace model.

## Model

**Source**: [ArcFace ONNX on Hugging Face](https://huggingface.co/garavv/arcface-onnx)
- **Model**: ArcFace (ResNet-based)
- **Input**: 112x112 RGB face images (cropped faces)
- **Output**: 512-dimensional embedding vector
- **Auto-download**: Model downloads automatically from Hugging Face on first use

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | UploadFile | Yes | - | Cropped face image file |
| `normalize` | bool | No | `True` | Whether to L2-normalize the embedding vector |
| `priority` | int | No | `5` | Job priority (0-10) |

## API Endpoint

```
POST /jobs/face_embedding
```

### Example Request

```bash
curl -X POST "http://localhost:8000/api/jobs/face_embedding" \
  -F "file=@cropped_face.jpg" \
  -F "normalize=true"
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
      "file_path": "/path/to/cropped_face.jpg",
      "embedding": {
        "embedding": [0.123, -0.456, 0.789, ...],
        "embedding_dim": 512,
        "quality_score": 0.85
      },
      "status": "success"
    }
  ],
  "total_files": 1,
  "normalize": true
}
```

## Embedding Format

- **Dimension**: 512D vector
- **Type**: Float32 array
- **Normalization**: L2-normalized (unit length) if `normalize=True`
- **Quality Score**: 0.0-1.0 range (based on image sharpness/blur)

## Use Cases

**Face Recognition**:
```python
# Compare two face embeddings using cosine similarity
from numpy import dot
from numpy.linalg import norm

similarity = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
# If normalized: similarity = dot(embedding1, embedding2)
```

**Face Clustering**:
- Group similar faces by embedding distance
- Use k-means, DBSCAN, or hierarchical clustering

**Face Search**:
- Build vector database (FAISS, Milvus, Qdrant)
- Search for similar faces by nearest neighbor

## Input Requirements

**IMPORTANT**: Input images should be **cropped faces**:
- Use face detection first to crop faces from images
- Recommended: 112x112 or larger face crops
- Face should be centered and well-lit

## Quality Score

The quality score indicates face image quality:
- **>0.7**: Good quality (sharp, well-lit)
- **0.4-0.7**: Moderate quality
- **<0.4**: Poor quality (blurry, low-resolution)

Based on Laplacian variance for blur detection. Requires `scipy` dependency.

## Dependencies

- `onnxruntime>=1.16.0`
- `numpy>=1.24.0`
- `Pillow>=10.0`
- `scipy>=1.10.0` (for quality score computation)

Installed automatically with `cl_ml_tools`.

## Model Cache Location

Downloaded models are cached at:
```
~/.cache/cl_ml_tools/models/arcface_face_embedding.onnx
```

## Notes

- Model downloads automatically on first use
- Optimized for CPU inference
- Best results with cropped, aligned faces
- L2 normalization recommended for face recognition tasks
