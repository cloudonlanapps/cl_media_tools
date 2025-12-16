# DINOv2 Embedding Plugin

Generate high-quality visual embeddings for image similarity search, clustering, and retrieval using Meta's DINOv2 vision transformer model.

## Overview

DINOv2 (Self-Distillation with No Labels v2) is a state-of-the-art self-supervised vision transformer that produces robust image embeddings. This plugin uses the ONNX-optimized `dinov2_vits14` model to generate 384-dimensional CLS token embeddings.

**Key Features:**
- **Self-supervised learning**: No reliance on labeled data
- **Robust representations**: Works well across diverse image types
- **CPU-optimized**: ONNX Runtime with CPU execution provider
- **L2 normalization**: Optional normalization for cosine similarity

## Use Cases

- **Visual similarity search**: Find similar images using cosine similarity
- **Image clustering**: Group images by visual content
- **Duplicate detection**: Identify near-duplicate images
- **Content-based retrieval**: Search images by visual features
- **Feature extraction**: Use embeddings as input for downstream ML tasks

## Model Information

- **Model**: DINOv2 ViT-S/14 (Vision Transformer Small, 14x14 patch size)
- **Source**: [RoundtTble/dinov2_vits14_onnx](https://huggingface.co/RoundtTble/dinov2_vits14_onnx)
- **Input**: 224x224 RGB images
- **Output**: 384-dimensional CLS token embedding
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **License**: Apache 2.0 (check model source for details)

## Parameters

### `DinoEmbeddingParams`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_paths` | `list[str]` | Required | List of image file paths to process |
| `output_paths` | `list[str]` | `[]` | Not used (embedding returned in task output) |
| `normalize` | `bool` | `true` | Whether to L2-normalize embedding vectors |

## API Usage

### Create DINOv2 Embedding Job

```bash
curl -X POST "http://localhost:8000/jobs/dino_embedding" \
  -F "file=@/path/to/image.jpg" \
  -F "normalize=true" \
  -F "priority=5"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

### Retrieve Results

Query the job repository using the `job_id` to get results:

```json
{
  "status": "ok",
  "task_output": {
    "files": [
      {
        "file_path": "/path/to/image.jpg",
        "embedding": {
          "embedding": [0.0234, -0.0156, 0.0891, ...],  // 384 float values
          "embedding_dim": 384,
          "normalized": true
        },
        "status": "success"
      }
    ],
    "total_files": 1,
    "normalize": true
  }
}
```

## Output Schema

### `DinoEmbeddingResult`

| Field | Type | Description |
|-------|------|-------------|
| `file_path` | `str` | Path to the input image |
| `embedding` | `DinoEmbedding \| None` | Embedding object (null if error) |
| `status` | `"success" \| "error"` | Processing status |
| `error` | `str \| None` | Error message if status is "error" |

### `DinoEmbedding`

| Field | Type | Description |
|-------|------|-------------|
| `embedding` | `list[float]` | 384-dimensional embedding vector |
| `embedding_dim` | `int` | Dimensionality (always 384) |
| `normalized` | `bool` | Whether embedding is L2-normalized |

## Computing Similarity

### Cosine Similarity (Recommended for Normalized Embeddings)

```python
import numpy as np

def cosine_similarity(emb1: list[float], emb2: list[float]) -> float:
    """Compute cosine similarity between two normalized embeddings."""
    a = np.array(emb1)
    b = np.array(emb2)
    return float(np.dot(a, b))  # If normalized, dot product = cosine similarity
```

### Euclidean Distance

```python
def euclidean_distance(emb1: list[float], emb2: list[float]) -> float:
    """Compute Euclidean L2 distance between embeddings."""
    a = np.array(emb1)
    b = np.array(emb2)
    return float(np.linalg.norm(a - b))
```

## Example Workflow

### 1. Generate Embeddings for Image Collection

```python
import httpx

async def generate_embedding(image_path: str) -> dict:
    async with httpx.AsyncClient() as client:
        with open(image_path, "rb") as f:
            response = await client.post(
                "http://localhost:8000/jobs/dino_embedding",
                files={"file": f},
                data={"normalize": "true", "priority": "5"}
            )
        return response.json()

# Generate embeddings for all images
job_ids = []
for image_path in image_paths:
    result = await generate_embedding(image_path)
    job_ids.append(result["job_id"])
```

### 2. Retrieve and Store Embeddings

```python
# Wait for jobs to complete and retrieve embeddings
embeddings_db = {}
for job_id in job_ids:
    result = get_job_result(job_id)  # Your job retrieval logic
    if result["status"] == "ok":
        for file_result in result["task_output"]["files"]:
            if file_result["status"] == "success":
                embeddings_db[file_result["file_path"]] = file_result["embedding"]["embedding"]
```

### 3. Find Similar Images

```python
import numpy as np

def find_similar_images(query_embedding: list[float], embeddings_db: dict, top_k: int = 5):
    """Find top K most similar images using cosine similarity."""
    query = np.array(query_embedding)

    similarities = []
    for image_path, embedding in embeddings_db.items():
        emb = np.array(embedding)
        similarity = np.dot(query, emb)  # Cosine similarity (if normalized)
        similarities.append((image_path, similarity))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Query with a new image
query_embedding = [...]  # From DINOv2 job result
similar_images = find_similar_images(query_embedding, embeddings_db, top_k=10)

for image_path, similarity in similar_images:
    print(f"{image_path}: {similarity:.4f}")
```

## Dependencies

- `numpy>=1.24.0`
- `onnx>=1.15.0`
- `onnxruntime>=1.16.0`
- `Pillow>=10.0`
- `httpx>=0.24` (for model download)

## Performance Considerations

- **Model size**: ~21 MB (ONNX FP32)
- **Inference time**: ~50-200ms per image on modern CPU (varies by hardware)
- **Memory**: Model loaded once per worker process (lazy initialization)
- **Batch processing**: Supported via multiple `input_paths`

## Troubleshooting

### Model Download Fails

**Error**: `Failed to download DINOv2 model`

**Solution**: Ensure internet connectivity and check Hugging Face availability. The model is cached at `~/.cache/cl_ml_tools/models/dinov2_vits14.onnx` after first download.

### Low-Quality Embeddings

**Issue**: Similar images have low cosine similarity scores

**Solutions**:
- Ensure `normalize=true` for cosine similarity comparisons
- Check input image quality (avoid heavily compressed or corrupted images)
- DINOv2 works best on natural images; may perform worse on highly synthetic/abstract content

### Out of Memory

**Issue**: Worker crashes during embedding generation

**Solutions**:
- Reduce concurrent workers
- Process smaller batches of images
- Monitor system memory usage

## References

- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [DINOv2 ONNX Model](https://huggingface.co/RoundtTble/dinov2_vits14_onnx)
- [Meta AI DINOv2 GitHub](https://github.com/facebookresearch/dinov2)
