# MobileCLIP Embedding Plugin

Generate semantic image embeddings for text-image similarity search and multimodal retrieval using Apple's MobileCLIP model optimized for mobile/edge devices.

## Overview

MobileCLIP is a family of efficient CLIP-style vision-language models designed for resource-constrained environments. This plugin uses the ONNX-optimized image encoder from MobileCLIP-S2 to generate 512-dimensional image embeddings that can be compared with text embeddings for semantic search.

**Key Features:**
- **Semantic understanding**: Captures high-level concepts beyond visual similarity
- **Text-image alignment**: Embeddings compatible with CLIP text encoder
- **CPU-efficient**: Optimized for edge deployment
- **L2 normalization**: Optional normalization for cosine similarity

## Use Cases

- **Semantic image search**: Search images using natural language queries
- **Cross-modal retrieval**: Find images similar to text descriptions
- **Image classification**: Zero-shot classification using text prompts
- **Content moderation**: Detect concepts via text-image similarity
- **Multimodal clustering**: Group images and text by semantic meaning

## Model Information

- **Model**: MobileCLIP-S2 Image Encoder
- **Source**: [Apple ml-mobileclip](https://github.com/apple/ml-mobileclip)
- **Input**: 256x256 RGB images
- **Output**: 512-dimensional image embedding
- **Normalization**: CLIP statistics (mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
- **License**: Apple Sample Code License (check model source for details)

**Note**: This plugin requires an ONNX-converted model. The original MobileCLIP is distributed as PyTorch weights and requires conversion to ONNX format.

## Parameters

### `ClipEmbeddingParams`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_paths` | `list[str]` | Required | List of image file paths to process |
| `output_paths` | `list[str]` | `[]` | Not used (embedding returned in task output) |
| `normalize` | `bool` | `true` | Whether to L2-normalize embedding vectors |

## API Usage

### Create MobileCLIP Embedding Job

```bash
curl -X POST "http://localhost:8000/jobs/clip_embedding" \
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
          "embedding": [0.0156, -0.0234, 0.0678, ...],  // 512 float values
          "embedding_dim": 512,
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

### `ClipEmbeddingResult`

| Field | Type | Description |
|-------|------|-------------|
| `file_path` | `str` | Path to the input image |
| `embedding` | `ClipEmbedding \| None` | Embedding object (null if error) |
| `status` | `"success" \| "error"` | Processing status |
| `error` | `str \| None` | Error message if status is "error" |

### `ClipEmbedding`

| Field | Type | Description |
|-------|------|-------------|
| `embedding` | `list[float]` | 512-dimensional embedding vector |
| `embedding_dim` | `int` | Dimensionality (512 for MobileCLIP-S2) |
| `normalized` | `bool` | Whether embedding is L2-normalized |

## Text-Image Similarity (Semantic Search)

To perform text-image similarity, you'll need a CLIP text encoder to generate text embeddings. Then compute cosine similarity between text and image embeddings.

### Example: Zero-Shot Image Classification

```python
import numpy as np

# Image embedding from MobileCLIP job result
image_embedding = np.array([...])  # 512D vector

# Text embeddings (generated separately using CLIP text encoder)
text_embeddings = {
    "a photo of a cat": np.array([...]),  # 512D vector
    "a photo of a dog": np.array([...]),
    "a photo of a car": np.array([...]),
}

# Compute similarities
similarities = {}
for label, text_emb in text_embeddings.items():
    similarity = np.dot(image_embedding, text_emb)  # Cosine similarity (if normalized)
    similarities[label] = similarity

# Get prediction
predicted_label = max(similarities, key=similarities.get)
print(f"Predicted: {predicted_label} (score: {similarities[predicted_label]:.4f})")
```

### Example: Semantic Image Search

```python
# User query text embedding
query_text_embedding = np.array([...])  # "sunset over mountains"

# Database of image embeddings
image_db = {
    "img1.jpg": np.array([...]),
    "img2.jpg": np.array([...]),
    # ...
}

# Find images matching text query
results = []
for image_path, image_embedding in image_db.items():
    similarity = np.dot(query_text_embedding, image_embedding)
    results.append((image_path, similarity))

# Sort by similarity
results.sort(key=lambda x: x[1], reverse=True)

print("Top 5 matching images:")
for image_path, similarity in results[:5]:
    print(f"  {image_path}: {similarity:.4f}")
```

## Image-to-Image Similarity

You can also use MobileCLIP embeddings for pure image similarity (similar to DINOv2):

```python
def cosine_similarity(emb1: list[float], emb2: list[float]) -> float:
    """Compute cosine similarity between two normalized embeddings."""
    a = np.array(emb1)
    b = np.array(emb2)
    return float(np.dot(a, b))

# Compare two images
similarity = cosine_similarity(image_emb1, image_emb2)
```

## ONNX Model Conversion

If you need to convert the MobileCLIP PyTorch model to ONNX:

### 1. Install Dependencies

```bash
pip install torch torchvision onnx
```

### 2. Export to ONNX

```python
import torch
import torch.onnx
from mobileclip import get_mobileclip  # From Apple's repo

# Load MobileCLIP model
model, preprocess = get_mobileclip('mobileclip_s2')
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 256, 256)

# Export image encoder to ONNX
torch.onnx.export(
    model.visual,  # Image encoder only
    dummy_input,
    "mobileclip_s2_image_encoder.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=14
)

print("ONNX export complete: mobileclip_s2_image_encoder.onnx")
```

### 3. Verify ONNX Model

```python
import onnx
import onnxruntime as ort

# Check model
model = onnx.load("mobileclip_s2_image_encoder.onnx")
onnx.checker.check_model(model)

# Test inference
session = ort.InferenceSession("mobileclip_s2_image_encoder.onnx")
test_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
output = session.run(None, {"input": test_input})
print(f"Output shape: {output[0].shape}")  # Should be (1, 512)
```

## Dependencies

- `numpy>=1.24.0`
- `onnx>=1.15.0`
- `onnxruntime>=1.16.0`
- `Pillow>=10.0`
- `httpx>=0.24` (for model download)

## Performance Considerations

- **Model size**: ~16 MB (ONNX FP32 for MobileCLIP-S2)
- **Inference time**: ~30-100ms per image on modern CPU
- **Memory**: Model loaded once per worker process (lazy initialization)
- **Batch processing**: Supported via multiple `input_paths`
- **Edge-optimized**: Designed for mobile/edge deployment

## Comparison: MobileCLIP vs DINOv2

| Feature | MobileCLIP | DINOv2 |
|---------|------------|--------|
| **Training** | Contrastive (text-image) | Self-supervised (image-only) |
| **Semantics** | High-level concepts, language-aligned | Visual features, structural patterns |
| **Best for** | Text-image search, zero-shot tasks | Visual similarity, clustering |
| **Embedding size** | 512D (S2) | 384D (ViT-S/14) |
| **Input size** | 256x256 | 224x224 |
| **Model size** | ~16 MB | ~21 MB |
| **Speed** | Faster (optimized for mobile) | Moderate |

**Recommendation**: Use MobileCLIP for semantic/conceptual search with text queries. Use DINOv2 for pure visual similarity and fine-grained feature matching.

## Troubleshooting

### Model Not Found

**Error**: `Failed to download MobileCLIP model`

**Solution**: The MobileCLIP ONNX model may not be pre-converted. Follow the "ONNX Model Conversion" section to create the model file, then place it at `~/.cache/cl_ml_tools/models/mobileclip_s2_image_encoder.onnx` or update the `MODEL_URL` in `clip_embedder.py`.

### Text Encoder Not Available

**Note**: This plugin only provides the **image encoder**. For text-image similarity, you'll need to deploy a separate CLIP text encoder (e.g., via a Python service using Hugging Face Transformers or ONNX).

### Low Text-Image Similarity

**Issue**: Expected matches have low similarity scores

**Solutions**:
- Ensure both text and image embeddings are L2-normalized
- Verify text encoder is compatible with MobileCLIP (use MobileCLIP text encoder, not OpenAI CLIP)
- Check text prompt phrasing (e.g., "a photo of a [object]" often works better than single words)

## References

- [MobileCLIP Paper](https://arxiv.org/abs/2311.17049)
- [Apple ml-mobileclip GitHub](https://github.com/apple/ml-mobileclip)
- [CLIP Paper (Original)](https://arxiv.org/abs/2103.00020)
