# DINO Embedding Algorithm

## Overview

Generates image embeddings using Meta AI's DINOv2 model, a self-supervised vision transformer trained without labels. Produces 384-dimensional normalized feature vectors suitable for semantic image search, similarity comparison, and visual clustering. Particularly strong at capturing object-centric and geometric features.

## Algorithm Details

### Model

- **Model:** DINOv2-ViT-S/14 (ONNX)
- **Architecture:** Vision Transformer Small with patch size 14
- **Input Size:** 224×224 RGB
- **Output:** 384-dimensional float32 vector (CLS token)
- **Source:** [https://huggingface.co/RoundtTble/dinov2_vits14_onnx](https://huggingface.co/RoundtTble/dinov2_vits14_onnx)
- **Original:** Meta AI DINOv2
- **License:** Apache 2.0

### Key Characteristics

**Self-Supervised Learning:**
- Trained without labels using DINO (self-distillation with no labels)
- Learns visual features from image structure alone
- Strong semantic understanding without explicit supervision

**CLS Token Embedding:**
- Uses [CLS] token from transformer output
- Global image representation (not patch-level)
- Captures high-level semantic content

**vs CLIP:**
- DINO: Visual features only (no text alignment)
- CLIP: Image-text aligned features
- DINO: Better for pure visual similarity
- CLIP: Better for text-to-image search

### Implementation

The implementation follows a standard embedding pipeline:

#### 1. Preprocessing
```python
# Convert to RGB (handles RGBA, grayscale, L mode)
if image.mode != "RGB":
    image = image.convert("RGB")

# Resize to 224×224 using BILINEAR resampling
image = image.resize((224, 224), Image.Resampling.BILINEAR)

# Normalize pixel values with ImageNet statistics
img_array = np.asarray(image, dtype=np.float32) / 255.0
img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

# ImageNet normalization constants:
# Mean: [0.485, 0.456, 0.406]  (R, G, B)
# Std:  [0.229, 0.224, 0.225]  (R, G, B)

# Transpose to CHW format (channels-first) and add batch dimension
img_array = np.transpose(img_array, (2, 0, 1))
img_array = np.expand_dims(img_array, axis=0)
```

#### 2. Inference
- ONNX Runtime with CPU execution provider
- Graph optimization enabled (ORT_ENABLE_ALL)
- Single forward pass through vision transformer
- Outputs sequence of patch embeddings + CLS token

#### 3. Postprocessing
```python
# Remove batch dimension
embedding = np.squeeze(embedding)

# Extract CLS token (first token in sequence)
if embedding.ndim > 1:
    embedding = embedding[0]  # CLS token embedding

# L2 normalization (optional, default=True)
if normalize:
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
```

### Performance Characteristics

- **Memory:** ~30MB (model) + ~5MB (processing buffer)
- **Speed:**
  - Raspberry Pi 5 (4 cores): ~180ms per image
  - Desktop CPU (8 cores): ~40ms per image
  - With Hailo 8 AI Hat+: TBD (future optimization)
- **Model Size:** ~22MB download
- **Disk Cache:** Models cached in `~/.cache/cl_ml_tools/models/`

### Technical Details

**Model Architecture:**
- Vision Transformer (ViT) with patch size 14
- Input: 224×224 → 16×16 patches
- Embedding dimension: 384
- Number of layers: 12 (Small variant)
- Attention heads: 6

**Training Approach:**
- Self-distillation with no labels (DINO)
- Trained on ImageNet-1K without using labels
- Teacher-student framework
- Learns visual representations from structure

**Optimization Techniques:**
- Graph-level optimizations via ONNX Runtime
- Efficient memory layout (CHW format)
- BILINEAR resampling for speed

## Parameters

### DinoEmbedder.__init__()
```python
def __init__(self, model_path: str | Path | None = None)
```

**Parameters:**
- `model_path` (str | Path | None): Custom model path, or None to auto-download
  - If None: Downloads from HuggingFace to cache directory
  - If specified: Loads model from given path
  - Raises FileNotFoundError if path doesn't exist

### DinoEmbedder.embed()
```python
def embed(
    self,
    image_path: str | Path,
    normalize: bool = True,
) -> NDArray[np.float32]
```

**Parameters:**
- `image_path` (str | Path): Path to input image
  - Supported formats: JPEG, PNG, WEBP, HEIF, BMP, GIF, TIFF
  - Any format supported by PIL/Pillow
- `normalize` (bool): Apply L2 normalization (default: True)
  - True: Output has L2 norm = 1.0 (unit vector)
  - False: Raw model output values

**Returns:**
- `NDArray[np.float32]`: Embedding vector of shape (384,)

**Raises:**
- `FileNotFoundError`: If image_path doesn't exist
- `PIL.UnidentifiedImageError`: If image format unsupported
- `RuntimeError`: If model inference fails

## Output Format

Returns a NumPy array of 384 float32 values representing the image embedding.

### Normalized Embeddings (default):
- Shape: `(384,)`
- Data type: `np.float32`
- Value range: approximately [-1, 1]
- L2 norm: exactly 1.0 (unit vector)
- **Usage:** Directly compute cosine similarity via dot product:
  ```python
  similarity = np.dot(embedding1, embedding2)
  # Result in [-1, 1]: 1 = identical, -1 = opposite, 0 = orthogonal
  ```

### Unnormalized Embeddings:
- Shape: `(384,)`
- Data type: `np.float32`
- Value range: varies (typically [-5, 5])
- **Usage:** Custom normalization or analysis of raw features

## Use Cases

### 1. Visual Similarity Search
```python
from cl_ml_tools.algorithms import DinoEmbedder
import numpy as np

embedder = DinoEmbedder()

# Index images
image_embeddings = {}
for image_path in image_database:
    embedding = embedder.embed(image_path)
    image_embeddings[image_path] = embedding

# Search with query image
query_embedding = embedder.embed("query.jpg")
similarities = {
    path: np.dot(query_embedding, emb)
    for path, emb in image_embeddings.items()
}
top_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]
```

### 2. Image Clustering
```python
from sklearn.cluster import KMeans
import numpy as np

# Generate embeddings for all images
embeddings = np.array([embedder.embed(img) for img in images])

# Cluster by visual similarity
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Group images by cluster
for cluster_id in range(10):
    cluster_images = [img for img, c in zip(images, clusters) if c == cluster_id]
    print(f"Cluster {cluster_id}: {len(cluster_images)} images")
```

### 3. Duplicate/Near-Duplicate Detection
```python
def find_duplicates(image_paths, threshold=0.98):
    """Find near-duplicate images using DINO embeddings."""
    embedder = DinoEmbedder()

    embeddings = {path: embedder.embed(path) for path in image_paths}
    duplicates = []

    paths_list = list(image_paths)
    for i, path1 in enumerate(paths_list):
        for path2 in paths_list[i+1:]:
            similarity = np.dot(embeddings[path1], embeddings[path2])
            if similarity > threshold:
                duplicates.append((path1, path2, similarity))

    return duplicates

# Find near-duplicates
dupes = find_duplicates(my_images, threshold=0.98)
for img1, img2, sim in dupes:
    print(f"Duplicate: {img1} ↔ {img2} (similarity: {sim:.3f})")
```

### 4. Image Retrieval by Example
```python
def retrieve_similar(query_image, database_images, top_k=5):
    """Retrieve most similar images to query."""
    embedder = DinoEmbedder()

    query_emb = embedder.embed(query_image)
    db_embeddings = [(img, embedder.embed(img)) for img in database_images]

    # Compute similarities
    similarities = [
        (img, np.dot(query_emb, emb))
        for img, emb in db_embeddings
    ]

    # Sort by similarity (descending)
    results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    return results

# Find similar product images
similar = retrieve_similar("query_product.jpg", product_catalog, top_k=10)
```

### 5. Visual Content Verification
```python
def verify_image_match(img1_path, img2_path, threshold=0.95):
    """Verify if two images show the same content."""
    embedder = DinoEmbedder()

    emb1 = embedder.embed(img1_path)
    emb2 = embedder.embed(img2_path)

    similarity = np.dot(emb1, emb2)

    if similarity > threshold:
        return True, f"Match (similarity: {similarity:.3f})"
    else:
        return False, f"No match (similarity: {similarity:.3f})"

# Check if two photos show the same object
is_match, msg = verify_image_match("photo1.jpg", "photo2.jpg")
print(msg)
```

## Edge Cases & Limitations

### Supported Formats
✅ RGB, RGBA, grayscale, L mode images
✅ All PIL-supported formats (JPEG, PNG, WEBP, HEIF, BMP, GIF, TIFF)
✅ Images of any size (automatically resized)

### Limitations

**Fixed Input Size:**
- 224×224 - images resized, may lose fine details
- Smaller than CLIP (256×256)

**No Batch Processing:**
- Images processed sequentially
- Use multiprocessing for parallelism

**Visual Features Only:**
- No text understanding (unlike CLIP)
- Cannot search with text queries
- Purely visual similarity

**Small Variant:**
- 384-dim embeddings (vs CLIP's 512)
- Smaller model = faster, but slightly less expressive
- Larger variants (Base, Large, Giant) available but not included

**Color Dependency:**
- Grayscale images converted to RGB
- May affect embedding quality for truly grayscale content

### Error Handling

```python
from PIL import UnidentifiedImageError

try:
    embedding = embedder.embed("image.jpg")
except FileNotFoundError:
    # Image file doesn't exist
    pass
except UnidentifiedImageError:
    # Corrupted or unsupported image format
    pass
except RuntimeError as e:
    # ONNX inference error (corrupted model, memory issues)
    pass
```

### Memory Considerations

- Model loaded once at initialization (~30MB)
- Processing buffer per image (~5MB)
- For batch processing: Use multiprocessing with shared model in memory

## Performance Optimization

### For High-Throughput Applications

**1. Process multiple images in parallel:**
```python
from concurrent.futures import ProcessPoolExecutor

def embed_image(image_path):
    embedder = DinoEmbedder()  # Load model per process
    return embedder.embed(image_path)

with ProcessPoolExecutor(max_workers=4) as executor:
    embeddings = list(executor.map(embed_image, image_paths))
```

**2. Preload and cache embeddings:**
```python
import pickle
from pathlib import Path

def get_or_compute_embedding(image_path, cache_dir):
    """Load cached embedding or compute if not exists."""
    cache_file = Path(cache_dir) / f"{Path(image_path).stem}.emb"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    embedder = DinoEmbedder()
    embedding = embedder.embed(image_path)

    with open(cache_file, "wb") as f:
        pickle.dump(embedding, f)

    return embedding
```

### For Resource-Constrained Devices

**1. Process images one at a time:**
```python
import gc

for image_path in large_image_collection:
    embedding = embedder.embed(image_path)
    # Process/store embedding
    gc.collect()  # Force garbage collection
```

**2. Use lower resolution if acceptable:**
- Note: Model requires 224×224 input (cannot be changed)
- Consider downsampling source images before processing if disk I/O is bottleneck

## Comparison: DINO vs CLIP

| Feature | DINOv2 (this) | CLIP |
|---------|--------------|------|
| **Embedding Dim** | 384 | 512 |
| **Input Size** | 224×224 | 256×256 |
| **Training** | Self-supervised (visual only) | Image-text pairs |
| **Text Search** | ❌ No | ✅ Yes |
| **Visual Similarity** | ✅ Excellent | ✅ Good |
| **Object Focus** | ✅ Strong | ⚠️ Moderate |
| **Semantic Understanding** | ✅ Strong | ✅ Strong |
| **Speed** | ~40ms | ~30ms |
| **Model Size** | ~22MB | ~23MB |

**When to use DINO:**
- Pure visual similarity search
- Object-centric applications
- Image clustering by visual content
- Duplicate detection
- Don't need text queries

**When to use CLIP:**
- Text-to-image search
- Cross-modal retrieval
- Semantic image tagging
- Caption generation

## References

- **DINOv2 Paper:** [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- **Original DINO:** [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- **HuggingFace Model:** [RoundtTble/dinov2_vits14_onnx](https://huggingface.co/RoundtTble/dinov2_vits14_onnx)
- **Meta AI DINOv2:** [https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- **Vision Transformers:** [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **ONNX Runtime:** [https://onnxruntime.ai/](https://onnxruntime.ai/)

## Version History

- **v0.2.1:** Current implementation with bug fixes
- **v0.2.0:** Initial DINOv2 embedding support

## Support

For issues or questions:
- Check tests: `tests/plugins/test_dino_embedding.py`
- Review source: `src/cl_ml_tools/plugins/dino_embedding/`
- File issues: [GitHub repository](https://github.com/your-org/cl_ml_tools/issues)
