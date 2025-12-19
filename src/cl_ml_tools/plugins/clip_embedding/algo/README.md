# CLIP Embedding Algorithm

## Overview

Generates image embeddings using Apple's MobileCLIP-S2 model, optimized for mobile and edge devices. Produces 512-dimensional normalized feature vectors suitable for semantic image search, similarity comparison, and content-based retrieval.

## Algorithm Details

### Model

- **Model:** MobileCLIP-S2 (ONNX)
- **Architecture:** Efficient vision transformer optimized for mobile deployment
- **Input Size:** 256×256 RGB
- **Output:** 512-dimensional float32 vector
- **Source:** [https://huggingface.co/apple/MobileCLIP-S2-onnx](https://huggingface.co/apple/MobileCLIP-S2-onnx)
- **License:** Apache 2.0

### Implementation

The implementation follows a standard embedding pipeline:

#### 1. Preprocessing
```python
# Convert to RGB (handles RGBA, grayscale, L mode)
if image.mode != "RGB":
    image = image.convert("RGB")

# Resize to 256×256 using BICUBIC resampling
image = image.resize((256, 256), Image.Resampling.BICUBIC)

# Normalize pixel values with CLIP-specific statistics
img_array = np.asarray(image, dtype=np.float32) / 255.0
img_array = (img_array - CLIP_MEAN) / CLIP_STD

# CLIP normalization constants:
# Mean: [0.48145466, 0.4578275, 0.40821073]
# Std:  [0.26862954, 0.26130258, 0.27577711]

# Transpose to CHW format (channels-first) and add batch dimension
img_array = np.transpose(img_array, (2, 0, 1))
img_array = np.expand_dims(img_array, axis=0)
```

#### 2. Inference
- ONNX Runtime with CPU execution provider
- Graph optimization enabled (ORT_ENABLE_ALL)
- Single forward pass through the network

#### 3. Postprocessing
```python
# Remove batch dimension and flatten if needed
embedding = np.squeeze(embedding)
if embedding.ndim != 1:
    embedding = embedding.flatten()

# L2 normalization (optional, default=True)
if normalize:
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
```

### Performance Characteristics

- **Memory:** ~50MB (model) + ~5MB (processing buffer)
- **Speed:**
  - Raspberry Pi 5 (4 cores): ~150ms per image
  - Desktop CPU (8 cores): ~30ms per image
  - With Hailo 8 AI Hat+: TBD (future optimization)
- **Model Size:** ~23MB download
- **Disk Cache:** Models cached in `~/.cache/cl_ml_tools/models/`

### Technical Details

**Model Architecture:**
- Efficient ViT-based architecture with mobile optimizations
- Distilled from larger CLIP models for efficiency
- Optimized for CPU inference without GPU requirements

**Optimization Techniques:**
- Graph-level optimizations via ONNX Runtime
- Efficient memory layout (CHW format)
- BICUBIC resampling for quality vs speed balance

## Parameters

### ClipEmbedder.__init__()
```python
def __init__(self, model_path: str | Path | None = None)
```

**Parameters:**
- `model_path` (str | Path | None): Custom model path, or None to auto-download
  - If None: Downloads from HuggingFace to cache directory
  - If specified: Loads model from given path
  - Raises FileNotFoundError if path doesn't exist

### ClipEmbedder.embed()
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
- `NDArray[np.float32]`: Embedding vector of shape (512,)

**Raises:**
- `FileNotFoundError`: If image_path doesn't exist
- `PIL.UnidentifiedImageError`: If image format unsupported
- `RuntimeError`: If model inference fails

## Output Format

Returns a NumPy array of 512 float32 values representing the image embedding.

### Normalized Embeddings (default):
- Shape: `(512,)`
- Data type: `np.float32`
- Value range: approximately [-1, 1]
- L2 norm: exactly 1.0 (unit vector)
- **Usage:** Directly compute cosine similarity via dot product:
  ```python
  similarity = np.dot(embedding1, embedding2)
  # Result in [-1, 1]: 1 = identical, -1 = opposite, 0 = orthogonal
  ```

### Unnormalized Embeddings:
- Shape: `(512,)`
- Data type: `np.float32`
- Value range: varies (typically [-10, 10])
- **Usage:** Custom normalization or analysis of raw features

## Use Cases

### 1. Semantic Image Search
```python
from cl_ml_tools.algorithms import ClipEmbedder
import numpy as np

embedder = ClipEmbedder()

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

### 2. Image Similarity/Deduplication
```python
def are_similar(img1_path, img2_path, threshold=0.95):
    emb1 = embedder.embed(img1_path)
    emb2 = embedder.embed(img2_path)
    similarity = np.dot(emb1, emb2)
    return similarity > threshold
```

### 3. Content-Based Clustering
```python
from sklearn.cluster import KMeans

# Generate embeddings for all images
embeddings = np.array([embedder.embed(img) for img in images])

# Cluster by visual similarity
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(embeddings)
```

## Edge Cases & Limitations

### Supported Formats
✅ RGB, RGBA, grayscale, L mode images
✅ All PIL-supported formats (JPEG, PNG, WEBP, HEIF, BMP, GIF, TIFF)
✅ Images of any size (automatically resized)

### Limitations
- **Fixed input size:** 256×256 - images resized, may lose fine details
- **No batch processing:** Images processed sequentially (use multiprocessing for parallelism)
- **Not optimized for tiny images:** <64px may lose quality during upsampling
- **Semantic understanding:** Limited to model's training data (natural images, objects, scenes)
- **No text understanding:** Use CLIP text encoder separately for text-image matching
- **Color dependency:** Grayscale images converted to RGB (may affect embedding quality)

### Error Handling
```python
try:
    embedding = embedder.embed("image.jpg")
except FileNotFoundError:
    # Image file doesn't exist
    pass
except PIL.UnidentifiedImageError:
    # Corrupted or unsupported image format
    pass
except RuntimeError as e:
    # ONNX inference error (corrupted model, memory issues)
    pass
```

### Memory Considerations
- Model loaded once at initialization (~50MB)
- Processing buffer per image (~5MB)
- For batch processing: Use multiprocessing with shared model in memory

## Performance Optimization

### For High-Throughput Applications

**1. Process multiple images in parallel:**
```python
from concurrent.futures import ProcessPoolExecutor

def embed_image(image_path):
    embedder = ClipEmbedder()  # Load model per process
    return embedder.embed(image_path)

with ProcessPoolExecutor(max_workers=4) as executor:
    embeddings = list(executor.map(embed_image, image_paths))
```

**2. Preload images for faster processing:**
```python
from PIL import Image

# Preload and preprocess
images = [Image.open(path) for path in paths]
preprocessed = [embedder.preprocess(img) for img in images]

# Fast inference loop
embeddings = []
for input_array in preprocessed:
    output = embedder.session.run([embedder.output_name],
                                   {embedder.input_name: input_array})[0]
    embeddings.append(embedder.postprocess(output))
```

### For Resource-Constrained Devices

**1. Process images one at a time and release memory:**
```python
import gc

for image_path in large_image_collection:
    embedding = embedder.embed(image_path)
    # Process/store embedding
    gc.collect()  # Force garbage collection
```

**2. Use lower resolution if acceptable:**
- Note: Model requires 256×256 input (cannot be changed)
- Consider downsampling source images before processing if disk I/O is bottleneck

## References

- **MobileCLIP Paper:** [MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training](https://arxiv.org/abs/2311.17049)
- **HuggingFace Model:** [apple/MobileCLIP-S2-onnx](https://huggingface.co/apple/MobileCLIP-S2-onnx)
- **Original CLIP:** [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **ONNX Runtime:** [https://onnxruntime.ai/](https://onnxruntime.ai/)

## Version History

- **v0.2.1:** Current implementation with bug fixes
- **v0.2.0:** Initial CLIP embedding support with MobileCLIP-S2

## Support

For issues or questions:
- Check tests: `tests/plugins/test_clip_embedding.py`
- Review source: `src/cl_ml_tools/plugins/clip_embedding/`
- File issues: [GitHub repository](https://github.com/your-org/cl_ml_tools/issues)
