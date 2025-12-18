# Face Embedding Algorithm

## Overview

Generates face embeddings using ArcFace model for face recognition and verification. Produces 512-dimensional normalized feature vectors from cropped face images. Includes optional quality scoring based on image sharpness to filter blurry/low-quality faces. Designed for face matching, clustering, and identity verification applications.

## Algorithm Details

### Model

- **Model:** ArcFace (ONNX)
- **Architecture:** Deep CNN trained with angular margin loss
- **Input Size:** 112×112 RGB (cropped face)
- **Output:** 512-dimensional float32 vector
- **Source:** [https://huggingface.co/garavv/arcface-onnx](https://huggingface.co/garavv/arcface-onnx)
- **Original Paper:** ArcFace: Additive Angular Margin Loss for Deep Face Recognition
- **License:** Model-specific (check HuggingFace)

### Key Characteristics

**High Discriminative Power:**
- Trained with angular margin loss for better face separation
- 512-dimensional embeddings capture fine facial details
- State-of-the-art accuracy on face verification benchmarks

**L2 Normalization:**
- Embeddings normalized to unit vectors
- Enables cosine similarity via simple dot product
- Consistent magnitude across all embeddings

**Quality Scoring:**
- Optional Laplacian variance-based quality metric
- Detects blurry or low-quality face images
- Helps filter poor-quality inputs before recognition

### Implementation

#### 1. Preprocessing
```python
# Convert to RGB
if image.mode != "RGB":
    image = image.convert("RGB")

# Resize to 112×112 using BILINEAR resampling
image = image.resize((112, 112), Image.Resampling.BILINEAR)

# Normalize to [0, 1]
img_array = np.asarray(image, dtype=np.float32) / 255.0

# Transpose to CHW format and add batch dimension
img_array = np.transpose(img_array, (2, 0, 1))
img_array = np.expand_dims(img_array, axis=0)
```

#### 2. Inference
- ONNX Runtime with CPU execution provider
- Graph optimization enabled
- Single forward pass through deep CNN
- Outputs 512-dimensional face representation

#### 3. Postprocessing
```python
# Remove batch dimension
embedding = embedding.squeeze(axis=0)

# L2 normalization (optional, default=True)
if normalize:
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
```

#### 4. Quality Scoring (optional)
```python
def compute_quality_score(image: Image) -> float:
    """Compute face quality based on Laplacian variance (blur detection)."""
    # Convert to grayscale
    img_gray = image.convert("L")
    img_array = np.asarray(img_gray, dtype=np.float32)

    # Laplacian kernel for edge detection
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    # Apply convolution
    lap_img = signal.convolve2d(img_array, kernel, mode="same")

    # Variance of Laplacian (higher = sharper)
    variance = np.var(lap_img)
    quality = min(variance / 100.0, 1.0)

    return quality
```

**Quality Score Interpretation:**
- **0.7-1.0:** Excellent quality (sharp, well-lit)
- **0.5-0.7:** Good quality (acceptable for recognition)
- **0.3-0.5:** Fair quality (may affect accuracy)
- **0.0-0.3:** Poor quality (blurry, reject for recognition)

### Performance Characteristics

- **Memory:** ~50MB (model) + ~5MB (processing buffer)
- **Speed:**
  - Raspberry Pi 5 (4 cores): ~100ms per face
  - Desktop CPU (8 cores): ~25ms per face
  - With Hailo 8 AI Hat+: TBD (future optimization)
- **Model Size:** ~23MB download
- **Disk Cache:** Models cached in `~/.cache/cl_ml_tools/models/`

### Typical Workflow

```
1. Detect faces → face_detection plugin
2. Crop faces → PIL/Pillow
3. Generate embeddings → face_embedding plugin (this)
4. Compare embeddings → cosine similarity (dot product)
5. Match/cluster → threshold-based matching or clustering
```

## Parameters

### FaceEmbedder.__init__()
```python
def __init__(self, model_path: str | Path | None = None)
```

**Parameters:**
- `model_path` (str | Path | None): Custom model path, or None to auto-download
  - If None: Downloads from HuggingFace to cache directory
  - If specified: Loads model from given path
  - Raises FileNotFoundError if path doesn't exist

### FaceEmbedder.embed()
```python
def embed(
    self,
    image_path: str | Path,
    normalize: bool = True,
    compute_quality: bool = True,
) -> tuple[NDArray[np.float32], float | None]
```

**Parameters:**
- `image_path` (str | Path): Path to cropped face image
  - **Important:** Should be a cropped face, not full image
  - Use face_detection plugin to crop faces first
  - Supported formats: JPEG, PNG, WEBP, HEIF, BMP, GIF, TIFF
- `normalize` (bool): Apply L2 normalization (default: True)
  - True: Output has L2 norm = 1.0 (unit vector)
  - False: Raw model output values
  - **Recommended:** Always True for face matching
- `compute_quality` (bool): Compute quality score (default: True)
  - Requires scipy for Laplacian computation
  - Returns None if scipy not installed or computation fails

**Returns:**
- `tuple[NDArray[np.float32], float | None]`:
  - `embedding`: 512-dimensional float32 array
  - `quality_score`: Quality score (0.0-1.0) or None

**Raises:**
- `FileNotFoundError`: If image_path doesn't exist
- `PIL.UnidentifiedImageError`: If image format unsupported
- `RuntimeError`: If model inference fails

## Output Format

Returns a tuple of (embedding, quality_score):

```python
(
    array([0.123, -0.456, 0.789, ..., -0.234], dtype=float32),  # 512 dimensions
    0.85  # quality score
)
```

**Embedding:**
- Shape: `(512,)`
- Data type: `np.float32`
- L2 norm: 1.0 (if normalized)
- Value range: approximately [-1, 1]

**Quality Score:**
- Type: `float | None`
- Range: 0.0-1.0
- Higher = better quality (sharper image)
- None if computation failed or disabled

## Use Cases

### 1. Face Verification (1:1 Matching)
```python
from cl_ml_tools.algorithms import FaceEmbedder
import numpy as np

embedder = FaceEmbedder()

# Generate embeddings for two faces
emb1, qual1 = embedder.embed("face1.jpg")
emb2, qual2 = embedder.embed("face2.jpg")

# Check quality
if qual1 and qual1 < 0.5:
    print("Warning: face1 has low quality")
if qual2 and qual2 < 0.5:
    print("Warning: face2 has low quality")

# Compute similarity (cosine similarity via dot product)
similarity = np.dot(emb1, emb2)

# Threshold-based decision
if similarity > 0.6:
    print(f"Same person (similarity: {similarity:.3f})")
else:
    print(f"Different people (similarity: {similarity:.3f})")
```

**Typical Thresholds:**
- **>0.7:** Very likely same person (high confidence)
- **0.6-0.7:** Probably same person (medium confidence)
- **0.4-0.6:** Uncertain (manual review recommended)
- **<0.4:** Likely different people

### 2. Face Recognition (1:N Matching)
```python
def recognize_face(query_face, known_faces_db, threshold=0.6):
    """Identify a person from a database of known faces."""
    embedder = FaceEmbedder()

    query_emb, query_qual = embedder.embed(query_face)

    if query_qual and query_qual < 0.5:
        return None, "Query face quality too low"

    best_match = None
    best_similarity = -1

    for person_id, face_path in known_faces_db.items():
        db_emb, _ = embedder.embed(face_path)
        similarity = np.dot(query_emb, db_emb)

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = person_id

    if best_similarity > threshold:
        return best_match, f"Matched with confidence {best_similarity:.3f}"
    else:
        return None, "No match found"

# Database of known people
known_faces = {
    "person_001": "/db/person_001_face.jpg",
    "person_002": "/db/person_002_face.jpg",
    "person_003": "/db/person_003_face.jpg",
}

# Recognize unknown face
person_id, message = recognize_face("unknown_face.jpg", known_faces)
print(f"{person_id}: {message}")
```

### 3. Face Clustering
```python
from sklearn.cluster import DBSCAN
import numpy as np

def cluster_faces(face_paths, eps=0.4, min_samples=2):
    """Cluster faces by identity."""
    embedder = FaceEmbedder()

    embeddings = []
    valid_paths = []

    for path in face_paths:
        emb, qual = embedder.embed(path)
        if qual and qual >= 0.5:  # Filter low-quality faces
            embeddings.append(emb)
            valid_paths.append(path)

    embeddings_array = np.array(embeddings)

    # Convert similarity to distance (1 - similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embeddings_array)
    distance_matrix = 1 - similarity_matrix

    # Cluster using DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = clustering.fit_predict(distance_matrix)

    # Group by cluster
    clusters = {}
    for path, label in zip(valid_paths, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(path)

    return clusters

# Cluster all faces in a directory
faces = list(Path("/faces").glob("*.jpg"))
clusters = cluster_faces(faces)

for cluster_id, cluster_faces in clusters.items():
    if cluster_id == -1:
        print(f"Outliers: {len(cluster_faces)} faces")
    else:
        print(f"Cluster {cluster_id}: {len(cluster_faces)} faces")
```

### 4. Quality Filtering
```python
def filter_high_quality_faces(face_paths, min_quality=0.6):
    """Keep only high-quality face images."""
    embedder = FaceEmbedder()
    high_quality = []

    for path in face_paths:
        emb, qual = embedder.embed(path)

        if qual and qual >= min_quality:
            high_quality.append((path, qual))

    # Sort by quality (descending)
    high_quality.sort(key=lambda x: x[1], reverse=True)

    print(f"Kept {len(high_quality)}/{len(face_paths)} high-quality faces")
    return high_quality

# Filter faces
good_faces = filter_high_quality_faces(all_faces, min_quality=0.7)
```

### 5. Building Face Database
```python
import pickle

def build_face_database(face_dir, output_file):
    """Build a database of face embeddings."""
    embedder = FaceEmbedder()
    database = {}

    for person_dir in Path(face_dir).iterdir():
        if not person_dir.is_dir():
            continue

        person_id = person_dir.name
        person_embeddings = []

        for face_img in person_dir.glob("*.jpg"):
            emb, qual = embedder.embed(face_img)

            if qual and qual >= 0.6:
                person_embeddings.append(emb)

        if person_embeddings:
            # Average multiple embeddings per person
            avg_embedding = np.mean(person_embeddings, axis=0)
            # Re-normalize
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            database[person_id] = avg_embedding

    with open(output_file, "wb") as f:
        pickle.dump(database, f)

    print(f"Built database with {len(database)} people")
    return database

# Directory structure:
# /face_db/
#   person_001/
#     face1.jpg
#     face2.jpg
#   person_002/
#     face1.jpg

db = build_face_database("/face_db", "embeddings.pkl")
```

## Edge Cases & Limitations

### Supported Scenarios
✅ Frontal faces (±30° rotation)
✅ Various lighting conditions
✅ Faces with glasses/hats
✅ Different ages (with reduced accuracy)
✅ Different expressions

### Limitations

**Input Requirements:**
- **Must be cropped face images** (not full photos)
  - Use face_detection plugin to crop faces first
  - Face should fill most of the image
  - Poor cropping severely affects accuracy
- **Face orientation:** Best with frontal faces
  - Profile faces (>45° rotation) may have poor embeddings
  - Extreme rotations not supported
- **Resolution:** Faces smaller than 112×112 may lose detail during resize

**Quality Factors:**
- Blurry images have lower quality scores
- Very low quality (<0.3) may produce unreliable embeddings
- Extreme lighting may affect embedding quality

**Recognition Limitations:**
- **Not suitable for identical twins** (very similar embeddings)
- **Age progression:** Embeddings change over time (years)
- **Major appearance changes:** Drastic hair/makeup/facial hair changes
- **Occlusions:** Masks covering >30% of face reduce accuracy

**Model Limitations:**
- No age/gender/emotion detection
- No facial landmark detection
- Designed for identification, not attribute analysis

### Error Handling

```python
from PIL import UnidentifiedImageError

try:
    embedding, quality = embedder.embed("face.jpg")
except FileNotFoundError:
    print("Face image not found")
except UnidentifiedImageError:
    print("Corrupted or unsupported image format")
except RuntimeError as e:
    print(f"Model inference error: {e}")

# Check quality
if quality is not None and quality < 0.5:
    print("Warning: Low quality face, may affect recognition accuracy")
```

## Performance Optimization

### Parallel Processing
```python
from concurrent.futures import ProcessPoolExecutor

def embed_single(face_path):
    embedder = FaceEmbedder()
    return embedder.embed(face_path)

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(embed_single, face_paths))

embeddings = [r[0] for r in results]
qualities = [r[1] for r in results]
```

### Caching Embeddings
```python
import pickle
from pathlib import Path

def get_or_compute_embedding(face_path, cache_dir):
    """Load cached embedding or compute if not exists."""
    cache_file = Path(cache_dir) / f"{Path(face_path).stem}.emb"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    embedder = FaceEmbedder()
    embedding, quality = embedder.embed(face_path)

    with open(cache_file, "wb") as f:
        pickle.dump((embedding, quality), f)

    return embedding, quality
```

## Similarity Thresholds

Recommended thresholds for different use cases:

| Use Case | Threshold | False Accept Rate | False Reject Rate |
|----------|-----------|-------------------|-------------------|
| **High security** | 0.75-0.80 | Very low (~0.1%) | Higher (~5-10%) |
| **Balanced** | 0.65-0.70 | Low (~1%) | Moderate (~2-5%) |
| **Convenient access** | 0.55-0.60 | Higher (~5%) | Low (~1%) |
| **Clustering** | 0.50-0.60 | N/A | N/A |

**Note:** Actual rates depend on image quality, lighting, and face characteristics.

## Complete Workflow Example

```python
from cl_ml_tools.algorithms import FaceDetector, FaceEmbedder
from PIL import Image
import numpy as np

def face_recognition_pipeline(image_path, database_embeddings, threshold=0.65):
    """Complete pipeline: detection → embedding → recognition."""

    # Step 1: Detect faces
    detector = FaceDetector()
    faces = detector.detect(image_path, confidence_threshold=0.7)

    if not faces:
        return "No faces detected"

    # Step 2: Crop and embed each face
    embedder = FaceEmbedder()
    results = []

    with Image.open(image_path) as img:
        for i, face in enumerate(faces):
            # Crop face region
            face_img = img.crop((
                int(face['x1']), int(face['y1']),
                int(face['x2']), int(face['y2'])
            ))

            # Save cropped face temporarily
            temp_path = f"/tmp/face_{i}.jpg"
            face_img.save(temp_path)

            # Generate embedding
            emb, qual = embedder.embed(temp_path)

            # Check quality
            if qual and qual < 0.5:
                results.append(f"Face {i+1}: Low quality (skipped)")
                continue

            # Step 3: Match against database
            best_match = None
            best_sim = -1

            for person_id, db_emb in database_embeddings.items():
                sim = np.dot(emb, db_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_match = person_id

            if best_sim > threshold:
                results.append(f"Face {i+1}: {best_match} (confidence: {best_sim:.3f})")
            else:
                results.append(f"Face {i+1}: Unknown person")

    return "\n".join(results)

# Load database
with open("face_db.pkl", "rb") as f:
    db = pickle.load(f)

# Run pipeline
result = face_recognition_pipeline("group_photo.jpg", db)
print(result)
```

## References

- **ArcFace Paper:** [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **HuggingFace Model:** [https://huggingface.co/garavv/arcface-onnx](https://huggingface.co/garavv/arcface-onnx)
- **ONNX Runtime:** [https://onnxruntime.ai/](https://onnxruntime.ai/)
- **Laplacian Blur Detection:** [https://en.wikipedia.org/wiki/Laplace_operator](https://en.wikipedia.org/wiki/Laplace_operator)

## Version History

- **v0.2.1:** Current implementation with quality scoring
- **v0.2.0:** Initial face embedding support

## Support

For issues or questions:
- Check tests: `tests/plugins/test_face_embedding.py`
- Review source: `src/cl_ml_tools/plugins/face_embedding/`
- File issues: [GitHub repository](https://github.com/your-org/cl_ml_tools/issues)
