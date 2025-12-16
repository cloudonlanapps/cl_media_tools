# 🚀 Finalized Development Plan: Feature Expansion and Maintenance

**Context:**
We are conducting a major feature expansion and maintenance session for the `cl_ml_tools` repository. This plan incorporates architectural reviews and addresses identified risks and missing operational details.

**Objective:**
Implement 4 new ML modules (using ONNX), refactor the Exif tool, update documentation, and maximize code coverage, with a focus on CPU-optimized, quantized performance.

---

## Session Plan & Instructions

### Step 0: Operational & Context Management

* **Context Folder Renaming:** Rename the temporary context folder from `.claude` to **`.ai_context`**. (Add this folder to `.gitignore`).
* **Context File:** Generate an `ai_context/CLAUDE.md` file to maintain high-level context and progress state.
* **Constraint:** Explicitly document that `CLAUDE.md` is **not a source of truth**, must **not be referenced by runtime code or tests**, and exists purely for large-language model context optimization.
* **Model Sourcing Strategy:**
    * Define a clear plan for obtaining **ONNX Quantized** models (self-quantization or pre-trained hub).
    * **Model Storage:** Models will be stored using **Git LFS** or downloaded on demand from a central, versioned repository. **[REVIEW: Define Model Storage Location (Git LFS / External URL)]**

### Step 1: Feature Implementation (New & Existing)

#### A. Exif Plugin Refactor

* **Task:** Convert the raw code at `src/cl_ml_tools/plugins/exif/algo` into a standard plugin using `src/cl_ml_tools/plugins/image_conversion` as the architectural reference.
* **Public API:** The plugin must function as a **Metadata Extractor**.
* **Output Schema:** The output must be a **Typed Pydantic model** for the core fields, allowing the raw dictionary to be stored in an internal field for completeness.
* **Error Handling:** Explicitly handle corrupt EXIF and unsupported formats by returning a predictable, empty/default Pydantic model state and logging the error, rather than raising an exception that halts the worker.
* **Risk Mitigation:** Explicitly document dependency on **Pillow** and its platform-specific quirks for EXIF reading.
* **Dependencies:** Update `pyproject.toml` with necessary packages.

#### B. New ML Modules (RPi/CPU Optimized via ONNX)

**Critical Constraint:** **Prefer ONNX Quantized models** for all ML tasks. The goal is to use `onnxruntime` (CPU) for efficient inference on low-power processors like Raspberry Pi. 

* **Performance Target:** Establish a soft goal of **>5 FPS** for the combined Face Detection/Embedding pipeline on a Raspberry Pi 4-class CPU.

| Module | Responsibility | Implementation Details | Output Schema |
| :--- | :--- | :--- | :--- |
| **1A: Face Detection** | Detect faces and return bounding boxes. | **ONNX Quantized** (MediaPipe or RetinaFace). Must resolve the trade-off and choose one. | List of `{"bbox": (x1, y1, x2, y2), "confidence": float}` |
| **1B: Face Embedding** | Given one face image, align and compute embedding. | **ONNX Quantized** MobileFaceNet (ArcFace). Alignment on CPU using **5-point landmark similarity transform**. L2-normalized output. | `{"embedding": np.ndarray, "quality": float \| None}` (128D or 512D) |
| **2A: DINOv2 Embedding** | Visual/memory similarity (duplicates, near-duplicates). | **ONNX Quantized** `dinov2_vits14`. Default to **CLS Token** for embedding. | `dino_embedding: np.ndarray` (~384D) |
| **2B: MobileCLIP/TinyCLIP** | Semantic/event similarity. | **ONNX Quantized** Image encoder only. Acknowledge risk of non-standardized availability; document plan if self-quantization is required. | `clip_embedding: np.ndarray` (~512D) |

**Module Details & Suggestions:**
* **1A Details:** Define **Minimum confidence threshold** (e.g., 0.7), use **normalized coordinates** (0.0 to 1.0), and specify the default **NMS behavior** used in the ONNX export.
* **1B Quality Metric:** The "quality" metric will be the **re-used detection confidence** if available, otherwise it will be calculated as a simple **blur/sharpness score**.
* **2A Details:** Define a standard input resolution (e.g., 224x224 or 384x384), standard RGB color space, and document the **normalization stats** used for the model input.

### Step 2: Documentation Overhaul

* **README-Driven Usability:** Update the main `README.md` to ensure users can utilize routes and workers without inspecting source code.
* **I/O Definitions:** Clearly define input and output Pydantic schemas (or NumPy structures) for every route and plugin.
* **Plugin READMEs:** Ensure every directory in `src/cl_ml_tools/plugins/` has an accurate `README.md`.
* **Contribution Guide:** Update the "Adding New Plugins" section to document the required folder structure (`algo` pattern).
* **Rename: Resize → Thumbnail:** Ensure this name change is tracked and updated in all code, tests, and documentation. Add a **temporary migration note** to `README.md` and old import locations to prevent silent drift.
* **Operational Details in Documentation:** Add sections covering:
    * **Model Download Strategy** (where to get models).
    * **Model Cache Locations** (where models are stored locally).
    * **Offline vs Online Behavior** (what happens if the model download fails).
    * **Licensing:** Briefly acknowledge the **licensing** requirements for MediaPipe, DINOv2, and CLIP/MobileCLIP models.

### Step 3: Testing & Coverage

* **Coverage Requirement:** Ensure code coverage for **ALL** plugins and utilities.
* **ML Testing Strategy (Needs Precision):**
    * Tests must be **shape-based** and ensure **deterministic inference** (same input $\rightarrow$ same output within acceptable tolerance).
    * Use **Golden Vectors** (pre-computed embeddings/bboxes for known input images) for critical plugins to check for **floating-point drift** and platform variance.
    * Tests must cover model loading, pre/post-processing pipelines, and main inference calls.
* **Utility Tests:** Add unit tests for:
    * `src/cl_ml_tools/utils/random_media_generator`
    * `src/cl_ml_tools/utils/media_types.py`
    * `src/cl_ml_tools/utils/timestamp.py`
* **Hash Module:** Update tests for the `hash` module using `random_media_generator`. **Do not skip any tests for hash.**

### Step 4: Quality Assurance

* **Linting:** Resolve errors and warnings in `src/` and `tests/`.
* **INTERNALS.md Protocol:** Use `INTERNALS.md` as a "known issues registry."
* **Format for INTERNALS.md:** Entries must follow a defined format:
    ```markdown
    - **File:** <path/to/file.py>
    - **Tool:** <ruff / basedpyright / pytest>
    - **Reason:** <Clear, justified explanation for keeping the warning/skip>
    - **Removal Criteria:** <Action required to resolve the issue>
    ```