# Implementation Plan: Feature Expansion and Maintenance

## Overview
This plan details the phased approach to implementing major feature expansion for `cl_ml_tools`, including 4 new ML modules, Exif plugin refactor, documentation overhaul, and comprehensive testing.

**Model Strategy**: Start with FP32 ONNX models to validate functionality, then optimize to INT8 quantization in Phase 6 (optional).

## Phase Breakdown

### Phase 0: Foundation & Tooling (Days 1-2)
**Objective**: Set up infrastructure for successful development

**Critical Actions**:
1. Rename `.claude` to `.ai_context` and update `.gitignore`
2. Create `.ai_context/CLAUDE.md` as context file (with disclaimer about non-runtime usage)
3. Set up Git LFS configuration via `.gitattributes` for ONNX models
4. Configure pytest-cov with coverage thresholds in `pyproject.toml`
5. Create `INTERNALS.md` with standardized format
6. Add ONNX runtime dependencies to `pyproject.toml`

**Deliverables**:
- `.ai_context/CLAUDE.md` (context tracker)
- `.gitattributes` (Git LFS for `*.onnx` files)
- Updated `.gitignore`
- `INTERNALS.md` (known issues registry)
- `pyproject.toml` with ONNX deps

---

### Phase 1: Exif Plugin Refactor (Days 3-5)
**Objective**: Convert raw exif code into production-ready plugin

**Implementation Steps**:
1. Create `schema.py` with `ExifParams` and `ExifMetadata` Pydantic models
2. Create `task.py` with `ExifTask` extending `ComputeModule[ExifParams]`
3. Create `routes.py` with FastAPI endpoint factory
4. Update `__init__.py` with proper exports
5. Refactor `algo/exif_tool_wrapper.py` with better error handling
6. Write comprehensive tests (`tests/test_exif_plugin.py`)
7. Create plugin README

**Reference Files**:
- `src/cl_ml_tools/plugins/image_conversion/` (architecture pattern)
- `src/cl_ml_tools/plugins/hash/task.py` (error handling pattern)

**Deliverables**:
- Full plugin implementation
- Tests with >80% coverage
- Plugin README

---

### Phase 2: Face Recognition Infrastructure (Days 6-12)
**Objective**: Implement Face Detection (1A) + Face Embedding (1B) with ONNX

**Module 1A: Face Detection**
- **Model**: MediaPipe Face Detection (FP32 ONNX)
- **Output**: List of bboxes (normalized 0.0-1.0) + confidence scores
- **Config**: Min confidence 0.7, NMS threshold 0.4

**Module 1B: Face Embedding**
- **Model**: MobileFaceNet (ArcFace) FP32 ONNX
- **Output**: 128D or 512D L2-normalized embeddings + quality score

**Testing Strategy**:
- Shape-based tests (input/output dimensions)
- Golden vectors (pre-computed embeddings for known faces)
- Deterministic inference checks
- Performance benchmarks

**Deliverables**:
- 2 plugins with full implementation
- Tests with golden vectors
- Model sourcing documentation
- READMEs for both plugins

---

### Phase 3: Embedding Infrastructure (Days 13-18)
**Objective**: Implement DINOv2 (2A) + MobileCLIP (2B) for visual/semantic search

**Module 2A: DINOv2 Embedding**
- **Model**: `dinov2_vits14` FP32 ONNX
- **Output**: 384D CLS token embedding
- **Config**: 224x224 input, ImageNet normalization

**Module 2B: MobileCLIP Embedding**
- **Model**: MobileCLIP image encoder FP32 ONNX
- **Output**: 512D embedding
- **Config**: CLIP-specific preprocessing

**Testing Strategy**:
- Shape validation tests
- Golden vectors for known images
- Deterministic inference checks
- Similarity score validation (cosine similarity)

**Deliverables**:
- 2 plugins with full implementation
- Tests with golden vectors
- READMEs

---

### Phase 4: Documentation & Testing Completion (Days 19-22)
**Objective**: Comprehensive documentation and test coverage

**Documentation Tasks**:
1. Update main `README.md`:
   - Add new plugins to features list
   - Document model sourcing strategy
   - Add model cache locations
   - Document offline behavior
   - Add licensing section (MediaPipe, DINOv2, CLIP)
2. Create plugin READMEs for all 4 new plugins
3. Document "resize → thumbnail" migration
4. Create `CONTRIBUTING.md` with plugin development guide

**Testing Tasks**:
1. Add tests for utilities (random_media_generator, media_types, timestamp)
2. Update hash module tests (use random_media_generator)
3. Add missing test for `image_conversion` plugin
4. Achieve >85% overall code coverage

**Deliverables**:
- Complete documentation suite
- >85% code coverage
- All utility tests implemented

---

### Phase 5: Quality Assurance & Production Readiness (Days 23-25)
**Objective**: Production-ready codebase with FP32 models

**QA Tasks**:
1. Run ruff linter on `src/` and `tests/`
2. Run basedpyright type checker
3. Run full test suite with coverage report
4. Performance benchmarking (FP32 baseline)

**Integration Testing**:
1. Test all plugins end-to-end via routes
2. Test job queue processing with all task types
3. Test error scenarios (missing models, corrupt files, invalid params)

**Production Deployment Guide**:
1. Document model installation process
2. Create deployment checklist
3. Document system requirements
4. Add troubleshooting section

**Deliverables**:
- Clean linter/type checker output (or documented exceptions)
- Performance benchmark report (FP32 baseline)
- Production deployment guide
- Release-ready codebase

---

### Phase 6: Model Optimization (Future - Days 26-30)
**Objective**: Quantize models to INT8 for production performance

**Optimization Tasks**:
1. Quantize all ONNX models to INT8 using `onnxruntime.quantization`
2. Create quantization pipeline (calibration, validation)
3. Update plugins to support both FP32 and INT8 models
4. Re-run performance benchmarks (target: >5 FPS on RPi 4)
5. Update documentation with quantization process

**Deliverables**:
- INT8 quantized models for all 4 ML plugins
- Quantization scripts and documentation
- Performance comparison report (FP32 vs INT8)

**Note**: This phase is optional for initial release.

---

## Critical Files Reference

**Files to Create**:
- `.ai_context/CLAUDE.md`
- `.gitattributes`
- `INTERNALS.md`
- `CONTRIBUTING.md`
- Plugin files for: exif, face_detection, face_embedding, dino_embedding, clip_embedding
- Test files for all new plugins + utilities

**Files to Modify**:
- `.gitignore` (add .ai_context/)
- `pyproject.toml` (ONNX deps, coverage config, plugin entry points)
- `README.md` (major updates)
- `tests/test_hash_plugin.py` (use random_media_generator)

**Files to Reference** (architecture patterns):
- `src/cl_ml_tools/plugins/image_conversion/` - Plugin structure
- `src/cl_ml_tools/plugins/hash/task.py` - Error handling, media type routing
- `src/cl_ml_tools/plugins/hls_streaming/schema.py` - Complex Pydantic models

---

## Risk Mitigation Strategies

1. **ONNX Model Availability**: Test on target hardware early, provide FP32 fallback
2. **Performance Targets**: Benchmark incrementally, profile bottlenecks early
3. **Testing Golden Vectors**: Generate and commit golden vectors early, version control expected outputs
4. **Context Management**: Update CLAUDE.md after each phase, maintain TASKS.md with granular status

---

## Success Criteria

- **Phase 0**: All tooling configured, dependencies resolved
- **Phase 1**: Exif plugin passes all tests, documented
- **Phase 2**: Face plugins functional with FP32 models
- **Phase 3**: Embedding plugins produce deterministic outputs
- **Phase 4**: >85% code coverage, all docs complete
- **Phase 5**: Production-ready, clean linter/type checker, deployment guide ready
- **Phase 6** (Optional): INT8 quantized models achieve >5 FPS target on RPi 4 CPU

---

## Progress Tracking

See `TASKS.md` for detailed task-by-task status tracking.
See `.ai_context/CLAUDE.md` for high-level context and decisions.
