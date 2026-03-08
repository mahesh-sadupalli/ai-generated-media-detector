# CLAUDE.md

## Git Commit Rules
- NEVER include "Co-Authored-By: Claude" or any Claude attribution in commit messages
- NEVER add Claude as a coauthor or contributor when pushing to GitHub
- Keep commit messages concise and descriptive of the actual change

## Project Overview
This is an AI-generated media detection system (deepfake detector) that identifies artifacts in both GAN-generated and diffusion-generated content. The project is transitioning from a research prototype to a production-ready product.

## Project Structure
```
gan_artifact_detector/
├── src/
│   ├── artifact_detectors/      # Core detection modules
│   │   ├── smoothing_detector.py       # FFT, Sobel, texture variance
│   │   ├── texture_detector.py         # LBP, GLCM, spectral entropy
│   │   ├── mode_collapse_detector.py   # Symmetry, autocorrelation
│   │   └── combined_artifact_classifier.py  # Weighted ensemble classifier
│   ├── artifact_generators/     # Synthetic artifact generation
│   │   └── controlled_gan.py           # Controlled GAN with adaptive loss
│   ├── utils/                   # Shared utilities
│   │   ├── simple_face_detection.py    # OpenCV Haar cascade face extraction
│   │   └── debug_face_detection.py     # Debugging utilities
│   └── models/                  # ML model definitions (future)
├── data/
│   ├── raw/                     # FaceForensics++ and other datasets
│   └── generated/               # Synthetic test samples (60 images)
├── tests/                       # Test suite (needs population)
├── config/                      # Configuration files (needs population)
├── docs/                        # Documentation
├── demo_runner.py               # Interactive demo script
└── requirements.txt             # Python dependencies
```

## Code Conventions
- Use Python type hints for all function signatures
- Use NumPy-style docstrings for classes and public methods
- Image arrays follow RGB color space convention inside detectors; convert from BGR immediately after cv2.imread or cv2.VideoCapture
- Use `pathlib.Path` or `os.path` with `__file__`-relative resolution for file paths — never hardcode relative paths like `../../data`
- Use `try/finally` or context managers for resource cleanup (especially cv2.VideoCapture)
- Add epsilon (1e-10) to all division operations for numerical stability
- Detector scores are normalized to [0.0, 1.0] range

## Import Rules
- Use proper Python package imports — never use `sys.path.append()` with relative paths
- Prefer relative imports within the `src/` package (e.g., `from .smoothing_detector import ...`)
- External dependencies must be listed in `requirements.txt`

## Testing Rules
- Tests go in the `tests/` directory, not alongside source files
- Use pytest as the test framework
- Test files must be named `test_*.py`
- Use synthetic/generated images for tests, not real video files that may not be present
- Cover edge cases: empty images, single-color images, wrong dimensions, None inputs

## Detection Architecture
The system uses a weighted ensemble of three hand-crafted detectors:
- Smoothing Detector (weight: 0.5) — detects over-smoothing from pixel-loss optimization
- Texture Detector (weight: 0.1) — detects perceptual-loss texture inconsistencies
- Mode Collapse Detector (weight: 0.4) — detects adversarial-loss repetition patterns
- Overall threshold for FAKE classification: 0.60

## Future Implementation Plan

### Phase 1: Diffusion Model Detection
- Add `diffusion_detector.py` implementing DIRE (Diffusion Reconstruction Error)
- Add frequency analysis tuned for diffusion model spectral signatures
- Extend classifier to 3-class output: real / GAN-generated / diffusion-generated
- Add diffusion-generated datasets (GenImage, DiffusionDB)

### Phase 2: Deep Learning Model
- Replace hand-crafted features with EfficientNet-B4 backbone
- Hybrid architecture: CNN features (1792d) + hand-crafted (12d) + frequency (256d)
- Train on FaceForensics++, CelebDF-v2, DFDC datasets
- Target: >92% accuracy, >0.95 AUC-ROC, <5% false positive rate

### Phase 3: Production Backend
- FastAPI async API with endpoints: POST /detect/image, POST /detect/video
- Celery + Redis task queue for video processing
- PostgreSQL for results storage
- Docker containerization

### Phase 4: Frontend
- Next.js web app with drag-and-drop upload
- Visual results: heatmap overlays, artifact breakdowns, confidence scores
- Responsive design for mobile and desktop

### Phase 5: Hardening
- Proper pyproject.toml package configuration
- CI/CD pipeline with GitHub Actions
- Load testing, security audit, edge case coverage
- Logging framework (replace all print statements)
- Configuration management (YAML/TOML for thresholds and paths)

## Known Issues to Fix
- LBP coordinate swap in texture_detector.py (row/col indices swapped)
- BGR/RGB color space inconsistency between face extractor and detectors
- Missing None check after cv2.imread calls
- Video capture resource leak (no try/finally)
- sys.path hacks instead of proper package imports
- Tests are inside src/ instead of tests/ directory
- Empty config/ directory with no configuration management
- Magic number thresholds without calibration documentation
