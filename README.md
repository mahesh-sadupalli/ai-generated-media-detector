# AI-Generated Media Detector

A deepfake detection system that identifies AI-generated content by analyzing visual artifacts produced by generative models (GANs and diffusion models). The system provides explainable verdicts — not just "real or fake" but *why* content appears artificially generated, with per-artifact breakdowns and confidence scores.

## Project Introduction

AI-generated content (deepfakes) is increasingly used in fraud, misinformation, identity theft, and scams. Deepfake incidents have grown 400%+ year-over-year, and governments worldwide are passing AI content regulation (EU AI Act, US DEFIANCE Act).

This project takes a research-driven approach to detection: instead of treating it as a black-box classification problem, we leverage knowledge of *how* generative models create artifacts. By understanding the relationship between GAN loss functions (pixel loss, perceptual loss, adversarial loss) and the specific visual signatures they produce, we build detectors that are interpretable and robust.

**Key Differentiator:** Not just "fake or real" — we tell users **why** with visual evidence (artifact breakdowns, confidence per detector, human-readable explanations).

## Current Status

**Stage:** Research Prototype (functional, not production-ready)

**What works:**
- Three hand-crafted artifact detectors (smoothing, texture, mode collapse)
- Combined weighted classifier with explainable predictions
- Face extraction pipeline (OpenCV Haar Cascade)
- Synthetic test dataset (60 generated samples across 3 artifact types)
- Working demo script with end-to-end pipeline

**What's missing:**
- No trained deep learning model (hand-crafted features only)
- No diffusion model detection (GAN-only)
- No web interface or API
- No real-world evaluation on standard benchmarks
- Simulated artifacts (post-processing heuristics, not real GAN outputs)
- No database, auth, or infrastructure

**Current detection accuracy (on synthetic data):**

| Input Type | Avg Score | Expected |
|---|---|---|
| Real faces | ~0.617 | Low (correct) |
| High pixel-loss artifacts | ~0.759 | High (correct) |
| High adversarial artifacts | ~0.779 (mode collapse) | High (correct) |
| **Score gap** | **~0.14** | Narrow — needs improvement |

## Methodology

The detection system is built on a core research insight: different GAN loss function components produce distinct, detectable visual artifacts.

### Loss Function to Artifact Mapping

| Loss Component | Artifact Type | Detection Method |
|---|---|---|
| Pixel loss (L1/L2) | Over-smoothing, blurriness | FFT frequency analysis, Sobel edge detection, texture variance |
| Perceptual loss (VGG) | Texture inconsistencies | Local Binary Patterns (LBP), GLCM co-occurrence matrices, spectral entropy |
| Adversarial loss | Mode collapse, repetition | Bilateral symmetry analysis, autocorrelation, spatial repetition detection |

### Detection Pipeline

```
Input (image/video)
    |
    v
Face Detection (OpenCV Haar Cascade)
    |
    v
Face Extraction & Normalization (224x224 RGB)
    |
    +---> Smoothing Detector ---> score [0, 1]    (weight: 0.5)
    |         FFT, Sobel, texture variance
    |
    +---> Texture Detector ----> score [0, 1]     (weight: 0.1)
    |         LBP, GLCM, spectral entropy
    |
    +---> Mode Collapse -------> score [0, 1]     (weight: 0.4)
    |         Symmetry, autocorrelation
    |
    v
Weighted Combination --> Overall Score
    |
    v
Threshold (0.60) --> REAL / FAKE + Explanation
```

## Scope

### Current Scope
- **GAN-generated face detection** using hand-crafted signal processing features
- **Image and video input** with per-frame analysis for video
- **Explainable output** with per-detector scores and human-readable reasoning
- **Synthetic data generation** with controlled artifact injection

### Planned Scope (Roadmap)
- **Diffusion model detection** (Stable Diffusion, Midjourney, DALL-E, Flux) via DIRE and diffusion-tuned frequency analysis
- **Multi-class classification** — real / GAN-generated / diffusion-generated
- **Deep learning backbone** — EfficientNet-B4 hybrid model with hand-crafted features
- **Production API** — FastAPI backend with async video processing
- **Web frontend** — Next.js app with drag-and-drop upload and visual heatmaps
- **Real-world benchmarks** — evaluated on FaceForensics++, CelebDF-v2, DFDC

## Existing Gaps

1. **No diffusion model coverage** — The biggest gap. Most AI-generated faces today come from diffusion models, which produce fundamentally different artifacts than GANs. The current detectors will miss them.

2. **Narrow detection margin** — The score gap between real (~0.617) and generated (~0.759) is only 0.14, which means high false positive/negative rates on real-world data.

3. **No trained ML model** — Detection relies entirely on hand-crafted features with manually tuned thresholds. A trained model on real deepfake datasets would significantly improve accuracy.

4. **Simulated artifacts only** — The test data is generated via post-processing heuristics, not actual GAN outputs. Results may not generalize to real deepfakes.

5. **No cross-dataset evaluation** — No testing on standard benchmarks (FaceForensics++, CelebDF-v2, DFDC), so real-world performance is unknown.

6. **Limited face detection** — Haar Cascade is fast but misses faces at angles, in poor lighting, or with occlusion. A more robust detector (MTCNN, RetinaFace) would improve coverage.

7. **Known code bugs** — LBP coordinate swap in texture detector, BGR/RGB color space inconsistency, missing input validation across detectors.

## Architecture

```
ai-generated-media-detector/
├── src/
│   ├── artifact_detectors/          # Core detection modules
│   │   ├── smoothing_detector.py    #   FFT + Sobel + texture variance
│   │   ├── texture_detector.py      #   LBP + GLCM + spectral entropy
│   │   ├── mode_collapse_detector.py#   Symmetry + autocorrelation
│   │   └── combined_artifact_classifier.py  # Weighted ensemble
│   ├── artifact_generators/         # Synthetic data generation
│   │   └── controlled_gan.py        #   Adaptive loss GAN for test data
│   ├── utils/                       # Shared utilities
│   │   ├── simple_face_detection.py #   Haar cascade face extraction
│   │   └── debug_face_detection.py  #   Debugging helpers
│   └── models/                      # ML model definitions (future)
├── data/
│   ├── raw/                         # Source datasets
│   └── generated/                   # Synthetic test samples
│       ├── high_pixel/              #   20 samples (pixel-loss artifacts)
│       ├── high_perceptual/         #   20 samples (perceptual-loss artifacts)
│       └── high_adversarial/        #   20 samples (adversarial-loss artifacts)
├── tests/                           # Test suite
├── config/                          # Configuration files
├── docs/                            # Documentation
├── demo_runner.py                   # Interactive demo
└── requirements.txt                 # Python dependencies
```

### Planned Architecture (Phase 1)

```
Input Image (224x224)
    |
    +---> EfficientNet-B4 Backbone ---------> CNN Features (1792d)
    |     (pretrained ImageNet, fine-tuned)
    |
    +---> Hand-Crafted Detectors -----------> Artifact Features (12d)
    |     Smoothing (4) + Texture (4) + Collapse (4)
    |
    +---> Diffusion Detector ---------------> Diffusion Features (TBD)
    |     DIRE + spectral analysis
    |
    +---> Frequency Analysis Head ----------> Frequency Features (256d)
    |     DCT/FFT spectral features
    |
    v
Concatenate --> FC Layers --> Output: P(real), P(GAN), P(diffusion)
```

## Results (Current)

Results on synthetic test data (60 samples, hand-crafted feature detectors):

### Per-Detector Scores

| Detector | Real Faces | Generated (Pixel) | Generated (Adversarial) |
|---|---|---|---|
| Smoothing | 0.617 | 0.759 | 0.680 |
| Texture | ~0.45 | ~0.50 | ~0.48 |
| Mode Collapse | 0.567 | 0.620 | 0.779 |

### Combined Classifier

| Metric | Value |
|---|---|
| Real face avg score | ~0.617 |
| Generated avg score | ~0.759 |
| Score gap | ~0.14 |
| Classification threshold | 0.60 |

### Key Findings
- **Smoothing detector** shows the best single-feature discrimination (0.617 vs 0.759)
- **Mode collapse detector** is strongest for adversarial artifacts specifically (0.567 vs 0.779)
- **Texture detector** has weak discrimination power (weight reduced to 0.1 in ensemble)
- The system correctly identifies loss-function-specific artifact types
- Narrow margin means the system is not yet reliable for real-world deployment

### Target Metrics (After Phase 1 Training)

| Metric | Target |
|---|---|
| Accuracy (FaceForensics++) | > 92% |
| AUC-ROC | > 0.95 |
| F1 Score | > 0.90 |
| False Positive Rate | < 5% |
| Inference Latency (GPU) | < 500ms |
| Inference Latency (CPU) | < 2s |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) 2025 Mahesh Sadupalli
