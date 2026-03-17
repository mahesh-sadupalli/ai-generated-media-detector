# AI-Generated Media Detector

A deepfake detection system that identifies AI-generated content by analyzing visual artifacts produced by generative models (GANs and diffusion models). The system provides explainable verdicts — not just "real or fake" but *why* content appears artificially generated, with per-artifact breakdowns and confidence scores.

## Project Introduction

AI-generated content (deepfakes) is increasingly used in fraud, misinformation, identity theft, and scams. Deepfake incidents have grown 400%+ year-over-year, and governments worldwide are passing AI content regulation (EU AI Act, US DEFIANCE Act).

This project takes a research-driven approach to detection: instead of treating it as a black-box classification problem, we leverage knowledge of *how* generative models create artifacts. By understanding the relationship between GAN loss functions (pixel loss, perceptual loss, adversarial loss) and the specific visual signatures they produce, we build detectors that are interpretable and robust.

**Key Differentiator:** Not just "fake or real" — we tell users **why** with visual evidence (artifact breakdowns, confidence per detector, human-readable explanations).

## Current Status

**Stage:** Research Prototype (functional, not production-ready)

**What works:**
- Four hand-crafted artifact detectors (smoothing, texture, mode collapse, diffusion)
- 3-class classifier: REAL / GAN-GENERATED / DIFFUSION-GENERATED
- Face extraction pipeline (OpenCV Haar Cascade)
- Synthetic test dataset (60 generated samples across 3 artifact types)
- Video analysis pipeline with per-frame breakdown and diagnostic visualizations

**What's missing:**
- No trained deep learning model (hand-crafted features only)
- No compression-aware detection (see [Real-World Testing](#real-world-testing-netanyahu-proof-of-life-video) below)
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

## Current Progress: Diffusion Model Detection

Currently implementing diffusion model detection to extend coverage beyond GAN-generated content. GANs are optimized for fast image generation, while diffusion models (Stable Diffusion, Midjourney, DALL-E, Flux) are optimized for diverse, high-quality output — each produces fundamentally different artifact signatures that require distinct detection approaches.

**What's being built:**
- DIRE (Diffusion Reconstruction Error) detector — measures how differently a diffusion model reconstructs real vs. AI-generated images
- Frequency analysis tuned for diffusion model spectral signatures
- Extending the classifier to 3-class output: real / GAN-generated / diffusion-generated
- Diffusion-generated test datasets (GenImage, DiffusionDB)

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
    +---> Smoothing Detector -----> score [0, 1]    (weight: 0.5)  ──┐
    |         FFT, Sobel, texture variance                           │
    |                                                                ├── GAN Score
    +---> Texture Detector -------> score [0, 1]    (weight: 0.1)  ──┤
    |         LBP, GLCM, spectral entropy                           │
    |                                                                │
    +---> Mode Collapse ----------> score [0, 1]    (weight: 0.4)  ──┘
    |         Symmetry, autocorrelation
    |
    +---> Diffusion Detector -----> score [0, 1]   ── Diffusion Score
    |         Reconstruction error, spectral fingerprint,
    |         noise residual analysis, patch consistency
    |
    v
3-Class Decision --> REAL / GAN-GENERATED / DIFFUSION-GENERATED + Explanation
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
- **Published Python package** — pip-installable library for AI-generated media detection
- **Production API** — FastAPI backend with async video processing
- **Web frontend** — Next.js app with drag-and-drop upload and visual heatmaps
- **Real-world benchmarks** — evaluated on FaceForensics++, CelebDF-v2, DFDC

## Existing Gaps

1. **Compression-blind detection** — The biggest gap, confirmed by real-world testing. Video codec artifacts (smoothing, uniform noise, detail loss) are indistinguishable from AI-generation artifacts to our hand-crafted features. See [Real-World Testing](#real-world-testing-netanyahu-proof-of-life-video).

2. **Narrow detection margin** — The score gap between real (~0.617) and generated (~0.759) is only 0.14, which means high false positive/negative rates on real-world data.

3. **No trained ML model** — Detection relies entirely on hand-crafted features with manually tuned thresholds. A trained model on real deepfake datasets would learn to distinguish compression from generation artifacts.

4. **Simulated artifacts only** — The test data is generated via post-processing heuristics, not actual GAN/diffusion outputs. Results may not generalize to real deepfakes.

5. **No cross-dataset evaluation** — No testing on standard benchmarks (FaceForensics++, CelebDF-v2, DFDC), so real-world performance is unknown.

6. **Limited face detection** — Haar Cascade is fast but misses faces at angles, in poor lighting, or with occlusion. Extracts blurry/partial crops that inflate artifact scores. A more robust detector (MTCNN, RetinaFace) would improve coverage.

7. **Known code bugs** — LBP coordinate swap in texture detector, BGR/RGB color space inconsistency, missing input validation across detectors.

## Architecture

```
ai-generated-media-detector/
├── src/
│   ├── artifact_detectors/          # Core detection modules
│   │   ├── smoothing_detector.py    #   FFT + Sobel + texture variance
│   │   ├── texture_detector.py      #   LBP + GLCM + spectral entropy
│   │   ├── mode_collapse_detector.py#   Symmetry + autocorrelation
│   │   ├── diffusion_detector.py   #   DIRE proxy + spectral + noise
│   │   └── combined_artifact_classifier.py  # 3-class weighted ensemble
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

## Real-World Testing: Netanyahu Proof-of-Life Video

In March 2026, Israeli PM Netanyahu posted a proof-of-life video at a Jerusalem café after Iranian media claimed his assassination. The video went viral — with online communities and even Grok (X's AI) claiming it was AI-generated. Fact-checkers (Snopes, Reuters) confirmed the video was **authentic**.

We tested our detector on this video as a real-world validation. **The model incorrectly classified it as DIFFUSION-GENERATED (56/74 frames, 0.609 confidence)** — a false positive that reveals critical limitations of hand-crafted feature detection.

### Per-Frame Score Timeline

![Score Timeline](docs/netanyahu_analysis/1_score_timeline.png)

GAN and diffusion scores hover right at the decision thresholds across all 74 frames. The model never reaches high confidence — it's making borderline calls on every frame.

### Score Distributions

![Score Distributions](docs/netanyahu_analysis/2_score_distributions.png)

Both GAN and diffusion score distributions cluster tightly between 0.43-0.65, sitting directly on top of the decision thresholds. There is no separation between the scores — the model has zero discriminative power in this range.

### Diffusion Sub-Detector Breakdown

![Diffusion Breakdown](docs/netanyahu_analysis/3_diffusion_breakdown.png)

**Reconstruction error (red) is the primary culprit** — firing at 0.85-1.0 on nearly every frame. Video codec compression smooths the image in ways that look identical to diffusion model reconstruction to our hand-crafted proxy. Spectral fingerprint and noise residual correctly stay low.

### GAN vs Diffusion Classification Space

![GAN vs Diffusion Scatter](docs/netanyahu_analysis/4_gan_vs_diffusion_scatter.png)

All 74 frames cluster in a tight blob at the intersection of decision boundaries. Frames classified as REAL (green) sit just below the diffusion threshold. The model cannot reliably separate real compressed video from generated content.

### Root Causes Identified

1. **Reconstruction error proxy is compression-blind** — our DIRE proxy (blur → sharpen → measure error) cannot distinguish video codec smoothing from diffusion model smoothing. Compressed video already sits on a "smooth manifold."

2. **Thresholds calibrated on synthetic data** — thresholds (0.55 diffusion, 0.60 GAN) were tuned on 60 synthetic images. Real-world compressed video scores land in a completely different range.

3. **Face detection quality** — Haar Cascade extracts blurry, partial, and sometimes non-face crops. Poor crops amplify smoothing and reconstruction error scores.

### Lessons Learned

- Hand-crafted features hit a ceiling on compressed social media video
- A trained deep learning model (Phase 2) must see compressed real video during training
- The 0.14 score gap confirmed insufficient for real-world reliability
- This is the same class of error that caused Grok to flag the same video as "100% deepfake"

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
