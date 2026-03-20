# Unified Research Plan: Data Compression × AI-Generated Media Detection

**Author**: Mahesh Sadupalli
**Date**: 2026-03-18
**Linking**: Master Thesis (INR Compression + Continual Learning) ↔ AI Media Detector Project

---

## Executive Summary

This research plan unifies two bodies of work:

1. **Master Thesis** — *Concurrent Neural Network Training for Compression of Spatio-Temporal Data* (BTU Cottbus-Senftenberg): Demonstrates that implicit neural representations (INRs) achieve extreme compression ratios (62K:1 – 242K:1) on spatio-temporal data, but suffer severe catastrophic forgetting in streaming/online settings. Experience Replay is the only effective continual learning strategy for INR-based compression; regularization methods (EWC, LwF) fail due to shared parameter spaces.

2. **Detection Project** — *AI-Generated Media Detector*: Hand-crafted ensemble detector (smoothing + texture + mode collapse + diffusion) with recently added compression-awareness module. Currently achieves only a 0.14 score gap between real and generated content. Real-world testing revealed video codec compression causes massive false positives.

**The Core Insight**: Both projects converge on the same fundamental problem — **how neural networks represent and reconstruct signals under lossy information loss, and how that process can be exploited for forensic purposes**. The thesis provides deep expertise in neural compression behavior and forgetting dynamics; the detection project provides the application domain. Together, they enable a novel research contribution:

> **INR Reconstruction Residuals as Compression-Invariant Forensic Features for AI-Generated Media Detection with Continual Learning**

---

## 1. Research Landscape — What the Literature Tells Us

### 1.1 The Compression Problem Is the #1 Deployment Blocker

| Finding | Source | Impact |
|---------|--------|--------|
| Detectors lose **50% AUC** on in-the-wild compressed content | Deepfake-Eval-2024 (Chandra et al., arXiv:2503.02857) | Lab results don't transfer to real-world |
| FaceForensics++ c23→c40 causes **10-25% AUC drop** | Multiple studies | Even moderate compression degrades detection |
| JPEG AI artifacts **mimic generative model artifacts** | Cannas et al. (ICCV 2025 Workshop, arXiv:2412.03261) | Neural codecs will break existing detectors |
| Social media compression causes **15-20% AP decrease** | Montibeller et al. (ACM DFW 2025, arXiv:2508.08765) | Platform pipelines are the real bottleneck |
| No CL paper evaluates under realistic compression | Gap identified across literature | Critical blind spot |

### 1.2 Continual Learning Is Essential But Limited

| Finding | Source | Impact |
|---------|--------|--------|
| Forward transfer AUC → **~0.5 (random)** within 3 unseen generators | Fontana et al. (arXiv:2509.07993) | CL alone can't solve generalization |
| Experience Replay is the **strongest baseline** (C-AUC ~0.89) | CDDB benchmark + chronological study | Simple replay beats complex methods |
| EWC/LwF **fail for shared parameter spaces** | Your thesis + literature | Confirms thesis findings transfer to detection |
| Larger models forget **more severely** | Your thesis | Counter-intuitive but critical for architecture design |
| GPL achieves 92.14% with reduced forgetting | Zhang et al. (ICCV 2025) | Hyperbolic alignment is promising |
| DevFD uses orthogonal LoRA MoE | NeurIPS 2025 (arXiv:2509.19230) | Parameter-efficient CL is practical |

### 1.3 The State-of-the-Art Detection Methods

| Method | Venue | Key Idea | Compression Robustness |
|--------|-------|----------|----------------------|
| **FIRE** | CVPR 2025 | Mid-frequency reconstruction error | Strong (mid-freq survives compression) |
| **CLIP-based** | CVPRW 2024 | Semantic features via foundation model | Inherently robust (+13% on laundered data) |
| **D3** | CVPR 2025 | Discrepancy learning from distortion | Best across JPEG QF 30-100 |
| **PLADA** | 2025 | Block effect eraser for OSN compression | First to explicitly model block effects |
| **NPR** | CVPR 2024 | Neighboring pixel relationships | Structural (partially survives compression) |
| **DINOv2 backbone** | Pellegrini 2025 | Self-supervised ViT features | 97.36% AUROC, robust |

### 1.4 The Gap Your Work Can Fill

**No published work** combines:
- INR reconstruction residuals as forensic features
- Compression-invariant detection
- Continual learning for evolving generators
- Practical deployment under social media compression

Only **one paper** (NRR, ECCV 2024) explores INR for manipulation detection, and it doesn't address compression robustness or continual learning.

---

## 2. The Novel Research Contribution

### 2.1 Core Hypothesis

> **Hypothesis**: The reconstruction residual from fitting an implicit neural representation to an image contains forensic information that (a) distinguishes real from AI-generated content, (b) is invariant to lossy compression because INRs learn continuous signal representations rather than pixel-level details, and (c) provides a stable feature space for continual learning across generative model families.

**Why this is plausible:**
1. Your thesis shows INRs learn **continuous function approximations** of signals — they capture structural patterns, not high-frequency noise
2. AI-generated images lie on low-dimensional manifolds of their generators — INRs should represent them more efficiently (lower fitting residual) than real images
3. Compression destroys high-frequency details but preserves the structural signal that INRs capture
4. INR fitting residuals are **generator-agnostic** — they measure "how well can a neural network reconstruct this signal?" rather than looking for specific artifact fingerprints

### 2.2 Theoretical Foundation

From your thesis:
- **Offline INR training** on CFD data achieves PSNR 32-36 dB with extreme compression ratios
- **Online INR training** reveals catastrophic forgetting: PSNR drops to 10-13 dB
- **Experience Replay** is the only CL strategy that works for INRs (ER-Scaled: 22.65 dB PSNR)
- **Regularization methods fail** because INR parameters are tightly coupled and shared

From the detection literature:
- **DIRE** (ICCV 2023): Reconstruction error via diffusion models discriminates real from fake
- **FIRE** (CVPR 2025): Mid-frequency reconstruction error is the key signal
- **NRR** (ECCV 2024): INR fitting residuals concatenated with RGB improve manipulation detection
- **Rate-distortion theory** (Saberi et al., ICLR 2024): Fundamental limits exist, but multi-signal fusion extends practical detection range

### 2.3 Proposed Architecture: INR-FIRE (INR-based Frequency-aware Image Reconstruction for Evidence)

```
Input Image (possibly compressed)
        │
        ├──→ [INR Fitting Module]
        │         │
        │         ├── Fit SIREN/ReLU MLP to image coordinates
        │         ├── Compute reconstruction residual map
        │         └── Extract residual statistics (per-frequency-band)
        │                    │
        ├──→ [Foundation Model Backbone] (CLIP ViT-L/14 or DINOv2, frozen)
        │         │
        │         └── Extract semantic feature vector (768d or 1024d)
        │                    │
        ├──→ [Compression Estimator] (your existing module, enhanced)
        │         │
        │         ├── Estimate compression type + quality
        │         ├── Estimate compression chain depth
        │         └── Output compression feature vector (16d)
        │                    │
        ├──→ [Frequency Decomposition] (DCT + DWT)
        │         │
        │         ├── Low/Mid/High frequency band energies
        │         ├── Mid-frequency reconstruction error (FIRE-inspired)
        │         └── Spectral fingerprint features (32d)
        │                    │
        └──→ [Feature Fusion + Classification Head]
                  │
                  ├── Concatenate: INR residual (64d) + semantic (768d) + compression (16d) + frequency (32d)
                  ├── Lightweight MLP classifier
                  ├── LoRA adapters per generator family (for CL)
                  └── Output: {REAL, GAN, DIFFUSION, NEURAL_CODEC} + confidence + evidence map
```

### 2.4 Why This Is Novel

| Component | Prior Work | Our Advance |
|-----------|-----------|-------------|
| INR residuals for forensics | NRR (ECCV 2024) — concatenation only | Frequency-decomposed INR residuals with compression normalization |
| Compression-robust detection | FIRE (CVPR 2025) — diffusion reconstruction | INR reconstruction is model-agnostic (works for GANs too) |
| Continual learning for detection | GPL (ICCV 2025), DevFD (NeurIPS 2025) | ER-informed CL strategy proven on INRs (thesis contribution) |
| Compression awareness | PLADA (2025) — block effect erasure | End-to-end compression estimation + attenuation + INR normalization |
| 4-class detection | Most do binary or 3-class | REAL / GAN / DIFFUSION / NEURAL_CODEC |

---

## 3. Research Plan — Phased Execution

### Phase 0: Foundation & Validation (Weeks 1-3)

**Goal**: Validate the core hypothesis that INR residuals differ between real and AI-generated images.

**Tasks**:
1. **Port INR training code from thesis** (`unified_training_utils.py`) to work with 2D images instead of 4D spatio-temporal data
   - Adapt: `f(x, y) → (R, G, B)` instead of `f(x, y, z, t) → (Vx, Vy, P, TKE)`
   - Reuse: MLP architectures (Base/Medium/Large), training loop, metrics
   - Keep: Same model sizes for compression ratio comparison

2. **Collect validation dataset** with controlled compression:
   - 500 real images (CelebA-HQ or FFHQ)
   - 500 GAN-generated (StyleGAN2/3)
   - 500 diffusion-generated (Stable Diffusion, DALL-E 3)
   - Each at 5 compression levels: raw, JPEG QF 85, QF 65, QF 40, QF 20
   - Total: 7,500 image-INR pairs

3. **Fit INR to each image** and compute:
   - Per-pixel reconstruction residual map
   - Frequency-decomposed residual (DCT bands: low [0-8], mid [8-32], high [32+])
   - Global statistics: mean, std, skewness, kurtosis of residual per band
   - Fitting efficiency: loss curve convergence rate, final loss

4. **Statistical analysis**:
   - Do real vs. GAN vs. diffusion images have significantly different INR residual distributions?
   - How do these distributions change under compression?
   - Is there a compression-invariant discriminative signal?

**Expected outcome**: Confirm/deny that INR residuals carry forensic signal that survives compression.

**Key thesis code to reuse**:
- `/Users/mahesh/master-thesis/src/unified_training_utils.py` — model definitions, training utilities
- `/Users/mahesh/master-thesis/src/continual_learning/replay_buffer.py` — reservoir sampling
- `/Users/mahesh/master-thesis/src/continual_learning/cl_strategies.py` — CL strategy implementations

---

### Phase 1: INR Forensic Feature Extractor (Weeks 4-6)

**Goal**: Build the INR-based forensic feature extraction module.

**Tasks**:
1. **Optimize INR fitting for speed**:
   - Current thesis approach: 150 epochs offline → too slow for per-image fitting
   - Target: <1 second per image on GPU
   - Methods: Meta-learning initialization (MAML-style), reduced architecture, early stopping
   - Alternative: Use a hypernetwork that predicts INR weights from an image embedding (amortized inference)

2. **Design INR residual feature vector (64d)**:
   - Per-band (low/mid/high) residual statistics: mean, std, skew, kurtosis (12d)
   - Spatial residual pattern features: gradient magnitude, Laplacian, local variance (12d)
   - Fitting dynamics: convergence rate, loss trajectory curvature, final/initial loss ratio (8d)
   - Cross-channel residual correlations: R-G, R-B, G-B residual coherence (12d)
   - Residual frequency spectrum: DCT energy distribution of residual map (20d)

3. **Integrate with existing detector**:
   - Add `src/artifact_detectors/inr_forensic_detector.py`
   - Add INR features as a new branch in `combined_artifact_classifier.py`
   - Compare: hand-crafted only vs. hand-crafted + INR vs. INR only

**Architecture for INR fitting module**:
```python
class ForensicINR(nn.Module):
    """SIREN-based INR for forensic feature extraction.

    Adapted from thesis Base model (4→64→64→32→4) to 2D image domain.
    Input: (x, y) coordinates normalized to [-1, 1]
    Output: (R, G, B) pixel values normalized to [0, 1]
    """
    # Architecture: 2→64→64→32→3 (adaptation of thesis Base model)
    # Activation: sin (SIREN) instead of ReLU — better for image signals
    # Parameters: ~6,500 (~26KB) per image
```

---

### Phase 2: Foundation Model Backbone Integration (Weeks 7-9)

**Goal**: Replace hand-crafted feature ensemble with foundation model backbone + INR features.

**Tasks**:
1. **Set up CLIP ViT-L/14 backbone** (frozen, no training):
   - Extract 768d feature vector per image
   - Benchmark: CLIP features alone vs. current hand-crafted ensemble
   - Expected: CLIP alone will already outperform current system (based on literature: +13% on compressed data)

2. **Hybrid feature fusion**:
   - Concatenate: CLIP (768d) + INR residual (64d) + compression estimation (16d) + frequency (32d)
   - Total: 880d feature vector
   - Lightweight classification head: 880→256→64→4 (with dropout)

3. **Train on multi-source dataset**:
   - FaceForensics++ (c23 + c40)
   - CelebDF-v2
   - GenImage subset (diffusion models)
   - With compression augmentation: JPEG QF [30-100], resize [0.5x-1.5x], Gaussian blur

4. **Benchmark against literature**:
   - Compare with FIRE (CVPR 2025), D3 (CVPR 2025), UnivFD (CVPR 2023)
   - Specifically test compression robustness at each JPEG QF level
   - Report: AUC, accuracy, F1 at each compression level

---

### Phase 3: Continual Learning Integration (Weeks 10-13)

**Goal**: Enable the detector to adapt to new generative models without forgetting.

**This is where thesis expertise directly transfers.**

**Tasks**:
1. **Implement replay-based CL** (proven effective in thesis):
   - Adapt `replay_buffer.py` from thesis to store image features + labels
   - Buffer size: 10K-50K feature vectors (not raw images — much smaller)
   - Reservoir sampling for balanced temporal coverage
   - Replay weight: 0.5-0.7 (thesis optimal range)

2. **Add LoRA adapters** (inspired by DevFD, NeurIPS 2025):
   - One LoRA module per generator family (GAN, diffusion, video-gen, neural-codec)
   - Orthogonal constraint between adapters (prevent interference)
   - New generator → new LoRA adapter + replay buffer update

3. **CL evaluation protocol** (following Fontana et al., arXiv:2509.07993):
   - Chronological task ordering: StyleGAN → StyleGAN2 → Stable Diffusion → DALL-E 3 → Midjourney → FLUX → Sora
   - Metrics: C-AUC (continual AUC), FWT-AUC (forward transfer), AF (average forgetting)
   - Compare: Naive, ER, ER+LoRA, GPL-style, DevFD-style

4. **Key insight from thesis to apply**:
   - EWC and LwF will likely fail (thesis shows they fail for shared parameters)
   - Experience Replay with scaled buffer is the way to go
   - Larger models forget more → use LoRA (small adapters) rather than growing the backbone

**Why thesis CL findings transfer to detection:**

| Thesis Finding (INR Compression) | Detection Application |
|----------------------------------|----------------------|
| ER-Scaled is best for INRs | Use replay-based CL for detection, not regularization |
| Larger models forget more | Don't fine-tune full backbone; use frozen backbone + lightweight adapters |
| LwF actively harms INRs | Avoid knowledge distillation on the backbone features |
| 10K buffer sufficient for base model | Feature-space replay is memory-efficient |
| Replay weight 0.5-0.7 optimal | Same range for detection feature replay |

---

### Phase 4: Compression-Robust Pipeline (Weeks 14-16)

**Goal**: Build end-to-end compression-robust detection with social media pipeline awareness.

**Tasks**:
1. **Enhance compression estimator** (`compression_estimator.py`):
   - Add neural codec detection (JPEG AI signature identification)
   - Add compression chain depth estimation (single vs. double vs. triple compression)
   - Add codec type identification (JPEG vs. WebP vs. H.264 vs. H.265 vs. VP9 vs. AV1)
   - Integrate Bergmann et al.'s three forensic cues for JPEG AI

2. **Build social media compression emulator** (following Montibeller et al., arXiv:2508.08765):
   - Estimate platform-specific CRF and resolution parameters
   - Support: WhatsApp, Instagram, Twitter/X, TikTok, YouTube, Facebook
   - Use for training data augmentation

3. **Differentiable compression layer** (following JPEG-DL, ICLR 2025):
   - Add differentiable JPEG simulation to training pipeline
   - Enable end-to-end gradient flow through compression
   - 128 additional trainable parameters (minimal overhead)

4. **Compression-conditioned detection**:
   - Compression features (16d) modulate the classification head
   - Different detection strategies for different compression regimes
   - Adaptive thresholding based on estimated compression level

---

### Phase 5: Interview Integrity Application (Weeks 17-22)

**Goal**: Apply compression-robust detection to real-time interview analysis.

**Tasks**:
1. **Video conference codec simulation**:
   - WebRTC (VP8/VP9): typical for browser-based interviews
   - Zoom (H.264): variable bitrate, adaptive quality
   - Teams/Meet: H.264 with SVC layers
   - Build codec-specific augmentation pipeline

2. **Multi-modal signal fusion under compression**:
   - Lip-sync (SyncNet) robustness to video compression
   - Eye gaze (L2CS-Net) robustness to resolution downscaling
   - Audio (AASIST) robustness to audio codec compression (Opus, AAC)
   - INR-based visual analysis for face ROI
   - Fusion: weighted combination with compression-aware confidence

3. **Floating-point bit-level analysis** (novel research angle from CLAUDE.md):
   - Extract sign/exponent/mantissa from lip/eye ROI pixel values
   - Analyze mantissa noise statistics — AI-generated content has different mantissa distributions
   - Benford's Law on exponent bits
   - Test: Does this signal survive video compression?

4. **Real-time pipeline**:
   - Stage 1 (live): Lightweight CLIP + compression estimator → rolling confidence
   - Stage 2 (post-hoc): Full INR fitting + frequency analysis + bit-level → detailed evidence report

---

## 4. Cross-Linking: Thesis ↔ Detection Project

### 4.1 Direct Code Reuse

| Thesis Module | Detection Application | Adaptation Needed |
|--------------|----------------------|-------------------|
| `unified_training_utils.py` (MLP models) | INR forensic feature extractor | Change I/O: 4D→2D input, 4→3 output |
| `replay_buffer.py` (reservoir sampling) | CL replay buffer for detection | Store feature vectors instead of data points |
| `cl_strategies.py` (8 CL strategies) | CL for generator adaptation | Use ER/ER-Scaled; skip EWC/LwF |
| `cl_training.py` (training loop) | Incremental detector training | Adapt for classification (not regression) |
| Offline training scripts | INR fitting baseline | Adapt for image-level fitting |
| Compression ratio metrics | INR fitting efficiency features | Use as forensic feature dimension |

### 4.2 Knowledge Transfer

| Thesis Insight | Application in Detection |
|---------------|-------------------------|
| INRs learn continuous functions, not pixels | INR residuals should be compression-invariant |
| ReLU MLPs achieve 32-36 dB PSNR on complex data | Small INRs can represent images well enough to extract residuals |
| Larger models forget more severely | Use small, frozen backbones + lightweight adapters |
| ER-Scaled (50K buffer, 0.7 weight) is optimal | Apply same configuration to detection CL |
| EWC fails for tightly coupled parameters | Don't use regularization-based CL for detection backbone |
| LwF creates conflicting distillation constraints | Avoid knowledge distillation on shared features |
| Catastrophic forgetting is worse than expected | Plan for CL from the architecture stage, not as an afterthought |
| Compression ratio correlates with model capacity | INR compression ratio itself may be a forensic feature |

### 4.3 Research Contribution Mapping

The thesis provides the **compression and CL foundation**. The detection project provides the **application domain and evaluation framework**. Together:

```
THESIS CONTRIBUTIONS                    DETECTION CONTRIBUTIONS
========================               ========================
INR compression expertise        →     INR forensic features
CL strategy evaluation           →     Informed CL design
Forgetting dynamics              →     Architecture decisions
Online training framework        →     Streaming detection
Compression ratio analysis       →     Forensic feature space

                    COMBINED NOVEL CONTRIBUTIONS
                    ===========================
                    INR residuals as forensic features
                    Compression-invariant detection
                    CL-ready detector architecture
                    4-class detection (real/GAN/diffusion/neural-codec)
                    Social media pipeline robustness
```

---

## 5. Evaluation Plan

### 5.1 Datasets

| Dataset | Type | Use |
|---------|------|-----|
| FaceForensics++ (c0, c23, c40) | Face manipulation | Compression-stratified baseline |
| CelebDF-v2 | Face swap | Cross-method generalization |
| DFDC | Diverse face manipulation | Large-scale evaluation |
| GenImage | Multi-generator images | GAN + diffusion coverage |
| DiffusionForensics | Diffusion-specific | Diffusion detection benchmark |
| WildDeepfake | In-the-wild | Real-world conditions |
| Deepfake-Eval-2024 | Multi-modal, in-the-wild | Ultimate real-world benchmark |

### 5.2 Metrics

| Metric | What It Measures |
|--------|-----------------|
| AUC-ROC | Overall discrimination at each compression level |
| AP (Average Precision) | Precision-recall trade-off |
| ACC@FPR5% | Accuracy at max 5% false positive rate |
| C-AUC | Continual learning: average AUC across sequential tasks |
| FWT-AUC | Forward transfer to unseen generators |
| AF (Average Forgetting) | How much old generator detection degrades |
| Δ-AUC(comp) | AUC drop from raw to compressed (compression robustness) |

### 5.3 Ablation Studies

1. **INR residual value**: Full system vs. system without INR features
2. **Compression awareness value**: With vs. without compression estimation
3. **CL strategy comparison**: Naive vs. ER vs. ER+LoRA vs. GPL-style
4. **INR architecture**: ReLU (thesis) vs. SIREN vs. Fourier features
5. **INR fitting time**: Quality vs. speed trade-off (50/100/200 epochs)
6. **Backbone comparison**: CLIP vs. DINOv2 vs. EfficientNet-B4
7. **Compression augmentation**: None vs. single-JPEG vs. multi-pass vs. platform emulation

---

## 6. Publication Strategy

### 6.1 Target Venues

| Paper | Venue | Contribution |
|-------|-------|-------------|
| **Paper 1**: "INR Reconstruction Residuals as Compression-Invariant Forensic Features" | CVPR/ICCV 2027 | Core novelty: INR for detection + compression robustness |
| **Paper 2**: "Continual Learning for Deepfake Detection Under Realistic Compression" | NeurIPS/ICML 2027 | CL + compression intersection (unexplored in literature) |
| **Paper 3**: "From Compression to Detection: Bridging Neural Representations" | IEEE TIFS | Journal paper linking thesis + detection work |

### 6.2 Key Claims to Support

1. INR residuals provide compression-invariant forensic features (Phase 0-1 validation)
2. Experience replay is the optimal CL strategy for detection (thesis finding + detection validation)
3. The compression-detection gap can be reduced by INR-based features (Phase 2 benchmarks)
4. A unified framework can handle evolving generators AND evolving codecs (Phase 3-4 results)

---

## 7. Timeline Summary

```
WEEK  1-3:  Phase 0  — Hypothesis validation (INR residuals on real vs. fake)
WEEK  4-6:  Phase 1  — INR forensic feature extractor module
WEEK  7-9:  Phase 2  — Foundation model integration + training
WEEK 10-13: Phase 3  — Continual learning integration (thesis CL code reuse)
WEEK 14-16: Phase 4  — Compression-robust pipeline
WEEK 17-22: Phase 5  — Interview integrity application
WEEK 23-26: Paper writing + evaluation consolidation
```

---

## 8. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| INR residuals don't differ for real vs. fake | Medium | Critical | Fallback: Use FIRE + CLIP (proven methods) with CL contribution |
| INR fitting too slow for practical use | High | Major | Amortized inference via hypernetwork; pre-computed features |
| CL doesn't improve over simple retraining | Low | Moderate | Literature strongly supports CL value; thesis data confirms |
| Compression destroys INR forensic signal | Medium | Major | Mid-frequency focus (FIRE insight); test early in Phase 0 |
| New generative models make approach obsolete | Low | Low | INR features are model-agnostic by design |

---

## 9. Key References

### Foundational (Directly Used)
1. Wang et al. "DIRE for Diffusion-Generated Image Detection." ICCV 2023. https://arxiv.org/abs/2303.09295
2. Chu et al. "FIRE: Robust Detection via Frequency-Guided Reconstruction Error." CVPR 2025. https://arxiv.org/abs/2412.07140
3. Zhang et al. "Image Manipulation Detection with Implicit Neural Representation (NRR)." ECCV 2024.
4. Cozzolino et al. "Raising the Bar with CLIP." CVPRW 2024. https://arxiv.org/abs/2312.00195
5. Fontana et al. "Chronological Continual Learning for Deepfake Detection." 2025. https://arxiv.org/abs/2509.07993
6. Zhang et al. "GPL: Generalization-Preserved Learning." ICCV 2025.
7. DevFD. "Orthogonal LoRA for Face Forgery Detection." NeurIPS 2025. https://arxiv.org/abs/2509.19230
8. Pellegrini et al. "Generalized Design Choices for Deepfake Detectors." 2025. https://arxiv.org/abs/2511.21507
9. Montibeller et al. "Social Network Compression Emulation." 2025. https://arxiv.org/abs/2508.08765
10. Cannas et al. "Is JPEG AI Going to Change Image Forensics?" ICCV 2025 Workshop. https://arxiv.org/abs/2412.03261

### Compression-Robust Detection
11. Li et al. "PLADA: Block Effect Eraser for OSN Deepfakes." 2025. https://arxiv.org/abs/2506.20548
12. Yang et al. "D3: Scaling Up Deepfake Detection." CVPR 2025. https://arxiv.org/abs/2404.04584
13. Shin et al. "JPEG-DL: Differentiable JPEG Layer." ICLR 2025. https://arxiv.org/abs/2410.07081
14. Gao et al. "HiFE: High-Frequency Enhancement Network." ESWA 2024.
15. Tan et al. "NPR: Neighboring Pixel Relationships." CVPR 2024. https://arxiv.org/abs/2312.10461

### Continual Learning for Detection
16. Li et al. "CDDB: Continual Deepfake Detection Benchmark." WACV 2023. https://arxiv.org/abs/2205.05467
17. Shen et al. "DARW: Domain-Aware Generative Replay." 2025. https://arxiv.org/abs/2511.18436
18. Cheng et al. "SUR-LID: Aligned Feature Isolation." CVPR 2025.
19. Saberi et al. "Fundamental Limits of AI-Image Detection Robustness." ICLR 2024. https://arxiv.org/abs/2310.00076
20. Bergmann et al. "Three Forensic Cues for JPEG AI." 2025. https://arxiv.org/abs/2504.03191

### Implicit Neural Representations
21. Sitzmann et al. "SIREN: Implicit Neural Representations with Periodic Activations." NeurIPS 2020.
22. Dupont et al. "COIN: Compression with Implicit Neural Representations." 2021.
23. Sadupalli, M. "Concurrent Neural Network Training for Compression of Spatio-Temporal Data." M.Sc. Thesis, BTU Cottbus-Senftenberg, 2026.

---

## 10. Appendix: Detailed Literature Research Reports

Full research reports with 100+ references are available at:
- `/docs/research_compression_effects_on_detection.md` — Compression effects on detection (25 refs)
- `/docs/research_continual_learning_detection_2026.md` — CL for detection (47 refs)
- `/research/neural_compression_detection_intersection_report.md` — Novel intersection approaches (44 refs)
