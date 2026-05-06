# Research Report: Neural Data Compression and AI-Generated Content Detection -- Cutting-Edge Intersections

**Date**: 2026-03-18
**Scope**: Comprehensive survey of novel approaches at the intersection of neural/learned compression and AI-generated media detection, covering 2023-2025 literature
**Domains covered**: Learned compression forensics, compression-as-forensic-tool, adversarial robustness against compression laundering, frequency-domain forensics, reconstruction-based detection (DIRE family), feature distillation for lightweight detection, implicit neural representations, information-theoretic detection limits

---

## Executive Summary

The convergence of neural data compression and AI-generated content detection represents one of the most consequential frontiers in media forensics. Three developments define the current landscape:

**First**, the JPEG AI standard (ratified February 2025) introduces neural image compression that creates artifacts nearly indistinguishable from those of generative models (GANs, diffusion models), fundamentally breaking the assumption that compression artifacts and generation artifacts occupy different feature spaces. Bergmann et al. (2025) have identified three forensic cues specific to JPEG AI, but the problem remains largely open.

**Second**, reconstruction-based detection methods -- DIRE (ICCV 2023), LaRE2 (CVPR 2024), AEROBLADE (CVPR 2024), and FIRE (CVPR 2025) -- have emerged as the dominant paradigm for diffusion-generated image detection. These methods exploit the observation that diffusion models reconstruct their own outputs more faithfully than real images. FIRE advances this by showing that mid-frequency reconstruction error is the most discriminative signal, and it survives common compression perturbations better than pixel-domain approaches.

**Third**, information-theoretic analysis (Saberi et al., ICLR 2024) has established fundamental trade-offs between detection reliability and robustness: as generative model quality improves (i.e., real and fake distributions converge), no detector can simultaneously achieve high accuracy and high robustness to perturbations including compression. This result has profound implications for the long-term viability of passive detection.

Real-world benchmarks (Deepfake-Eval-2024, SocialDF, So-Fake) now confirm that detector performance drops 40-50% when evaluated on content that has passed through social media compression pipelines, compared to lab-quality benchmarks. The gap between academic evaluation and deployment reality remains the central practical challenge.

---

## 1. Learned Image/Video Compression: How Neural Codecs Differ from Traditional Codecs

### 1.1 The JPEG AI Standard and Its Forensic Implications

The JPEG AI standard, published in February 2025, is the first international image coding standard based on end-to-end neural network learning. It achieves impressive image quality at bitrates an order of magnitude lower than traditional JPEG, using a learned autoencoder architecture with a hyperprior entropy model and quantized latent representations.

**Cannas et al. (2024)** ["Is JPEG AI going to change image forensics?" arXiv:2412.03261, also at ICCV 2025 Workshop] conducted the first systematic study of JPEG AI's impact on forensic detectors. Key findings:
- Pristine images compressed with JPEG AI are misclassified as deepfakes by state-of-the-art detectors
- Splicing localization algorithms produce false alarms on JPEG AI content
- Traditional JPEG forensic tools (double compression detection, quantization table analysis) do not transfer to JPEG AI
- The neural decoder's upsampling operations produce grid-pattern artifacts in the frequency domain that closely resemble GAN upsampling artifacts

**Bergmann et al. (2024)** ["Forensic analysis of AI-compression traces in spatial and frequency domain," Pattern Recognition Letters, 2024] characterized the artifacts left by neural codecs including HiFiC and Balle et al.'s hyperprior model:
- Both spatial and frequency domain artifacts stem primarily from upsampling operations in the neural decoder
- HiFiC (which uses a GAN-based decoder) produces frequency-domain grid artifacts nearly identical to GAN-generated images
- The similarity between neural compression artifacts and generative model artifacts is a fundamental challenge, not an incidental one, because both use similar decoder architectures

**Bergmann, Brand, and Riess (2025)** ["Three Forensic Cues for JPEG AI Images," arXiv:2504.03191] proposed three interpretable forensic cues specific to JPEG AI:

| Cue | Mechanism | Best Accuracy | Limitation |
|-----|-----------|---------------|------------|
| Color channel correlations | YUV preprocessing introduces measurable RGB correlations | 85.6% at 0.06 bpp | Weaker at high bitrates |
| Rate-distortion recompression | Repeated JPEG AI compression shows diminishing PSNR changes | 87-92% for strong compression | Fails above 0.75 bpp |
| Latent quantization features | JPEG AI quantizer leaves traces absent in generative models | 91.7-98.5% | Degrades with JPEG postprocessing |

The third cue is particularly important: it can distinguish JPEG AI-compressed real images from AI-generated images (94-98% accuracy for diffusion models) because generative models lack the quantization step present in JPEG AI's encoding pipeline.

### 1.2 Neural Video Codecs

Neural video compression has reached parity with H.266/VVC:
- **DCVC-HEM** and **DCVC-RT** surpass VVC in rate-distortion performance (Li et al., CVPR 2023, 2024)
- **Cool-Chic** achieves competitive compression with only ~800 parameters per image/frame
- **YODA** uses one-step diffusion for video compression, embedding multi-scale temporal features

However, **no published work** has yet systematically studied the forensic implications of neural video codecs -- this represents a significant research gap. The forensic challenges documented for JPEG AI (decoder upsampling artifacts mimicking generative model signatures) likely extend to neural video codecs, but this remains unconfirmed.

### 1.3 Key Insight: The Decoder Architecture Problem

The fundamental issue is architectural convergence: neural codecs and generative models increasingly share the same building blocks (learned upsampling, transposed convolutions, attention mechanisms). This means:
- Forensic features that detect "generation artifacts" will increasingly fire on "compression artifacts"
- The feature space for distinguishing "compressed real" from "generated fake" is shrinking
- Future forensic methods must look beyond decoder-side artifacts to encoder-side or latent-space cues

---

## 2. Compression as a Forensic Tool

### 2.1 Double JPEG Compression Detection

Double JPEG compression detection remains a cornerstone of image forensics, now extending to new compression standards:

**Multi-branch network for double JPEG detection** (Scientific Reports, 2025): FF-Net learns JPEG compression fingerprints by analyzing high-frequency image components, using a multi-branch CNN architecture to detect and localize double compression.

**Dual JPEG Compatibility** (Bianchi and Piva, arXiv:2408.17106, 2024): A principled approach that finds the JPEG antecedent of 8x8 blocks; if an antecedent exists, the block is "compatible." This enables detection of inpainting, copy-move, and splicing, and critically, it works even when a second compression is applied after manipulation (provided the second quality > first quality).

**Double compression detection for HEIF** (2024): Extends detection to HEIF/HEIC using "coding ghosts" -- the first work to address double compression in the HEIF standard used by modern smartphones.

### 2.2 Codec Fingerprinting

**Forensic Recognition of Codec-Specific Image Compression Artefacts** (ACM IH&MMSec 2024): Demonstrates that different codec implementations leave distinguishable fingerprints even at the same quality level, enabling identification of which specific encoder processed an image.

**Device-Specific Fingerprinting Beyond PRNU** (2024): A new robust fingerprint extracted from low- and mid-frequency bands (not high-frequency like PRNU) that is highly resilient to aggressive JPEG compression, rotation, and gamma correction. This is significant because traditional PRNU fingerprints are destroyed by heavy compression.

### 2.3 Neural Codec Forensics for Audio

**"Unmasking Neural Codecs: Forensic Identification of AI-compressed Speech"** (Interspeech 2024, Moussa et al.): Demonstrates that neural audio codecs (Encodec, SoundStream, etc.) leave identifiable forensic traces, enabling source identification of audio that has passed through different neural compression pipelines. This work is directly relevant to audio deepfake detection, as voice cloning systems often use neural codecs internally.

---

## 3. Adversarial Robustness Against Compression-Based Laundering

### 3.1 Fundamental Limits (Information-Theoretic)

**Saberi et al. (ICLR 2024)** ["Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks," arXiv:2310.00076] established the most important theoretical result in this space:

- **For watermarking-based detection** with low perturbation budgets: there exists a fundamental trade-off between the evasion error rate (watermarked images evading detection) and the spoofing error rate (non-watermarked images falsely flagged). Diffusion purification attacks exploit this trade-off.
- **For classifier-based deepfake detectors**: there is a fundamental trade-off between robustness and reliability. As the distributions of real and fake images grow more similar (as generative models improve), a detector can achieve either good performance or high robustness, but not both simultaneously.
- **Practical attacks demonstrated**: (a) diffusion purification removes low-perturbation watermarks; (b) model substitution attacks remove high-perturbation watermarks; (c) spoofing attacks add watermark signatures to real images using only black-box access.

This result implies that passive detection alone cannot be a long-term solution as generative model quality continues to improve.

### 3.2 Compression as a Laundering Vector

Compression naturally acts as a laundering operation because it:
- Destroys high-frequency forensic traces (GAN fingerprints, noise patterns)
- Removes subtle pixel-level artifacts that detectors rely on
- Applies lossy quantization that smooths statistical anomalies
- Introduces its own artifacts that can mask generation artifacts

**Social media platforms** apply particularly aggressive compression:
- Instagram: JPEG quality ~70-85, resolution capping
- WhatsApp: JPEG quality ~60-70 with resolution downscaling
- TikTok: H.264/H.265 transcoding with variable bitrate
- YouTube: VP9/AV1 re-encoding
- Facebook: Adaptive quality compression

### 3.3 Diffusion-Based Anti-Forensic Attacks

**Evading DeepFake Detectors via Conditional Diffusion Models** (ACM IH&MMSec 2024): Fake images are encoded into latent space, optimized to fool detectors while preserving visual quality, then reconstructed using conditional diffusion -- effectively "laundering" the image through a diffusion process.

**StealthDiffusion** (ACM MM 2024, arXiv:2408.05669): Uses diffusion models to specifically target and evade diffusion forensic detectors, creating a cat-and-mouse dynamic within the diffusion model family itself.

**AADD-2025 Challenge** (ACM MM 2025): The "Adversarial Attacks on Deepfake Detectors" challenge tasked participants with generating adversarial examples that evade four diverse classifiers simultaneously while preserving structural similarity -- demonstrating the practical feasibility of evasion attacks.

### 3.4 Proactive Defense: Semi-Fragile Watermarking

A growing body of work uses proactive watermarking as a complement to passive detection:

**FractalForensics** (arXiv:2504.09451, 2025): Fractal watermarks with a parameter-driven generation pipeline; semi-fragile framework robust against benign image processing (compression, resize) but fragile to deepfake manipulation.

**DeepForgeSeal** (arXiv:2511.04949, 2025): Latent-space watermarking using Multi-Agent Adversarial Reinforcement Learning (MAARL) for adaptive robustness.

**WaveGuard** (arXiv:2505.08614, 2025): Embeds watermarks in high-frequency sub-bands via dual-tree complex wavelet transform (DT-CWT) for robustness against distortions and forgeries.

---

## 4. Frequency-Domain Forensics That Survive Compression

### 4.1 DCT-Based Approaches

**High-Frequency Enhancement Network (HiFE)** (Expert Systems with Applications, 2024): A three-branch architecture:
- Basic branch: RGB domain features
- Local HFE branch: Block-wise DCT coefficients with channel attention for adaptive frequency-aware spatial attention
- Global HFE branch: Multi-level DWT for multi-scale high-frequency cues

Key contribution: adaptively recovers high-frequency details lost during compression, significantly improving detection on highly compressed content.

### 4.2 Wavelet-Based Approaches

**Wavelet-CLIP** (arXiv:2409.18301, 2024): Integrates wavelet transforms with ViT-L/14 CLIP features, deeply analyzing both spatial and frequency features. The wavelet decomposition isolates frequency bands that are differentially affected by generation vs. compression.

**Interactive Dual-Branch Network (IDBN)** (Journal of Supercomputing, 2025): Adversarial knowledge distillation with CNN spatial branch + Wavelet Attention-Based Transformer (WABT) frequency branch, specifically designed for compressed deepfake detection.

### 4.3 NPR: Neighboring Pixel Relationships

**Tan et al. (CVPR 2024)** ["Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection"]: Introduced NPR (Neighboring Pixel Relationships), which captures structural artifacts from upsampling at the pixel level rather than in the frequency domain. Achieves 93.3% mean accuracy across GAN and diffusion sources when trained only on ProGAN. Importantly, NPR focuses on local pixel interdependencies that can partially survive compression because they encode structural rather than high-frequency information.

### 4.4 Forensic Self-Descriptions

**Nguyen et al. (CVPR 2025)** ["Forensic Self-Descriptions Are All You Need for Zero-Shot Detection, Open-Set Source Attribution, and Clustering of AI-generated Images"]: Learns diverse predictive filters in a self-supervised manner to extract residuals that constitute unique forensic "self-descriptions" for each image. Achieves:
- Zero-shot detection: 0.972 average AUC across JPEG quality factors 50-100
- Maintains 0.972 AUC even at JPEG QF=60
- Enables open-set source attribution and unsupervised clustering

This represents state-of-the-art compression robustness for a zero-shot method.

---

## 5. Reconstruction-Based Detection Methods (DIRE Family)

### 5.1 Evolution of Reconstruction Error Methods

| Method | Venue | Year | Key Innovation | Speed | Compression Robustness |
|--------|-------|------|---------------|-------|----------------------|
| **DIRE** | ICCV | 2023 | Pixel-space reconstruction error via diffusion | Slow (full diffusion) | High (robust to JPEG, Gaussian blur) |
| **SeDID** | -- | 2023 | Stepwise denoising error at specific timesteps | Very slow | Limited (sensitive to hyperparams) |
| **AEROBLADE** | CVPR | 2024 | Autoencoder reconstruction error only (no diffusion) | Fast (single AE pass) | Moderate |
| **LaRE2** | CVPR | 2024 | Latent-space reconstruction error + error-guided refinement | 8x faster than DIRE | Good |
| **FIRE** | CVPR | 2025 | Frequency-guided reconstruction error (mid-band focus) | Moderate | Strong (frequency decomposition preserves cues) |
| **Snap-Back** | arXiv | 2025 | Multi-strength reconstruction dynamics analysis | Moderate | 0.993 AUROC under compression |

### 5.2 DIRE (Wang et al., ICCV 2023)

DIRE measures the error between an input image and its reconstruction via a pre-trained diffusion model. Diffusion-generated images can be approximately reconstructed while real images cannot. Claims 99.9% accuracy and 100% AP across diffusion models, with robustness to JPEG compression and Gaussian blur.

**Limitation**: Requires full forward-backward diffusion process, making it computationally expensive. Also, robustness claims were tested on relatively mild perturbations.

### 5.3 LaRE2 (Luo et al., CVPR 2024)

LaRE2 moves reconstruction error computation to the latent space of the VAE encoder, achieving 8x speedup over DIRE while surpassing SOTA by up to 11.9%/12.1% average accuracy/AP on GenImage benchmark across 8 generators. The Error-Guided feature REfinement module (EGRE) uses the reconstruction error signal to enhance discriminative features.

### 5.4 AEROBLADE (Ricker et al., CVPR 2024)

Training-free detection using only the autoencoder component of latent diffusion models. Achieves 0.992 mean AP across Stable Diffusion, Kandinsky, and Midjourney. Key advantage: no training required, no costly diffusion forward-backward pass. Vulnerability: relies on the specific autoencoder, so generalization to non-LDM generators is limited.

### 5.5 FIRE (Chu et al., CVPR 2025)

The current state-of-the-art in reconstruction-based detection. Key insight: diffusion models struggle specifically to reconstruct mid-band frequency information in real images. By comparing reconstruction error before and after frequency decomposition, FIRE achieves:
- Near-perfect AUC on standard benchmarks (100% on DiffusionForensics)
- Strong generalization to unseen models (86.5% AUC on Kandinsky 3)
- Robustness to JPEG compression across quality factors
- Mid-frequency reconstruction loss contributes 24 percentage points to AUC (ablation: 54.3% without it vs. 78.3% with)

**Weakness**: Performance degrades under severe Gaussian blur because blur destroys the very mid-frequency information FIRE relies on.

### 5.6 Diffusion Snap-Back (arXiv:2511.00352, 2025)

Novel approach: instead of measuring reconstruction error at a single noise level, analyzes how reconstruction metrics (LPIPS, SSIM, PSNR) evolve across varying noise strengths. Real images degrade abruptly under increasing noise, while AI-generated ones degrade smoothly, reflecting their alignment with the learned model manifold. Achieves 0.993 AUROC and remains robust to compression and noise.

### 5.7 How Compression Affects Reconstruction-Based Methods

Compression introduces a challenge for all reconstruction-based methods because it adds a deterministic distortion component to the reconstruction error. However, the methods show differential robustness:
- Frequency-domain methods (FIRE) are most robust because compression primarily destroys high-frequency information while FIRE targets mid-frequencies
- Latent-space methods (LaRE2, AEROBLADE) are moderately robust because the VAE encoder provides some invariance to pixel-level perturbations
- Pixel-space methods (DIRE) are least robust to severe compression but survive mild compression well

---

## 6. Feature Distillation and Knowledge Compression for Lightweight Detection

### 6.1 Model Compression Techniques

**Karathanasis et al. (arXiv:2504.21066, 2025)** surveyed compression and transfer learning for deepfake detection:
- Pruning, knowledge distillation, and quantization can achieve 90% compression while maintaining performance when test data matches training distribution
- **Critical finding**: generalization to unseen generators degrades significantly after compression, revealing a tension between model efficiency and cross-generator robustness

### 6.2 Spatial-Frequency Knowledge Distillation

**Spatial-Frequency Feature Fusion via Knowledge Distillation** (Engineering Applications of AI, 2024): Teacher model learns rich spatial-frequency representations; student model retains forensic discriminability through distillation, with strong generalization on compressed images.

### 6.3 Dual-Branch Adversarial Distillation

**IDBN** (Journal of Supercomputing, 2025): Teacher and student models each contain a CNN spatial branch + wavelet attention transformer frequency branch. Adversarial knowledge distillation forces the student to learn robust compressed-domain features from the teacher.

### 6.4 Audio Deepfake Detection

**DK-CAST** (Discover Computing, 2025): Dynamic Knowledge Condensation with Audio-Selective Transformer. High-capacity XLS-R teacher trained on clean speech supervises a compact student operating on degraded audio -- directly addressing the compression/quality gap in audio forensics.

### 6.5 Federated Distillation

**Federated Knowledge Distillation (FedKD)** (2025): Combines lightweight model deployment with privacy preservation; student models gain teacher-level performance without sharing raw training data. Enables distributed deepfake detection across organizations.

### 6.6 Practical Deployment Numbers

**Pellegrini et al. (2025)** ["Generalized Design Choices for Deepfake Detectors," arXiv:2511.21507] provide the most comprehensive study of deployment-relevant design choices:
- DINOv2 backbone: 97.36% AUROC (best overall, but heavy)
- ViT-L CLIP: 94.6% AUROC (good accuracy-efficiency trade-off)
- ResNet-50 CLIP: 93.2% AUROC (lightest, suitable for edge deployment)
- Harmonic replay strategy reduces training compute by 36-40% vs. full retraining while maintaining performance for continual learning scenarios

---

## 7. Implicit Neural Representations in Forensics

### 7.1 INR-Based Steganography (Attack Vector)

Implicit Neural Representations have been primarily explored as steganographic tools, representing a potential attack vector rather than a defensive forensic tool:

**StegaINR** (2024): First work using INR for steganography -- embeds secret functions into stego functions that serve as both message extractors and stego media.

**INRSteg** (2024): Cross-modal steganography framework using INR with masked parameter updates, enabling hiding data across images, audio, video, and 3D shapes.

**U-INR** (arXiv:2505.01749, 2025): Unified steganography via INR supporting images, videos, audio, SDF, and NeRF representations.

**Noise-NeRF** (2024): NeRF steganography using trainable noise -- only updates input noise without changing weights, preserving rendering quality.

### 7.2 INR as Detection Features (Emerging)

The use of INR as a detection feature is still nascent, but the principle is promising: if an image can be efficiently represented by a compact INR, this may indicate it lies on a low-dimensional manifold consistent with neural generation. This connects to the reconstruction-based detection paradigm (Section 5), where the efficiency of neural representation serves as the discriminating signal.

### 7.3 Forensic Implications

INR-based steganography poses a new challenge for forensics:
- Traditional steganalysis methods may not detect data hidden in neural network weights
- INR steganography can survive format conversions and moderate compression because the hidden data is encoded in the network architecture, not pixel-level statistics
- Multi-modal hiding (simultaneously across image, audio, video) creates new cross-modal forensic challenges

---

## 8. Data Compression Theory and Detection Reliability

### 8.1 Rate-Distortion-Perception Trade-off

The **rate-distortion-perception (RDP) framework** (Blau and Michaeli, 2019; extended 2024-2025) formalizes the three-way trade-off between:
- **Rate**: bits per pixel for compression
- **Distortion**: pixel-level fidelity (MSE, PSNR)
- **Perception**: distributional fidelity (FID, IS)

This has direct implications for forensics: at low rates, neural codecs must sacrifice either distortion or perception, and the choice determines which forensic traces survive. Codecs optimized for perception (like HiFiC) produce outputs that look realistic but have altered statistical properties -- these can be confused with generative model outputs.

### 8.2 Fundamental Detection Limits

Building on Saberi et al. (ICLR 2024), the key theoretical insight is:

**As generative models improve** (real/fake distributions converge in total variation distance):
- The best achievable detection accuracy approaches 50% (random)
- Any robustness guarantee against perturbations (including compression) further reduces achievable accuracy
- Watermarking offers stronger guarantees than passive detection, but faces its own evasion/spoofing trade-offs

**Practical implication**: For the interview integrity platform described in this project's CLAUDE.md, this means passive visual deepfake detection will become unreliable as a standalone signal. Multi-modal fusion (lip-sync + gaze + audio + visual) is essential, and proactive measures (watermarking, provenance) should be pursued in parallel.

### 8.3 Geometry of Latent Representations

**"The geometry of efficient codes"** (PLOS Computational Biology, 2025): Investigates how rate-distortion trade-offs shape latent representations in variational autoencoders. Under strong rate constraints, latent representations become more categorical and less informative about fine details -- this has implications for both neural compression artifacts and detection feature robustness.

---

## 9. Compression-Robust Benchmarks and Evaluation Protocols

### 9.1 In-the-Wild Benchmarks

| Benchmark | Year | Scale | Compression | Key Finding |
|-----------|------|-------|-------------|-------------|
| **Deepfake-Eval-2024** | 2025 | 44h video, 56.5h audio, 1975 images | Real social media (88 websites) | SOTA AUC drops 50% (video), 48% (audio), 45% (image) vs. lab benchmarks |
| **SocialDF** | 2025 | Social media content | Platform-native compression | Detection model for mitigating harmful deepfake content on social platforms |
| **So-Fake** | 2025 | Two components: controlled + OOD | Social media oriented | So-Fake-Set (in-domain) + So-Fake-OOD (cross-domain robustness) |
| **AI-GenBench** | 2024-25 | Multi-generator | Augmented with compression | Used by Pellegrini et al. for design choice evaluation |

### 9.2 Social Network Compression Emulation

**Montibeller et al. (2025)** ["Bridging the Gap: A Framework for Real-World Video Deepfake Detection via Social Network Compression Emulation," arXiv:2508.08765]:
- Estimates CRF (Constant Rate Factor) and resolution changes from uploaded-then-downloaded videos
- Requires ~30 shared videos per resolution for stable parameter estimation
- Stores (input resolution, output resolution, estimated CRF) triplets
- Emulated videos closely match real platform degradation
- **Key result**: Detectors fine-tuned on emulated data achieve comparable performance to those trained on actually-shared media
- Platforms studied: Facebook, YouTube, BlueSky
- Open-source: github.com/truebees-ai/social_emulator

### 9.3 Compression-Aware Training Protocols

**Best practices from Pellegrini et al. (2025)**:
- Apply up to 3 successive JPEG compression passes with variable quality (30-100) during training
- Include random resized cropping, color jitter, Gaussian noise, blurring
- Evaluation-based augmentation pipeline (mimicking realistic post-processing) outperforms aggressive augmentation: 94.5% vs. 90.1% AUROC
- DINOv2 backbone converges in 1-2 epochs; ResNet-50 needs more training

**D3 framework (CVPR 2025)**: Takes distorted image features as "discrepancy signals" in a parallel branch, achieving +5.3% OOD accuracy improvement. Directly addresses the challenge of learning from compression artifacts rather than being confused by them.

---

## 10. Provenance and Watermarking Standards

### 10.1 C2PA / Content Credentials (v2.2, 2025)

- First international standard for content provenance metadata
- Combines hard binding (cryptographic hashing) with soft binding (watermarking/fingerprinting)
- **C2PA 2.1+**: Digital watermarks enable content credentials to survive social media upload/download cycles
- **Limitation**: Metadata-based approach can be stripped; soft binding adds robustness but not full reliability
- Expected ISO standardization by 2025

### 10.2 Google SynthID

- Embeds imperceptible watermarks into AI-generated content across text, images, audio, video
- Co-trained embedder and detector with adversarial transformations (JPEG compression, filters, rotation, noise, resizing)
- SynthID Text open-sourced October 2024 (Hugging Face)
- Unified SynthID Detector released May 2025 for cross-modal verification

### 10.3 Meta Video Seal

- Open-source neural video watermarking (December 2024)
- Temporal watermark propagation: converts image watermarking to video without watermarking every frame
- Specifically designed for robustness to video compression (H.264, H.265, VP9, AV1)
- Deployed at production scale at Meta (November 2025)
- Addresses the critical gap where prior video watermarking methods lacked codec robustness

---

## 11. Continual Learning for Evolving Generative Models

### 11.1 The Catastrophic Forgetting Problem in Detection

As new generative models emerge (Midjourney v6, DALL-E 3, Sora, Stable Diffusion 3, FLUX), detectors must adapt without forgetting how to detect older generators.

**Zhang et al. (ICCV 2025)** ["Generalization-Preserved Learning: Closing the Backdoor to Catastrophic Forgetting in Continual Deepfake Detection"]:
- Hyperbolic Visual Alignment: learnable watermarks align incremental data with base set in hyperbolic space
- Generalized Gradient Projection: prevents parameter updates conflicting with generalization constraints
- 92.14% accuracy, outperforming replay-based SOTA by 2.15%, reducing forgetting by 2.66%

**MSSM** (IEEE Trans., 2025): Multi-Perspective Sample Selection evaluating prediction error, temporal instability, and sample diversity for exemplar memory bank management.

**Pellegrini et al. (2025)**: Harmonic replay achieves near-baseline performance with 36-40% compute reduction vs. full retraining.

### 11.2 The Non-Universal Distribution Hypothesis

**"Revisiting Deepfake Detection: Chronological Continual Learning"** (arXiv:2509.07993, 2025): Proposes that each deepfake generator leaves a unique, non-transferable signature, meaning that detection must be continuously updated. This contradicts the hope for a single universal detector and supports the continual learning paradigm.

---

## 12. Cross-Domain Connections

### 12.1 Compression <-> Detection Reliability
- Neural compression artifacts resemble generation artifacts (Section 1), creating false positives
- Compression destroys forensic traces (Section 3), creating false negatives
- These opposing effects mean compression simultaneously increases both error types
- Rate-distortion theory provides theoretical bounds on this trade-off (Section 8)

### 12.2 Reconstruction-Based Detection <-> Compression
- Methods like FIRE and LaRE2 are inherently connected to compression: they measure how well a model can "compress and decompress" an image
- The reconstruction error IS essentially a rate-distortion measurement at a specific operating point
- This connection could be exploited: using the same neural codec both for compression-aware training and as a reconstruction-based detector

### 12.3 Continual Learning <-> Compression Robustness
- New generators may produce content with different compression interaction profiles
- A detector robust to compression for GANs may not be robust for diffusion model content after compression
- Continual learning must account for evolving compression pipelines (JPEG -> JPEG AI, H.265 -> AV1 -> neural codecs)

### 12.4 Watermarking <-> Neural Compression
- JPEG AI's latent space quantization provides a natural embedding channel for watermarks
- C2PA soft bindings could potentially be embedded in neural codec latent spaces
- Meta Video Seal is already designed for neural-era video compression robustness

---

## 13. Open Problems and Research Gaps

### 13.1 Critical Gaps

1. **Neural video codec forensics**: No systematic study of forensic traces in neural video compression. As neural video codecs approach deployment (DCVC-RT matches VVC performance), this is urgent.

2. **Joint compression-generation artifact disentanglement**: No robust method exists to separate neural compression artifacts from generative model artifacts when both use similar decoder architectures.

3. **Cross-codec forensic transfer**: Methods that detect JPEG AI compression may not generalize to other neural codecs (HiFiC, ELIC, etc.). No unified neural codec forensic framework exists.

4. **Real-time detection under compression**: Most robust detection methods (FIRE, DIRE, LaRE2) require neural network inference at detection time, making real-time deployment challenging for video applications.

5. **Adversarial robustness under realistic compression chains**: Most adversarial robustness evaluations use single compression operations, but real-world content undergoes chains of compression (capture -> edit -> upload -> transcode -> download).

### 13.2 Promising Underexplored Directions

1. **Latent-space forensics**: Bergmann et al.'s quantization cue (Section 1.1) shows that latent-space analysis can distinguish compression from generation. This should be extended to all neural codecs.

2. **Compression-aware reconstruction error**: Train reconstruction-based detectors (DIRE family) with explicit compression simulation in the pipeline, potentially using differentiable approximations of JPEG/JPEG AI.

3. **Multi-modal compression interaction**: How does compression affect lip-sync, gaze, and audio forensic signals differently? This is directly relevant to the interview integrity platform.

4. **INR-based forensic features**: Use implicit neural representation fitting efficiency as a proxy for image "naturalness" -- images on generative model manifolds should be more efficiently representable.

5. **Provenance-preserving neural codecs**: Design neural codecs that explicitly preserve forensic traces or watermarks in their latent space, rather than treating forensics as an afterthought.

---

## 14. Recommendations for This Project

Based on this research, specific recommendations for the AI-generated media detection project:

### Immediate (Phase 1 -- Diffusion Detection)
1. **Implement FIRE-style frequency-guided reconstruction error** alongside DIRE. FIRE's mid-frequency focus provides better compression robustness than pixel-space DIRE.
2. **Add compression augmentation** to training: multiple JPEG quality factors (30-100), with up to 3 successive compression passes per Pellegrini et al.'s findings.
3. **Incorporate LaRE2's latent-space reconstruction** for efficient diffusion detection (8x faster than DIRE).

### Medium-term (Phase 2-3 -- Deep Learning + Interview Detection)
4. **Use DINOv2 or CLIP-ViT backbone** rather than EfficientNet-B4. Pellegrini et al. show DINOv2 achieves 97.36% AUROC with minimal fine-tuning.
5. **Build a social media compression emulator** following Montibeller et al.'s framework for training data augmentation.
6. **For lip-sync and gaze detection**, investigate how compression affects these signals differently and build compression-robust feature extractors for each modality.

### Long-term (Phase 4+ -- Production)
7. **Integrate C2PA / content provenance** as a complementary signal alongside passive detection.
8. **Plan for JPEG AI**: As neural codecs enter mainstream use, the current detector will need updating. Monitor the Three Forensic Cues work for integration.
9. **Implement continual learning** (GPL framework from ICCV 2025) to handle new generative models without full retraining.
10. **Consider watermarking for the interview platform**: Semi-fragile watermarking (FractalForensics-style) could mark original video streams, enabling detection of tampering.

---

## References

### Learned Compression and Forensics
1. Cannas, E.D. et al. "Is JPEG AI going to change image forensics?" arXiv:2412.03261, 2024. ICCV 2025 Workshop. https://arxiv.org/abs/2412.03261
2. Bergmann, S., Brand, F., and Riess, C. "Three Forensic Cues for JPEG AI Images." arXiv:2504.03191, 2025. https://arxiv.org/abs/2504.03191
3. Bergmann, S. et al. "Forensic analysis of AI-compression traces in spatial and frequency domain." Pattern Recognition Letters, 2024. https://doi.org/10.1016/j.patrec.2024.02.015
4. Moussa et al. "Unmasking Neural Codecs: Forensic Identification of AI-compressed Speech." Interspeech 2024.

### Reconstruction-Based Detection
5. Wang, Z. et al. "DIRE for Diffusion-Generated Image Detection." ICCV 2023. https://arxiv.org/abs/2303.09295
6. Luo, Y. et al. "LaRE2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection." CVPR 2024. https://arxiv.org/abs/2403.17465
7. Ricker, J. et al. "AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error." CVPR 2024. https://arxiv.org/abs/2401.17879
8. Chu, B. et al. "FIRE: Robust Detection of Diffusion-Generated Images via Frequency-Guided Reconstruction Error." CVPR 2025. https://arxiv.org/abs/2412.07140
9. "Detecting AI-Generated Images via Diffusion Snap-Back Reconstruction: A Forensic Approach." arXiv:2511.00352, 2025. https://arxiv.org/abs/2511.00352

### Adversarial Robustness and Fundamental Limits
10. Saberi, M. et al. "Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks." ICLR 2024. https://arxiv.org/abs/2310.00076
11. "Evading DeepFake Detectors via Conditional Diffusion Models." ACM IH&MMSec 2024.
12. "StealthDiffusion: Towards Evading Diffusion Forensic Detection through Diffusion Model." ACM MM 2024. https://arxiv.org/abs/2408.05669
13. "Adversarial Attacks on Deepfake Detectors: A Challenge in the Era of AI-Generated Media (AADD-2025)." ACM MM 2025.

### Frequency-Domain Forensics
14. "DeepFake detection based on high-frequency enhancement network for highly compressed content." Expert Systems with Applications, 2024.
15. "Harnessing Wavelet Transformations for Generalizable Deepfake Forgery Detection." arXiv:2409.18301, 2024.
16. Tan, C. et al. "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection." CVPR 2024. https://arxiv.org/abs/2312.10461
17. Nguyen et al. "Forensic Self-Descriptions Are All You Need for Zero-Shot Detection, Open-Set Source Attribution, and Clustering of AI-generated Images." CVPR 2025. https://arxiv.org/abs/2503.21003

### Compression Forensics
18. "Multi-branch network for double JPEG detection and localization." Scientific Reports, 2025.
19. "Dual JPEG Compatibility: a Reliable and Explainable Tool for Image Forensics." arXiv:2408.17106, 2024.
20. "Double Compression Detection of HEIF Images Using Coding Ghosts." 2024.
21. "Forensic Recognition of Codec-Specific Image Compression Artefacts." ACM IH&MMSec 2024.

### Knowledge Distillation and Lightweight Detection
22. Karathanasis, A. et al. "A Brief Review for Compression and Transfer Learning Techniques in DeepFake Detection." arXiv:2504.21066, 2025.
23. "Spatial-frequency feature fusion based deepfake detection through knowledge distillation." Engineering Applications of AI, 2024.
24. "Interactive dual-branch network based on adversarial knowledge distillation for compressed deepfake detection." Journal of Supercomputing, 2025.
25. "Dynamic knowledge condensation with audio-selective transformer for audio deepfake detection." Discover Computing, 2025.

### Benchmarks and Evaluation
26. Chandra, N.A. et al. "Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024." arXiv:2503.02857, 2025.
27. "SocialDF: Benchmark Dataset and Detection Model for Mitigating Harmful Deepfake Content on Social Media Platforms." ACM MAD Workshop, 2025.
28. "SO-FAKE: Benchmarking Social Media Image Forgery Detection." arXiv:2505.18660, 2025.
29. Montibeller, A. et al. "Bridging the Gap: A Framework for Real-World Video Deepfake Detection via Social Network Compression Emulation." arXiv:2508.08765, 2025.

### Design Choices and Training Strategies
30. Pellegrini, L. et al. "Generalized Design Choices for Deepfake Detectors." arXiv:2511.21507, 2025.
31. Yang et al. "D3: Scaling Up Deepfake Detection by Learning from Discrepancy." CVPR 2025. https://arxiv.org/abs/2404.04584

### Continual Learning
32. Zhang et al. "Generalization-Preserved Learning: Closing the Backdoor to Catastrophic Forgetting in Continual Deepfake Detection." ICCV 2025.
33. "Continual Deepfake Detection Based on Multi-Perspective Sample Selection Mechanism." IEEE Transactions, 2025.
34. "Revisiting Deepfake Detection: Chronological Continual Learning and the Limits of Generalization." arXiv:2509.07993, 2025.

### Provenance and Watermarking
35. C2PA Technical Specification v2.2. https://spec.c2pa.org/specifications/specifications/2.2/specs/C2PA_Specification.html
36. "SynthID-Image: Image watermarking at internet scale." arXiv:2510.09263, 2025.
37. "Video Seal: Open and Efficient Video Watermarking." Meta AI, 2024. https://ai.meta.com/research/publications/video-seal-open-and-efficient-video-watermarking/
38. "FractalForensics: Proactive Deepfake Detection and Localization via Fractal Watermarks." arXiv:2504.09451, 2025.

### Universal Detection
39. Ojha, U., Li, Y., and Lee, Y.J. "Towards Universal Fake Image Detectors that Generalize Across Generative Models." CVPR 2023. https://arxiv.org/abs/2302.10174

### Rate-Distortion Theory
40. "Rate-Distortion-Perception Trade-Off in Information Theory, Generative Models, and Intelligent Communications." 2025. PMC.
41. "The geometry of efficient codes: How rate-distortion trade-offs distort the latent representations of generative models." PLOS Computational Biology, 2025.

### INR-Based Steganography
42. "Hiding Functions within Functions: Steganography by Implicit Neural Representations." TST, 2025.
43. "Unified Steganography via Implicit Neural Representation." arXiv:2505.01749, 2025.
44. "Noise-NeRF: Hide Information in Neural Radiance Field using Trainable Noise." 2024. https://arxiv.org/abs/2401.01216
