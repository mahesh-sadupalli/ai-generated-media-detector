# Research Report: Effects of Lossy Data Compression on AI-Generated Media Detection

**Date**: 2026-03-18
**Scope**: Comprehensive literature review of how lossy compression (JPEG, H.264/H.265, social media pipelines) degrades deepfake and AI-generated media detection, and methods to achieve compression-robust detection.
**Domains covered**: Data Compression Effects on AI Model Performance, AI-Generated Media Detection Robustness, Continual Learning (intersection)

---

## Executive Summary

Lossy compression is the single largest obstacle between laboratory-grade deepfake detection and real-world deployment. Detection models trained on raw or lightly compressed data routinely lose 15--50% absolute AUC when evaluated on content that has passed through social media upload/download pipelines (JPEG quality 60--85, H.264/H.265 re-encoding at variable bitrates, resolution downscaling). The core mechanism is that compression destroys exactly the high-frequency forensic traces -- GAN upsampling artifacts, diffusion model spectral fingerprints, noise residual patterns -- that detectors rely on, while simultaneously introducing block-boundary and quantization artifacts that mimic forgery signatures and inflate false positive rates.

Three converging research directions have emerged to address this gap in 2023--2025. First, **compression-aware training** -- including differentiable JPEG layers, random quality-factor augmentation, and social-network compression emulation -- has become the standard hardening technique. Second, **frequency-domain architectures** that adaptively recover or exploit surviving mid-frequency forensic cues (HiFE, WaveDIF, frequency masking) have shown strong gains on heavily compressed data. Third, **semantic-level detectors** based on pre-trained vision-language models (CLIP) have demonstrated surprising compression resilience because they rely on high-level features rather than fragile pixel-level traces. The arrival of JPEG AI (neural compression, ISO standard published February 2025) introduces a new wildcard: its artifacts resemble those of generative models, threatening to confuse existing forensic tools entirely.

Despite progress, a substantial gap remains. Even the best current methods top out around AUC 0.65--0.70 on heavily compressed in-the-wild video deepfakes (Deepfake-Eval-2024), and no single method simultaneously achieves robustness to compression, generalization across generators, and resilience to adversarial post-processing. This report synthesizes the state of the art across 30+ papers from 2023--2025.

---

## 1. Background: How Compression Destroys Forensic Traces

### 1.1 JPEG Compression Effects on Image Forensics

JPEG compression operates via 8x8 block-wise Discrete Cosine Transform (DCT), quantization of frequency coefficients, and entropy coding. The quantization step is controlled by a Quality Factor (QF), where lower QF means more aggressive quantization. Key effects on forensic traces:

- **High-frequency artifact destruction**: GAN upsampling artifacts, checkerboard patterns, and noise-residual fingerprints reside in high-frequency DCT coefficients that are quantized to zero at QF < 80.
- **Block boundary artifacts**: JPEG introduces its own 8x8 blocking artifacts that can be confused with face-swap boundary artifacts, increasing false positive rates.
- **Double compression**: Content compressed once during generation and again during social media upload creates "ghost artifacts" that further confound analysis.

At JPEG QF 75 (typical web compression), simple detection signatures still partially survive. At QF < 50, most pixel-level forensic traces are effectively laundered.

### 1.2 Video Compression (H.264/H.265) Effects

Modern video codecs introduce additional complications beyond JPEG:

- **Variable-strength macroblock quantization**: H.264 and H.265 use different quantization parameters for different macroblocks within a single frame, creating local forensic-trace variations that detectors misinterpret as anomalous regions.
- **Motion compensation and inter-frame prediction**: These operations blend information across frames, diluting per-frame forensic signals.
- **Quantization parameter (QP)**: FaceForensics++ uses QP 23 (c23, "high quality") and QP 40 (c40, "low quality") as standard test conditions. Most detectors see 10--25% AUC drops from c23 to c40.

### 1.3 Social Media Compression Pipelines

Social media platforms apply aggressive, proprietary, and frequently updated compression:

| Platform | Image Compression | Video Compression | Additional Processing |
|----------|------------------|-------------------|----------------------|
| WhatsApp | JPEG QF ~60-70, resolution downscaling | H.264, aggressive bitrate | Heavy downscaling, metadata stripping |
| Instagram | JPEG QF ~70-85, resolution capping (1080px) | H.264/H.265, variable bitrate | Color space conversion, sharpening |
| Twitter/X | JPEG compression, PNG-to-JPEG conversion | H.264 re-encoding | Resolution limits, aspect ratio cropping |
| TikTok | -- | H.264/H.265 transcoding, variable bitrate | Heavy re-encoding, watermark overlay |
| YouTube | -- | VP9/AV1 re-encoding | Multi-resolution ladder encoding |
| Facebook | Adaptive quality JPEG | H.264/H.265, adaptive bitrate | Resolution-dependent quality |

These pipelines result in **15--20% average precision decrease** for detection models compared to uncompressed evaluation. The compression parameters change over time as platforms update their infrastructure.

---

## 2. Detection Degradation Under Compression: Empirical Evidence

### 2.1 FaceForensics++ Compression Benchmarks

The FaceForensics++ dataset (Rossler et al., 2019) provides the canonical compression-stratified benchmark with three quality levels:

- **c0 (raw)**: No compression. Detectors achieve >95% AUC on most manipulation types.
- **c23 (HQ)**: H.264 QP 23. Most detectors maintain >90% AUC. Best methods: MCW achieves AUC 0.929.
- **c40 (LQ)**: H.264 QP 40. Severe degradation. Best reported: GFADE achieves AUC 76.29%. Most methods fall below 75%.

The c23-to-c40 drop is typically 10--25% absolute AUC, demonstrating that even moderate video compression substantially degrades detection.

### 2.2 In-the-Wild Performance Collapse

**Deepfake-Eval-2024** (Chandra et al., 2025) provides the most sobering real-world assessment:

- Collected 45 hours of video, 56.5 hours of audio, and 1,975 images from 88 websites in 52 languages
- **AUC drops**: 50% for video detectors, 48% for audio detectors, 45% for image detectors compared to controlled benchmarks
- Even fine-tuned models fell short of professional forensic analyst accuracy
- Content reflects 2024's latest manipulation technologies including diffusion-based generators

**Reference**: Chandra, N.A., Murtfeldt, R., Qiu, L., et al. "Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024." arXiv:2503.02857, 2025. https://arxiv.org/abs/2503.02857

### 2.3 GenImage and AI-Generated Image Detection Benchmarks

The GenImage benchmark (Zhu et al., NeurIPS 2024) explicitly tests robustness under:
- Downsampling to resolutions 112 and 64
- JPEG compression at QF 65 and QF 30
- Gaussian blurring

A notable finding from recent analysis: a simple 3-layer CNN can maintain 92% AUC after resizing at QF 75. However, at QF 30, detection signatures are largely destroyed for pixel-level methods.

**AI-GenBench** (2025) and **AIGIBench** (Li et al., ICCV 2025) further formalize robustness testing through four core tasks: multi-source generalization, robustness to image degradation, sensitivity to data augmentation, and impact of test-time pre-processing.

**References**:
- Zhu, M. et al. "GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image." NeurIPS 2024. https://openreview.net/pdf?id=GF84C0z45H
- Li et al. "Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios." ICCV 2025.
- AI-GenBench: arXiv:2504.20865. https://arxiv.org/abs/2504.20865

---

## 3. Compression-Robust Detection Methods and Architectures

### 3.1 CLIP-Based Semantic Detectors

The most significant finding from 2024--2025 is that **pre-trained vision-language models (CLIP) provide inherent compression robustness** because they encode high-level semantic features rather than fragile pixel-level artifacts.

**Cozzolino et al., "Raising the Bar of AI-generated Image Detection with CLIP," CVPR Workshop 2024.**
- A single linear layer on CLIP features trained on a handful of images from one generator
- +6% AUC on out-of-distribution generators vs. prior SoTA
- **+13% AUC on impaired/laundered data** (compression, resizing, cropping)
- Detectors with augmentation are "basically insensitive to compression, no matter JPEG or WebP, and resizing"
- Authors: Davide Cozzolino, Giovanni Poggi, Riccardo Corvi, Matthias Niessner, Luisa Verdoliva

**Reference**: https://arxiv.org/abs/2312.00195

**Guillaro et al., "A Bias-Free Training Paradigm for More General AI-generated Image Detection," CVPR 2025.**
- B-Free generates fake training images from self-conditioned reconstructions of real images using diffusion model conditioning
- Eliminates content/format/resolution biases that cause spurious correlations
- Significant improvements in both generalization and robustness over SoTA
- Authors: Guillaro et al. (GRIP, University of Naples)

**Reference**: https://arxiv.org/abs/2412.17671

### 3.2 D3: Discrepancy-Based Scaling (CVPR 2025)

**Yang et al., "D3: Scaling Up Deepfake Detection by Learning from Discrepancy," CVPR 2025.**
- Dual-branch framework using distorted image features as discrepancy signals
- +5.3% Out-of-Domain accuracy over SoTA while maintaining In-Domain performance
- Explicitly tested on JPEG compression QF 30--100 and Gaussian blur 0--2
- "Exhibits the best robustness with a large margin compared to other methods"
- Code available at https://github.com/BigAandSmallq/D3

**Reference**: https://arxiv.org/abs/2404.04584

### 3.3 PLADA: Handling Block Effects from OSN Compression (2025)

**Li et al., "Pay Less Attention to Deceptive Artifacts: Robust Detection of Compressed Deepfakes on Online Social Networks," arXiv 2025.**
- Two modules: Block Effect Eraser (B2E) uses dual-stage attention to suppress JPEG block artifacts; Open Data Aggregation (ODA) handles both paired and unpaired compressed data
- First method to explicitly model "block effects" as a confounding factor
- Evaluated on **26 datasets**; outperforms SoTA on OSN-compressed deepfakes
- Authors: Manyi Li, Renshuai Tao, Yufan Liu, Chuangchuang Tan, Haotong Qin, Bing Li, Yunchao Wei, Yao Zhao
- Code: https://github.com/ManyiLee/PLADA

**Reference**: https://arxiv.org/abs/2506.20548

### 3.4 Social Network Compression Emulation (2025)

**Montibeller et al., "Bridging the Gap: A Framework for Real-World Video Deepfake Detection via Social Network Compression Emulation," ACM Deepfake Forensics Workshop 2025.**
- Estimates compression and resizing parameters from <50 uploaded videos per resolution
- Builds a local emulator reproducing platform-specific artifacts (Facebook, YouTube, BlueSky tested)
- Detectors fine-tuned on emulated videos match those trained on actual shared media
- Validated on both legacy 2021 data and 2025 uploads
- Planned expansion to TikTok, X, Telegram, WhatsApp, Slack
- Authors: Andrea Montibeller, Dasara Shullani, Daniele Baracchi, Alessandro Piva, Giulia Boato

**Reference**: https://arxiv.org/abs/2508.08765

### 3.5 Compression-Aware Hybrid Framework (IEEE Access 2025)

**"Compression-Aware Hybrid Framework for Deep Fake Detection in Low-Quality Video," IEEE Access 2025.**
- Integrates wavelet transforms and Conv3D-based spatiotemporal descriptors
- Lightweight ResNet-inspired classifier
- 92.45% accuracy, 0.45 GFLOPs on FFIW 10K dataset
- Four configurations tested (RGB/grayscale x with/without attention)
- Emphasizes interpretability and computational efficiency

**Reference**: IEEE Access, DOI: 10.1109/ACCESS.2025.3592358

---

## 4. Compression-Aware Training Techniques

### 4.1 Differentiable JPEG Layers

**Shin et al., "JPEG Inspired Deep Learning," ICLR 2025.**
- JPEG-DL framework prepends any DNN with a trainable JPEG compression layer
- Novel differentiable soft quantizer replaces the non-differentiable quantization step
- Joint optimization of quantization parameters and model weights
- Only adds 128 trainable parameters
- Up to 20.9% accuracy improvement on fine-grained classification; also improves adversarial robustness
- Enables the network to learn which frequency components matter for the task

**Reference**: https://arxiv.org/abs/2410.07081

Differentiable JPEG modules are now commonly integrated into deepfake detection and adversarial watermarking pipelines to simulate compression during training, allowing gradients to flow through the compression step.

### 4.2 Data Augmentation with Compression Simulation

The standard compression augmentation protocol for deepfake detection training (widely adopted 2023--2025):

```
During training, with probability p=0.5:
  - Apply JPEG compression with random QF in [40, 100]
  - Optionally apply Gaussian blur (sigma 0--2)
  - Optionally apply Gaussian noise
  - Optionally apply random resize (0.5x--1.5x)
```

**Empirical impact of augmentation** (aggregated from multiple studies):
- Celeb-DF-V1: AUC improved from 80.15% to 88.40% (+8.25%)
- Celeb-DF-V2: AUC improved from 72.96% to 79.00% (+6.04%)
- DFDCP: AUC improved from 73.40% to 75.88% (+2.48%)

### 4.3 Latent Space Augmentation

**Yan et al., "Transcending Forgery Specificity with Latent Space Augmentation for Generalizable Deepfake Detection," CVPR 2024.**
- LSDA (Latent Space Data Augmentation) constructs and simulates variations within and across forgery features in latent space
- Enriches domain-specific features and facilitates smoother transitions between forgery types
- More effective than pixel-level augmentation for cross-domain generalization

**Reference**: https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Transcending_Forgery_Specificity_with_Latent_Space_Augmentation_for_Generalizable_Deepfake_CVPR_2024_paper.pdf

### 4.4 Self-Blended Images (SBI) and Neighboring Pixel Relationships (NPR)

- **SBI** (Shiohara & Yamasaki, CVPR 2022): Generates pseudo-fakes from pristine images via self-blending, creating general visual artifacts that implicitly represent compression-robust forgery cues.
- **NPR** (Tan et al., CVPR 2024): "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection." NPR heatmaps capture artifacts in local pixel neighborhoods that are more robust to compression than global spectral features.

---

## 5. Frequency-Domain Methods That Survive Compression

### 5.1 HiFE: High-Frequency Enhancement Network

**Gao et al., "DeepFake detection based on high-frequency enhancement network for highly compressed content," Expert Systems with Applications, 2024.**
- Three-branch architecture:
  - Basic branch: RGB domain features
  - Local HiFE (LHiFE): Block-wise DCT with channel attention -- recovers local high-frequency details
  - Global HiFE (GHiFE): Multi-level DWT with cascade-residual fusion -- recovers global high-frequency patterns
- Two-Stage Cross-Fusion strategy combines complementary domain information
- First method using adaptive local AND global high-frequency enhancement without uncompressed supervision
- Code: https://github.com/Gina-YUE/HiFE

**Reference**: https://doi.org/10.1016/j.eswa.2024.123732

### 5.2 WaveDIF: Wavelet Sub-Band Deepfake Identification

**Dutta et al., "WaveDIF: Wavelet sub-band based Deepfake Identification in Frequency Domain," CVPR Workshop 2025.**
- Decomposes video frames into LL, LH, HL, HH wavelet sub-bands using Haar filter
- Computes energy values per sub-band as discriminative features
- Applies DFT to filter high-frequency noise before wavelet decomposition
- Lightweight, strict frequency-domain approach
- Authors: Anurag Dutta, Arnab Kumar Das, Ruchira Naskar, Rajat Subhra Chakraborty

**Reference**: https://openaccess.thecvf.com/content/CVPR2025W/CVEU/papers/Dutta_WaveDIF_Wavelet_sub-band_based_Deepfake_Identification_in_Frequency_Domain_CVPRW_2025_paper.pdf

### 5.3 Frequency Masking for Universal Detection

**Doloriel & Cheung, "Frequency Masking for Universal Deepfake Detection," ICASSP 2024.**
- Random frequency-domain masking during training forces model to learn from diverse spectral signatures
- Frequency masking more effective than spatial masking for cross-generator generalization
- Extended in 2024 to "Towards Sustainable Universal Deepfake Detection with Frequency-Domain Masking" (arXiv:2512.08042)
  - Maintains performance under significant model pruning
  - Scalable and resource-conscious

**References**:
- https://arxiv.org/abs/2401.06506
- https://arxiv.org/abs/2512.08042

### 5.4 Spatial-Frequency Fusion Approaches

Multiple 2024--2025 papers propose dual-stream architectures fusing spatial and frequency features:

- **SFIAD** (2025): Spatial-frequency feature integration with dynamic margin optimization. Uses frequency domain information to enhance robustness since compression-induced anomalies manifest more clearly in frequency domain.
- **Adaptive Fusion** (Wang et al., 2024): Deepfake detection based on adaptive fusion of spatial-frequency features. Published in Int. J. Intelligent Systems.
- **Spatial-Frequency Interaction** (Zhai et al., 2024): Learning spatial-frequency interaction for generalizable deepfake detection. IET Image Processing.

### 5.5 Key Insight: Mid-Frequency Bands Are the Sweet Spot

A consistent finding across frequency-domain methods: **low frequencies survive compression but lack discriminative power for forgery detection; very high frequencies are destroyed by compression; mid-frequency bands (roughly DCT coefficients 8--32) carry the best balance of forensic signal and compression resilience.** Methods that adaptively focus on these mid-frequency bands consistently outperform those that rely on either purely spatial or purely high-frequency features.

---

## 6. Neural Compression (Learned Codecs) vs. Traditional Codecs

### 6.1 JPEG AI: A New Forensic Challenge

**Cannas et al., "Is JPEG AI going to change image forensics?" ICCV Workshop 2025.**
- JPEG AI was published as the first international learned image coding standard in February 2025
- Neural compression artifacts "closely resemble those generated by image synthesis techniques and image splicing pipelines"
- Leading forensic detectors show reduced performance when analyzing JPEG AI-compressed content
- Double JPEG AI compression has counter-forensic effects
- **Critical implication**: Neural compression could inadvertently launder AI-generated content by overwriting generative model traces with compression artifacts that look similar

**Reference**: https://arxiv.org/abs/2412.03261

### 6.2 JPEG AI Artifact Detection

**"JPEG AI Image Compression Visual Artifacts: Detection Methods and Dataset," arXiv 2024.**
- Identifies three distinct artifact types from neural compression: texture/boundary degradation, color change, and text corruption
- These artifacts are fundamentally different from traditional JPEG blocking
- Traditional quality metrics (PSNR) are inadequate for assessing neural compression artifacts
- Proposes separate detection modules for each artifact type

**Reference**: https://arxiv.org/html/2411.06810v1

### 6.3 Three Forensic Cues for JPEG AI Images

**"Three Forensic Cues for JPEG AI Images," arXiv 2025.**
- Identifies specific forensic signatures unique to JPEG AI compression
- Provides methods to distinguish JPEG AI-compressed images from both traditionally compressed and AI-generated images

**Reference**: https://arxiv.org/abs/2504.03191

### 6.4 Implications

The emergence of neural compression as a deployment codec creates a two-sided problem:
1. **For detectors**: Existing methods trained on JPEG/H.264 artifacts will not generalize to content compressed with learned codecs
2. **For forensics**: Neural compression can serve as an effective "laundering" operation, destroying generative model fingerprints while introducing artifacts that look similar to generation artifacts

This is an **urgent and underexplored** research direction. Very few detection methods have been evaluated against neural compression, and current benchmarks do not include JPEG AI or other learned codec conditions.

---

## 7. Continual Learning at the Intersection

### 7.1 Adapting Detectors to New Generators AND New Compression

The continual learning problem in deepfake detection is doubly compounded by compression: detectors must simultaneously adapt to (a) new generative architectures and (b) evolving platform compression pipelines, without forgetting how to handle older generators or compression settings.

**Zhang et al., "Generalization-Preserved Learning: Closing the Backdoor to Catastrophic Forgetting in Continual Deepfake Detection," ICCV 2025.**
- Hyperbolic Visual Alignment: learnable watermarks align incremental data with base set in hyperbolic space
- Generalized Gradient Projection: prevents parameter updates conflicting with generalization constraints
- Four-phase incremental training evaluated on accuracy (ACC), average accuracy (AA), and average forgetting (AF)
- Key insight: stability and plasticity can coexist through the model's inherent generalization capability

**Reference**: https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Generalization-Preserved_Learning_Closing_the_Backdoor_to_Catastrophic_Forgetting_in_Continual_ICCV_2025_paper.pdf

**"Revisiting Deepfake Detection: Chronological Continual Learning and the Limits of Generalization," arXiv 2025.**
- Proposes chronological ordering of training phases to better reflect real-world deployment
- Reveals that even with continual learning, generalization has inherent limits

**Reference**: https://arxiv.org/abs/2509.07993

### 7.2 Continual Learning for Audio Deepfake Detection Under Compression

**"Region-Based Optimization in Continual Learning for Audio Deepfake Detection," AAAI 2025.**
- Addresses catastrophic forgetting when adapting to new TTS/voice-cloning methods
- Audio compression (MP3, AAC, Opus) causes analogous degradation to image JPEG

**Reference**: https://ojs.aaai.org/index.php/AAAI/article/view/34535

---

## 8. State-of-the-Art Methods Summary

| Method | Year/Venue | Key Contribution | Compression Robustness | Limitations |
|--------|-----------|------------------|----------------------|-------------|
| CLIP-based detector (Cozzolino) | CVPRW 2024 | Linear probe on CLIP features | +13% AUC on laundered data; insensitive to JPEG/WebP | Requires CLIP backbone; limited localization |
| B-Free (Guillaro) | CVPR 2025 | Bias-free training via diffusion self-conditioning | Strong generalization + robustness | Diffusion-model specific training |
| D3 (Yang) | CVPR 2025 | Discrepancy signal from distorted branch | Best robustness across JPEG QF 30-100 | Two-branch overhead |
| PLADA (Li) | arXiv 2025 | Block Effect Eraser + Open Data Aggregation | Designed for OSN compression | Requires paired data training |
| HiFE (Gao) | ESWA 2024 | 3-branch: RGB + Block-DCT + Multi-level DWT | Strong on highly compressed content | Computational cost of 3 branches |
| WaveDIF (Dutta) | CVPRW 2025 | Wavelet sub-band energy features | Lightweight frequency-domain approach | Video-only; limited generator coverage |
| Freq. Masking (Doloriel) | ICASSP 2024 | Random frequency masking during training | Good cross-generator generalization | Training strategy only, not architecture |
| Compression Emulation (Montibeller) | ACM DFW 2025 | Platform-specific compression emulation | Fine-tuning matches real uploads | Per-platform calibration needed |
| LSDA (Yan) | CVPR 2024 | Latent space data augmentation | Cross-domain generalization | Augmentation only, not compression-specific |
| GPL (Zhang) | ICCV 2025 | Continual learning with hyperbolic alignment | Handles evolving generators | Compression robustness not primary focus |
| JPEG-DL (Shin) | ICLR 2025 | Differentiable JPEG layer for any DNN | Enables compression-aware end-to-end training | General framework, not detection-specific |

---

## 9. Open Problems and Research Gaps

### 9.1 The Video Compression Gap
Most compression-robustness research focuses on JPEG image compression. Video-specific compression effects (inter-frame prediction, motion compensation, B-frame/P-frame dynamics) are significantly understudied. The gap between image and video detection robustness remains large (AUC 0.65--0.70 for best video methods in the wild vs. >0.90 for image methods on benchmarks).

### 9.2 Neural Compression Forensics
JPEG AI and other learned codecs present a fundamentally new challenge. Their artifacts overlap with generative model artifacts, and very few detection methods have been evaluated under neural compression. This is an urgent gap as JPEG AI moves toward deployment.

### 9.3 Dynamic Platform Pipelines
Social media compression parameters change frequently and without notice. The compression emulation approach (Montibeller et al.) requires per-platform re-calibration. There is no adaptive method that can handle unknown or changing compression.

### 9.4 Joint Compression + Adversarial Robustness
Compression and adversarial attacks interact in complex ways. An adversarial perturbation designed to fool a detector may be partially neutralized by subsequent compression, or compression may amplify adversarial effects. This interaction is poorly understood.

### 9.5 Audio-Visual Consistency Under Compression
For interview-integrity and video-call deepfake detection, audio and video are compressed independently and with different codecs (Opus/AAC for audio, H.264/VP9 for video). How audio-visual synchronization cues survive this dual-compression pipeline is underexplored.

### 9.6 Continual Learning + Compression Robustness
No existing continual learning method for deepfake detection explicitly addresses the evolving compression landscape. Detectors must adapt to both new generators AND new compression pipelines, creating a compound distribution shift.

### 9.7 Calibration Under Compression
Detection confidence scores become poorly calibrated after compression -- a model reporting 95% confidence on uncompressed data may report 50% on the same image after JPEG QF 40, without the image actually being harder to classify. Calibration-aware methods under compression are absent from the literature.

---

## 10. Recommendations for the AI-Generated Media Detector Project

Based on this literature review, the following recommendations are directly relevant to the project at `/Users/mahesh/ai-generated-media-detector/`:

### Immediate (Phase 1-2)

1. **Add JPEG compression augmentation to training pipeline**: Apply random JPEG QF [40, 100] with p=0.5 during all training. This is the single highest-impact change for real-world robustness.

2. **Evaluate frequency-domain features alongside spatial features**: The current hand-crafted detector ensemble (smoothing + texture + mode collapse) operates primarily in spatial domain. Adding DCT and DWT-based features (inspired by HiFE) would improve compression resilience.

3. **Benchmark against compression levels**: Test the current detector ensemble at JPEG QF 30, 50, 65, 75, 85, 100 and H.264 QP 23, 30, 40 to establish a compression degradation profile.

4. **Consider CLIP-based detection as a compression-robust baseline**: A simple linear probe on CLIP features can serve as a robust complement to the existing hand-crafted features, especially for the Phase 2 deep learning model.

### Medium-term (Phase 3)

5. **Implement social media compression emulation**: For the interview integrity platform, build a compression simulation pipeline that mimics common video conferencing codecs (VP8/VP9 for WebRTC, H.264 for Zoom) and social media re-encoding.

6. **Frequency masking augmentation**: Incorporate random frequency-domain masking during training per Doloriel & Cheung (ICASSP 2024) to force the model to learn from diverse spectral signatures.

7. **Test lip-sync and eye-gaze modules under compression**: Audio-visual synchronization cues (SyncNet, AV-HuBERT) must be validated under realistic video compression conditions.

### Long-term (Phase 4+)

8. **Plan for JPEG AI / neural compression**: Include JPEG AI-compressed images in test sets as learned codecs move toward deployment.

9. **Implement continual learning**: Use GPL-style (ICCV 2025) or replay-based continual learning to adapt to new generators and compression pipelines without forgetting.

10. **Consider the compression estimator module**: The existing `src/utils/compression_estimator.py` (currently in development) should output estimated compression type and quality factor, which can be used to select compression-appropriate detection strategies at inference time.

---

## References

1. Cozzolino, D., Poggi, G., Corvi, R., Niessner, M., Verdoliva, L. "Raising the Bar of AI-generated Image Detection with CLIP." CVPR Workshops, 2024. https://arxiv.org/abs/2312.00195

2. Guillaro, F., et al. "A Bias-Free Training Paradigm for More General AI-generated Image Detection." CVPR, 2025. https://arxiv.org/abs/2412.17671

3. Yang, et al. "D3: Scaling Up Deepfake Detection by Learning from Discrepancy." CVPR, 2025. https://arxiv.org/abs/2404.04584

4. Li, M., Tao, R., Liu, Y., Tan, C., Qin, H., Li, B., Wei, Y., Zhao, Y. "Pay Less Attention to Deceptive Artifacts: Robust Detection of Compressed Deepfakes on Online Social Networks." arXiv:2506.20548, 2025. https://arxiv.org/abs/2506.20548

5. Montibeller, A., Shullani, D., Baracchi, D., Piva, A., Boato, G. "Bridging the Gap: A Framework for Real-World Video Deepfake Detection via Social Network Compression Emulation." ACM Deepfake Forensics Workshop, 2025. https://arxiv.org/abs/2508.08765

6. Gao, J., et al. "DeepFake detection based on high-frequency enhancement network for highly compressed content." Expert Systems with Applications, 2024. https://doi.org/10.1016/j.eswa.2024.123732

7. Dutta, A., Das, A.K., Naskar, R., Chakraborty, R.S. "WaveDIF: Wavelet sub-band based Deepfake Identification in Frequency Domain." CVPR Workshops, 2025.

8. Doloriel, C.T., Cheung, A.S. "Frequency Masking for Universal Deepfake Detection." ICASSP, 2024. https://arxiv.org/abs/2401.06506

9. Doloriel, C.T., Cheung, A.S. "Towards Sustainable Universal Deepfake Detection with Frequency-Domain Masking." arXiv:2512.08042, 2024. https://arxiv.org/abs/2512.08042

10. Yan, et al. "Transcending Forgery Specificity with Latent Space Augmentation for Generalizable Deepfake Detection." CVPR, 2024.

11. Zhang, et al. "Generalization-Preserved Learning: Closing the Backdoor to Catastrophic Forgetting in Continual Deepfake Detection." ICCV, 2025.

12. "Revisiting Deepfake Detection: Chronological Continual Learning and the Limits of Generalization." arXiv:2509.07993, 2025. https://arxiv.org/abs/2509.07993

13. Cannas, et al. "Is JPEG AI going to change image forensics?" ICCV Workshops, 2025. https://arxiv.org/abs/2412.03261

14. "Three Forensic Cues for JPEG AI Images." arXiv:2504.03191, 2025. https://arxiv.org/abs/2504.03191

15. "JPEG AI Image Compression Visual Artifacts: Detection Methods and Dataset." arXiv:2411.06810, 2024. https://arxiv.org/html/2411.06810v1

16. Shin, et al. "JPEG Inspired Deep Learning." ICLR, 2025. https://arxiv.org/abs/2410.07081

17. Chandra, N.A., et al. "Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024." arXiv:2503.02857, 2025. https://arxiv.org/abs/2503.02857

18. Rossler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., Niessner, M. "FaceForensics++: Learning to Detect Manipulated Facial Images." ICCV, 2019.

19. Zhu, M., et al. "GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image." NeurIPS, 2024.

20. "Compression-Aware Hybrid Framework for Deep Fake Detection in Low-Quality Video." IEEE Access, 2025. DOI: 10.1109/ACCESS.2025.3592358

21. "Region-Based Optimization in Continual Learning for Audio Deepfake Detection." AAAI, 2025.

22. Li et al. "Continual Deepfake Detection Based on Multi-Perspective Sample Selection Mechanism." IEEE Trans. on Multimedia, 2025.

23. Tan, C., et al. "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection." CVPR, 2024.

24. Shiohara, K., Yamasaki, T. "Detecting Deepfakes with Self-Blended Images." CVPR, 2022.

25. Wang, Z., et al. "DIRE for Diffusion-Generated Image Detection." ICCV, 2023.

---

**Limitations of this review**: This report is based on papers available through March 2026. The social media compression parameters listed are approximate and change frequently. Performance comparisons across papers should be interpreted cautiously as different papers use different evaluation protocols, datasets, and preprocessing. Some papers cited are preprints and have not undergone peer review.
