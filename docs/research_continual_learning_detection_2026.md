# Research Report: Continual Learning, Catastrophic Forgetting, and Robustness in AI-Generated Media Detection

**Date**: 2026-03-18
**Scope**: Comprehensive survey of continual learning for deepfake/AI-media detection, compression robustness, domain generalization, zero-shot detection, and implicit neural representations in media forensics
**Domains covered**: All three core domains (compression effects, catastrophic forgetting/CL, detection robustness) plus INR/neural compression forensics

---

## Executive Summary

The rapid evolution of generative models -- from early GANs (ProGAN, StyleGAN) through diffusion models (Stable Diffusion, DALL-E, Midjourney) to video generators (Sora, Runway) -- has created a fundamental challenge for AI-generated media detectors: **each new generation of generative technology can render existing detectors ineffective**. This report synthesizes the state-of-the-art on how continual learning (CL) methods are being applied to maintain detector efficacy over time, how compression pipelines (especially social media platforms) degrade forensic cues, and emerging approaches including CLIP-based zero-shot detection, parameter-efficient adaptation via LoRA, and the nascent use of implicit neural representations (INRs) in media forensics.

**Key finding**: Recent work (Fontana et al., 2025) proposes the **Non-Universal Deepfake Distribution Hypothesis** -- each generative model leaves a unique, non-transferable signature. Empirically, forward transfer AUC drops to near-random (~0.5) within 3 unseen generators, meaning CL alone cannot solve generalization; it must be paired with representation learning that captures generator-agnostic artifacts. The most promising directions combine foundation model features (CLIP ViT), frequency-domain analysis, and CL-specific replay/regularization strategies.

The intersection of compression robustness and continual learning remains **critically underexplored**. Social media compression launders the very forensic traces that detectors learn incrementally, yet almost no CL-for-detection papers evaluate under realistic compression conditions. This represents a major open research gap directly relevant to production deployment.

---

## 1. Background and Context

### 1.1 The Generative Model Evolution Problem

AI-generated media detectors face a unique variant of distribution shift: the "adversary" (generative models) evolves discontinuously. The transition from GAN-based generation to diffusion-based generation fundamentally changed the artifact landscape:

- **GAN artifacts**: Spectral anomalies from upsampling (checkerboard patterns in FFT), mode collapse signatures, smooth texture regions from pixel-loss optimization, cross-shaped spectral artifacts from convolutional upsampling
- **Diffusion artifacts**: Reconstruction patterns, denoising traces, mid-frequency reconstruction errors, subtler spectral fingerprints that resist traditional frequency-domain detection
- **Video generator artifacts**: Temporal inconsistencies, inter-frame coherence failures, motion compensation artifacts layered on top of per-frame generation artifacts

A detector trained on FaceSwap-GAN (2018) may achieve >95% accuracy on that domain but drop to near-random on Stable Diffusion outputs (2023+). This is not merely domain shift -- it represents a categorical change in the artifact distribution.

### 1.2 Why Standard Continual Learning Is Insufficient

Standard CL benchmarks (Split-CIFAR, Split-ImageNet) involve distributional shifts within a shared visual manifold. Deepfake detection CL presents unique challenges:
- **Binary classification with shifting decision boundaries**: Real vs. fake, but the "fake" class encompasses radically different distributions
- **Asymmetric forgetting**: Forgetting how to detect old generators is more dangerous than forgetting specific visual classes
- **Temporal ordering matters**: Generators arrive chronologically, and later generators may subsume or diverge from earlier ones
- **Compression confounds**: Real-world deployment always involves lossy compression that destroys the subtle cues CL methods learn to preserve

---

## 2. Continual Learning Methods Applied to Deepfake Detection

### 2.1 Benchmarks and Evaluation Frameworks

#### CDDB: Continual Deepfake Detection Benchmark (Li et al., WACV 2023)
- **Paper**: "A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials"
- **Contribution**: First dedicated benchmark for CL in deepfake detection
- **Design**: Evaluates detection over easy, hard, and long sequences of deepfake tasks from both known and unknown generative models
- **Methods evaluated**: Multiple approaches adapted from multiclass incremental learning to the binary continual deepfake detection problem
- **Key finding**: Replay-based methods consistently outperform regularization-only approaches in the deepfake detection domain
- **Link**: https://arxiv.org/abs/2205.05467 | https://github.com/Coral79/CDDB

#### Chronological CL Framework (Fontana et al., 2025)
- **Paper**: "Revisiting Deepfake Detection: Chronological Continual Learning and the Limits of Generalization"
- **Contribution**: Reframes deepfake detection as a CL problem with realistic chronological ordering spanning 7 years (2018-2024)
- **Datasets used chronologically**: DeepfakeTIMIT (2018), WildDeepfake (2020), DFFD (2020), FakeAVCelebv2 (2021), COCOFake (2023), CIFAKE (2024)
- **CL methods compared**: Naive, Replay, EWC, CLS-ER, ESMER, DER++, Replay+EWC, CLS-ER+EWC
- **Novel metrics**: Continual AUC (C-AUC) and Forward Transfer AUC (FWT-AUC)
- **Key quantitative results**:
  - Best C-AUC: ViT-Tiny with CLS-ER achieved **0.915 mean C-AUC**
  - FWT-AUC: **Consistently near-random (~0.49-0.57)** across ALL methods
  - Empirical decay factor of 0.54 -- approaches random guessing within 3 unseen generators
  - CL methods achieve **~155x speedup** vs. full retraining
- **Central hypothesis**: Non-Universal Deepfake Distribution Hypothesis -- each generator leaves a unique, non-transferable signature
- **Link**: https://arxiv.org/abs/2509.07993

### 2.2 Regularization-Based Approaches

#### EWC in Deepfake Detection
Elastic Weight Consolidation (Kirkpatrick et al., 2017) has been adapted for deepfake detection in multiple works. The core idea -- penalizing changes to weights important for previous tasks via the Fisher information matrix -- is appealing for detection because forensic feature extractors develop specialized neurons for different artifact types.

**Empirical finding from Fontana et al.**: EWC alone provides modest improvements over naive sequential training for deepfake detection (C-AUC improvement of ~2-4%), but hybrid approaches (Replay+EWC, CLS-ER+EWC) consistently outperform pure regularization.

#### CADE: Continual Audio Defense Enhancer
- Integrates regularization with replay for audio deepfake detection
- Combines three loss functions: Knowledge Distillation Loss, Attention Distillation Loss, and Positive Sample Alignment Loss
- Demonstrates that hybrid approaches outperform pure regularization or pure replay in the audio domain

#### Region-Based Optimization (AAAI 2025)
- **Paper**: "Region-Based Optimization in Continual Learning for Audio Deepfake Detection"
- Proposes region-based weight modification strategies (RAWM, RWM) specifically designed for audio deepfake CL
- Published at AAAI 2025, indicating growing recognition of CL-for-detection at top venues

### 2.3 Replay-Based Approaches

#### Experience Replay for Deepfake Detection
Standard experience replay -- storing a buffer of past examples and mixing them with new task data -- is the **strongest baseline** in most deepfake CL benchmarks. Fontana et al. showed that simple replay already achieves competitive C-AUC (~0.89), with more sophisticated methods providing incremental gains.

#### GAN-CNN Ensemble with Generative Replay (2024)
- **Paper**: "GAN-CNN Ensemble: A Robust Deepfake Detection Model of Social Media Images Using Minimized Catastrophic Forgetting and Generative Replay Technique"
- Uses GANs to generate pseudo-rehearsal data from previous deepfake distributions
- Replays synthetic samples during training on new deepfake types
- Specifically evaluated on social media images
- Published in Procedia Computer Science, 2024

#### DARW: Domain-Aware Relative Weighting (Shen et al., 2025)
- **Paper**: "When Generative Replay Meets Evolving Deepfakes: Domain-Aware Relative Weighting for Incremental Face Forgery Detection"
- **First systematic investigation** of generative replay feasibility for face forgery detection
- **Key insight**: When replay generators resemble the new forgery model, generated "real" samples blur domain boundaries (domain-risky samples)
- **Method**: Uses diffusion models to synthesize past distributions; applies Relative Separation Loss for domain-risky samples and direct supervision for domain-safe samples
- **Results**:
  - Protocol 1 (Mixed-Era): **0.9574 average AUC**, performance drop rate of 0.0425 (lowest)
  - Protocol 2 (Benchmark-Aligned): **0.9429 average AUC** (vs. SUR-LID 0.9047, DFIL 0.8520)
- **Surpasses**: LwF, iCaRL, DER, CoReD, HDP, DFIL, SUR-LID
- **Link**: https://arxiv.org/abs/2511.18436

#### RAIS: Rehearsal with Auxiliary-Informed Sampling (2025)
- **Paper**: "Rehearsal with Auxiliary-Informed Sampling for Audio Deepfake Detection"
- Mitigates catastrophic forgetting by ensuring diverse, informative samples are retained
- Applied specifically to audio deepfake detection

### 2.4 Architecture-Based Approaches

#### GPL: Generalization-Preserved Learning (Zhang et al., ICCV 2025)
- **Paper**: "Generalization-Preserved Learning: Closing the Backdoor to Catastrophic Forgetting in Continual Deepfake Detection"
- **Key innovation**: Argues stability and plasticity can coexist through the model's inherent generalization, rather than treating them as conflicting
- **Components**:
  1. **Hyperbolic Visual Alignment**: Introduces learnable watermarks to align incremental data with the base set in hyperbolic space
  2. **Generalized Gradient Projection**: Prevents parameter updates that conflict with generalization constraints
- **Results**:
  - **92.14% accuracy** (outperforms replay-based SOTA by 2.15%)
  - Reduces forgetting by 2.66%
  - **18.38% improvement on unseen forgeries** using only 1% of baseline parameters
- **Venue**: ICCV 2025
- **Link**: https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Generalization-Preserved_Learning_Closing_the_Backdoor_to_Catastrophic_Forgetting_in_Continual_ICCV_2025_paper.pdf

#### SUR-LID: Aligned Feature Isolation (Cheng et al., CVPR 2025)
- **Paper**: "Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection"
- Achieves incrementing new tasks with isolated distributions and aligned decision boundaries
- **Stable Uniform Replay (SUR)** strategy enhances performance of aligned feature isolation
- Under forgery-type incremental protocol: average forgetting of **13.94%** (vs. DFIL's 40.72%, HDP's 29.30%)
- **Venue**: CVPR 2025
- **Link**: https://github.com/beautyremain/SUR-LID

#### DevFD: Developmental Face Forgery Detection (NeurIPS 2025)
- **Paper**: "DevFD: Developmental Face Forgery Detection by Learning Shared and Orthogonal LoRA Subspaces"
- **Architecture**: Developmental Mixture of Experts (MoE) using LoRA modules as individual experts
  - **Real-LoRA**: Learns and refines knowledge of real faces
  - **Fake-LoRAs**: Multiple modules capturing incremental information from different forgery types
- **Anti-forgetting mechanism**: Ensures learning direction of new Fake-LoRAs is orthogonal to established subspace via orthogonal gradient integration
- **Parameter efficiency**: Lightweight multi-head LoRA modules in every transformer block
- **Venue**: NeurIPS 2025
- **Link**: https://arxiv.org/abs/2509.19230

#### HDP: Historical Distribution Preserving (IJCV 2024)
- **Paper**: "Continual Face Forgery Detection via Historical Distribution Preserving"
- Preserves learned distributions from previous forgery types during incremental updates
- Published in International Journal of Computer Vision, 2024

### 2.5 Continuous Adaptation Frameworks

#### Tassone et al. (Computer Vision and Image Understanding, 2024)
- **Paper**: "Continuous fake media detection: adapting deepfake detectors to new generative techniques"
- Examines CL strategies on heterogeneous deepfakes (GANs, CG, unknown sources)
- **Key findings**:
  - Task similarity and ordering significantly influence model performance
  - Grouping tasks by similarity substantially improves results in longer sequences
  - Integration with CI/CD pipelines for maintaining detector effectiveness
- **Link**: https://arxiv.org/abs/2406.08171

#### CoReD: Continual Representation using Distillation (2021)
- Uses knowledge distillation to maintain representations from previous tasks
- Foundational method that later works (DFIL, SUR-LID, DARW) build upon and surpass

---

## 3. Zero-Shot and Few-Shot Detection of New Generative Models

### 3.1 CLIP-Based Universal Detection

#### UnivFD (Ojha et al., CVPR 2023)
- **Paper**: "Towards Universal Fake Image Detectors that Generalize Across Generative Models"
- **Core idea**: Use pre-trained CLIP:ViT feature space (not explicitly trained for real/fake) with nearest neighbor or linear probing
- **Trained on**: ProGAN real/fake images only
- **Results on unseen diffusion/autoregressive models**: +19.49 mAP and +23.39% accuracy (linear probing) over prior SOTA
- **Why it works**: CLIP features capture semantic and statistical regularities that generalize across generators
- **Link**: https://arxiv.org/abs/2302.10174 | https://github.com/WisconsinAIVision/UniversalFakeDetect

#### CLIP with Hyperspherical Metric Learning (Yermakov et al., 2025)
- **Paper**: "Unlocking the Hidden Potential of CLIP in Generalizable Deepfake Detection"
- Uses CLIP ViT-L/14 visual encoder with minimal modifications
- **LN-tuning**: Parameter-efficient fine-tuning preserving CLIP's pre-trained knowledge
- **Metric learning**: Uniformity and alignment losses on hyperspherical manifold
- **Results**: Celeb-DF-v2: 96.62%, DFD: 98.0%, DFDC: 87.15%, FFIW: 91.52% AUROC
- **Link**: https://arxiv.org/abs/2503.19683

#### Visual Language Models as Zero-Shot Deepfake Detectors (Pirogov, 2025)
- **Paper**: "Visual Language Models as Zero-Shot Deepfake Detectors"
- Evaluates VLMs with instruction tuning for zero-shot deepfake detection
- InternVL3-78B achieves nearly **90% accuracy** with a simple MLP classification head
- **Link**: https://arxiv.org/abs/2507.22469

#### Zero-Shot Visual Deepfake Detection (2025)
- **Paper**: "Zero-Shot Visual Deepfake Detection: Can AI Predict and Prevent Fake Content Before It's Created?"
- Explores detection without any training on fake examples
- **Link**: https://arxiv.org/abs/2509.18461

#### CausalCLIP (2025)
- **Paper**: "CausalCLIP: Causally-Informed Feature Disentanglement and Filtering for Generalizable Detection of Generated Images"
- Addresses spurious correlations in CLIP features that hurt generalization
- **Link**: https://arxiv.org/abs/2512.13285

### 3.2 Reprogramming and Adaptation Approaches

#### Standing on the Shoulders of Giants (2024)
- **Paper**: "Standing on the Shoulders of Giants: Reprogramming Visual-Language Model for General Deepfake Detection"
- Reprograms pre-trained CLIP for deepfake detection using few learnable parameters
- Leverages VLM generalization abilities with minimal fine-tuning
- **Link**: https://arxiv.org/abs/2409.02664

#### Facial Component Guided Adaptation (CVPR 2025)
- **Paper**: "Towards More General Video-based Deepfake Detection through Facial Component Guided Adaptation for Foundation Model"
- Side-network decoder extracts spatial and temporal cues from CLIP image encoder
- Specifically designed for video-based deepfake detection

#### Open-Set Deepfake Detection with Forgery Style Mixture (2024)
- **Paper**: "Open-Set Deepfake Detection: A Parameter-Efficient Adaptation Method with Forgery Style Mixture"
- Parameter-efficient adaptation for detecting previously unseen forgery types
- **Link**: https://arxiv.org/abs/2408.12791

#### Meta-Learned LoRA for Speech Deepfake Detection (2025)
- **Paper**: "Generalizable speech deepfake detection via meta-learned LoRA"
- Uses meta-learning to train LoRA adapters for rapid adaptation to new speech deepfake types
- **Link**: https://arxiv.org/abs/2502.10838

### 3.3 Self-Supervised and Augmentation-Based Generalization

#### SBI: Self-Blended Images (Shiohara & Yamasaki, CVPR 2022)
- **Paper**: "Detecting Deepfakes with Self-Blended Images"
- Generates training data by blending pseudo source/target images from single pristine images
- Learns generic blending artifacts rather than generator-specific ones
- Outperforms baselines on DFDC by 4.90% and DFDCP by 11.78% in cross-dataset evaluation
- Augmentations include ImageCompression, RGBShift, HueSaturation
- **Link**: https://arxiv.org/abs/2204.08376

#### FSBI: Frequency Enhanced Self-Blended Images (2024)
- Extension of SBI incorporating frequency-domain information
- **Link**: https://arxiv.org/abs/2406.08625

#### DiffusionFake (NeurIPS 2024)
- **Paper**: "DiffusionFake: Enhancing Generalization in Deepfake Detection via Guided Stable Diffusion"
- Reverses the generative process of face forgeries to enhance detection generalization
- Plug-and-play framework requiring no additional parameters during inference
- Significantly improves cross-domain generalization of various detector architectures
- **Link**: https://arxiv.org/abs/2410.04372

#### D3: Scaling Up Deepfake Detection (Yang et al., CVPR 2025)
- **Paper**: "D3: Scaling Up Deepfake Detection by Learning from Discrepancy"
- Addresses the multi-generator training problem where models sacrifice in-domain performance for out-of-domain generalization
- Parallel network branch with distorted image features as discrepancy signal
- **5.3% OOD accuracy improvement** while maintaining in-domain performance
- **Link**: https://arxiv.org/abs/2404.04584 | https://github.com/BigAandSmallq/D3

#### NPR: Neighboring Pixel Relationships (Tan et al., CVPR 2024)
- **Paper**: "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection"
- Exploits correlations between neighboring pixels caused by upsampling operators
- Works across both GAN and diffusion models (upsampling is universal)
- **Mean accuracy: 92.5%**, surpassing LGrad by 6.4% and Ojha (UnivFD) by 3.4%
- **Link**: https://arxiv.org/abs/2312.10461

---

## 4. Compression Robustness and Its Relationship to Generalization

### 4.1 Social Media Compression as a Forensic Laundering Mechanism

#### Bridging the Gap Framework (Montibeller et al., 2025)
- **Paper**: "Bridging the Gap: A Framework for Real-World Video Deepfake Detection via Social Network Compression Emulation"
- **Problem**: Social networks apply aggressive proprietary compression that launders forensic cues
- **Method**: Estimates platform-specific compression parameters (CRF, resolution, codec) from <50 uploaded videos, then emulates locally
- **Platforms studied**: Facebook (2021/2025), YouTube (2021/2025), BlueSky (2025)
- **Key finding**: Fine-tuning on emulated compressed videos achieves comparable detection performance to training on actual platform-shared videos
- **Minimum data**: 30 shared videos per resolution for reliable CRF estimation
- **Link**: https://arxiv.org/abs/2508.08765

#### PLADA (Li et al., 2025)
- **Paper**: "Pay Less Attention to Deceptive Artifacts: Robust Detection of Compressed Deepfakes on Online Social Networks"
- **Core insight**: JPEG block effects from social media compression resemble deepfake traces, misleading detectors
- **Method**: Block Effect Eraser (B2E) with dual-stage attention shift mechanism; Open Data Aggregation (ODA) for handling paired/unpaired data
- **Uses adversarial learning** with gradient reversal to convert compression detection signals into beneficial guidance
- **Results**: 76.7% mean accuracy on quality-aware GANs (vs. 72.6% prior SOTA); 73.4% on diffusion models
- **Link**: https://arxiv.org/abs/2506.20548

#### FIRE: Frequency-Guided Reconstruction Error (Chu et al., CVPR 2025)
- **Paper**: "FIRE: Robust Detection of Diffusion-Generated Images via Frequency-Guided Reconstruction Error"
- **Observation**: Diffusion models struggle to reconstruct mid-band frequency information in real images
- **Method**: Extracts frequency bands, computes reconstruction error differential
- **Results**: **100% AUC and ACC** across 11 diffusion models on DiffusionForensics dataset
- Outperforms other detectors on JPEG compression, cropping, blur, and noise perturbations
- **Link**: https://arxiv.org/abs/2412.07140 | https://github.com/Chuchad/FIRE

### 4.2 Compression-Aware Training

#### Differentiable JPEG Layers
- JPEG-DL framework prepends any DNN with a trainable JPEG compression layer using differentiable soft quantizer
- Pseudo-noise modeling approximates JPEG as additive Gaussian noise (differentiable)
- Enables end-to-end training of compression-robust detectors

#### Compression Augmentation as Standard Practice
Most recent detectors now include compression augmentation:
- SBI (CVPR 2022) uses ImageCompression augmentation
- Standard augmentation pipeline includes Gaussian blur, JPEG compression (quality 30-100), random crop, resize
- However, **simulated compression differs from real platform compression** (see Montibeller et al.)

### 4.3 Neural Compression and Forensics

#### JPEG AI Forensic Impact (Cannas et al., ICCV 2025 Workshop)
- **Paper**: "Is JPEG AI going to change image forensics?"
- **Critical finding**: JPEG AI (neural compression standard) introduces artifacts that **mislead forensic detectors**
- Pristine JPEG AI-compressed images are classified as manipulated content (TPR up to 96% false alarm)
- JPEG AI shifts score distributions toward "fake" classifications -- phenomenon absent with traditional JPEG
- Standard JPEG does not produce equivalent counter-forensic effects
- **Implication**: As neural compression standards are adopted, current detectors will produce massive false positives
- **Link**: https://arxiv.org/abs/2412.03261

#### AI Compression Trace Analysis (Bergmann et al., 2023)
- **Paper**: "Forensic analysis of AI-compression traces in spatial and frequency domain"
- Characterizes forensic compression traces in both frequency and spatial domains
- Both domains exhibit similar artifacts likely stemming from decoder upsampling operations
- Neural compression artifacts are detectable but generalization across codecs is challenging

#### Neural Compression Preserves Forgery Traces
- JPEG AI demonstrates superior ability to preserve forensic traces compared to HiFiC
- Copy-move forgery traces remain more detectable after neural compression than after traditional lossy compression
- **Counterintuitive**: Neural compression may be *better* for forensics than traditional JPEG in some scenarios

### 4.4 The Compression-Generalization Connection

**Critical open question**: The relationship between compression robustness and cross-generator generalization is bidirectional but poorly understood:

1. **Compression destroys generator-specific artifacts**: High-frequency GAN fingerprints, subtle spectral anomalies, and noise patterns are the first casualties of lossy compression. This forces models to rely on more robust (but potentially less discriminative) features.

2. **Compression-robust features may generalize better**: Features that survive JPEG compression at quality 70 are inherently more robust semantic/structural features, which may generalize better across generators than fragile spectral fingerprints.

3. **But compression can also introduce confounds**: JPEG block artifacts, WebP smoothing, and H.264 quantization artifacts can themselves be mistaken for generation artifacts (PLADA addresses this directly).

4. **Almost no CL papers evaluate under compression**: This is a critical gap. CL methods that achieve 0.91 C-AUC on clean data may perform far worse when each task's data has passed through different social media compression pipelines.

---

## 5. Implicit Neural Representations in Media Forensics

### 5.1 INR for Image Manipulation Detection

#### NRR: Neural Representation Reconstruction (Zhang et al., ECCV 2024)
- **Paper**: "Image Manipulation Detection with Implicit Neural Representation and Limited Supervision"
- **Core idea**: Uses INR as a pre-processing stage -- fits an INR to the image, then uses the reconstruction error map as an additional input channel
- **Neural Representation Reconstruction (NRR)**: Concatenates INR reconstruction error with RGB input before feeding to the backbone
- Uses NRR as a contrastive sample generator for selective pixel-level contrastive learning
- Focuses on high-confidence regions to mitigate uncertainty in weakly supervised settings
- **Venue**: ECCV 2024
- **Link**: https://link.springer.com/chapter/10.1007/978-3-031-73223-2_15

### 5.2 INR for Neural Video Compression

#### NeRV and Variants
- **NeRV** (Neural Representations for Videos): Encodes entire videos as neural networks; input is frame index, output is frame pixels
- **DS-NeRV** (CVPR 2024): Decomposes static and dynamic components for better video representation
- **ActINR** (CVPR 2025): Shares INR weights across frames, uses unique biases per frame via hypernetwork with continuous time index
- **NeRV-Enc**: Transformer-based hyper-network achieving 10^4x encoding speedup by eliminating gradient-based optimization
- **QS-NeRV** (ACM MM 2024): Quantization-aware variant for improved compression

#### Forensic Implications of INR-Based Video Compression
INR-based video codecs present unique forensic challenges:
- Each video is encoded as a unique neural network, making traditional codec fingerprinting inapplicable
- The compression artifacts depend on network architecture, training procedure, and quantization scheme
- **No published work directly addresses forensic detection of INR-compressed video** -- this is a significant research gap

### 5.3 INR for Image Compression and Forensics

#### COIN and Successors
- **COIN** (Dupont et al., 2021): Pioneered using INR for image compression -- fit a small neural network to represent an image, then compress the network weights
- SIREN-based models demonstrated efficiency over JPEG at low bitrates
- Post-training quantization of INR weights enables further compression

#### Secure Learned Image Codec (SLIC)
- Proposes proactive forensic use: creates self-destructive artifacts upon re-compression
- Any subsequent re-compression or editing attempts produce visually severe distortions
- Bridges neural compression and media authenticity verification

### 5.4 Potential Applications of INR in Forensics (Emerging/Speculative)

Based on the surveyed literature, several promising but underexplored directions emerge:

1. **INR fitting residuals as forensic features**: Real images and AI-generated images may have systematically different INR fitting residuals due to different statistical properties. The NRR approach (ECCV 2024) demonstrates this is viable.

2. **INR-based compression fingerprinting**: Just as JPEG leaves quantization table fingerprints, neural compression leaves architecture-specific artifacts. These could serve as provenance indicators.

3. **Coordinate-based anomaly detection**: SIREN's ability to represent signals continuously could be used to detect discontinuities introduced by splicing or inpainting at sub-pixel resolution.

4. **INR for compression-invariant representation**: Fitting an INR to an image and using the network weights (rather than pixel values) as a representation could potentially produce features that are more robust to subsequent compression.

---

## 6. Knowledge Distillation and Lightweight Deployment

### 6.1 DistilDIRE (2024)
- **Paper**: "DistilDIRE: A Small, Fast, Cheap and Lightweight Diffusion Synthesized Deepfake Detection"
- Addresses DIRE's impracticality (requires computing diffusion trajectory at inference)
- Uses knowledge distillation to create a lightweight student model
- **Link**: https://arxiv.org/abs/2406.00856

### 6.2 DK-CAST (2025)
- Tri-stream knowledge distillation framework for audio deepfake detection
- Achieves EER of 0.38% and 2.18% on benchmark datasets
- Maintains robustness against codec-induced distortions

### 6.3 Interactive Dual-Branch Network (IDBN, 2025)
- Adversarial knowledge distillation between CNN (spatial) and wavelet attention transformer (frequency) branches
- Specifically designed for compressed deepfake detection

### 6.4 DEEPDISTAL (CVPR 2024 Workshop)
- **Paper**: "DEEPDISTAL: Deepfake Dataset Distillation using Active Learning"
- Distills deepfake datasets into smaller, more informative training sets
- Reduces training data requirements while maintaining detection performance

---

## 7. Proactive Defense and Watermarking

### 7.1 Proactive Deepfake Detection
- **FractalForensics** (2025): Proactive detection via fractal watermarks
- **Big Brother is Watching** (2025): Embeds learnable hidden face template via semi-fragile invertible steganography
- **FaceSigns** (2024): Embeds 128-bit semi-fragile binary watermarks via UNet
- **EditGuard/OmniGuard** (2024/2025): Joint image-bit steganography for detection and localization

### 7.2 Survey on Proactive Defense
- **Paper**: "A Survey on Proactive Deepfake Defense: Disruption and Watermarking" (ACM Computing Surveys, 2025)
- Comprehensive survey of perturbation-based and watermarking-based proactive methods

---

## 8. State-of-the-Art Methods Summary

| Method | Year | Venue | Type | Key Contribution | Performance | Limitations |
|--------|------|-------|------|-----------------|-------------|-------------|
| CDDB | 2023 | WACV | Benchmark | First CL deepfake benchmark | Baseline comparisons | Limited generator diversity |
| UnivFD | 2023 | CVPR | Zero-shot | CLIP features for universal detection | +19.49 mAP on unseen generators | Requires CLIP backbone |
| SBI | 2022 | CVPR | Augmentation | Self-blended training data | +11.78% on DFDCP cross-dataset | No CL mechanism |
| NPR | 2024 | CVPR | Feature | Neighboring pixel relationships | 92.5% mean accuracy | Upsampling-dependent |
| DiffusionFake | 2024 | NeurIPS | Generalization | Reverse diffusion for feature enhancement | Cross-domain improvement | Diffusion model required |
| D3 | 2025 | CVPR | Multi-generator | Discrepancy learning | +5.3% OOD accuracy | Training cost |
| GPL | 2025 | ICCV | CL+Architecture | Hyperbolic alignment + gradient projection | 92.14% acc, -2.66% forgetting | Hyperbolic space complexity |
| SUR-LID | 2025 | CVPR | CL+Replay | Aligned feature isolation | 13.94% forgetting (vs 40.72% DFIL) | Feature space constraints |
| DevFD | 2025 | NeurIPS | CL+LoRA | Orthogonal LoRA MoE | Effective incremental adaptation | Growing parameter count |
| DARW | 2025 | Preprint | CL+Generative Replay | Domain-aware replay weighting | 0.9574 AUC (Protocol 1) | Requires replay generator |
| FIRE | 2025 | CVPR | Frequency | Mid-frequency reconstruction error | 100% AUC on DiffusionForensics | Diffusion-specific theory |
| PLADA | 2025 | Preprint | Compression-robust | Block effect erasure | 76.7% on compressed GANs | Moderate improvements |
| CLIP-Hyperspherical | 2025 | Preprint | Zero-shot | Metric learning on CLIP features | 96.62% on Celeb-DF-v2 | Foundation model dependency |
| Chronological CL | 2025 | Preprint | Benchmark | 7-year chronological evaluation | 0.915 C-AUC, ~0.5 FWT-AUC | Exposes fundamental limits |

---

## 9. Cross-Domain Connections

### 9.1 Compression x Continual Learning
- **Gap identified**: Almost no CL-for-detection paper evaluates under realistic compression conditions
- Montibeller et al. (2025) provides the framework for emulating social media compression, but has not been combined with CL methods
- **Hypothesis**: CL methods that rely on subtle forensic features (e.g., EWC preserving GAN fingerprint neurons) may be especially vulnerable to compression-induced forgetting -- a form of "compression-accelerated catastrophic forgetting"

### 9.2 Compression x Zero-Shot Detection
- CLIP-based detectors show inherent compression robustness because they rely on high-level semantic features rather than pixel-level artifacts
- NPR (neighboring pixel relationships) may be partially robust to compression since upsampling correlations are structural, not high-frequency
- FIRE's mid-frequency approach is explicitly designed to be robust to common perturbations including compression

### 9.3 Continual Learning x Zero-Shot Detection
- Foundation model features (CLIP) provide a strong initialization that may reduce the severity of catastrophic forgetting
- DevFD's LoRA approach is inherently compatible with foundation models -- could be combined with CLIP backbone
- **Promising direction**: Use CLIP features as a fixed representation, with lightweight CL-aware classification heads that adapt incrementally

### 9.4 INR x Compression x Detection
- INR fitting residuals could serve as compression-invariant forensic features (NRR, ECCV 2024)
- Neural compression (JPEG AI) creates new forensic challenges that traditional and CL-based detectors are not prepared for
- The convergence of neural compression standards and generative AI means future forensic systems must distinguish between three types of artifacts: generation artifacts, neural compression artifacts, and legitimate image content

---

## 10. Open Problems and Research Gaps

### 10.1 Critical Gaps

1. **CL under realistic compression**: No published work evaluates continual deepfake detection methods after data passes through actual social media pipelines. This is the single most important gap for production deployment.

2. **Video-specific CL**: Almost all CL-for-detection work focuses on images. Video deepfake detection with CL -- accounting for temporal artifacts, video codecs (H.264/H.265/VP9/AV1), and platform-specific transcoding -- is largely unexplored.

3. **Neural compression forensics at scale**: JPEG AI adoption will break current detectors (Cannas et al., 2025). No CL framework accounts for the shifting compression landscape alongside shifting generative models.

4. **Evaluation standardization**: Different papers use different protocols, datasets, and metrics, making fair comparison difficult. CDDB is a start but needs updating for diffusion-era generators.

### 10.2 Fundamental Challenges

5. **The forward transfer wall**: Fontana et al.'s FWT-AUC ~0.5 result suggests a fundamental limit. Can any CL method achieve meaningful forward transfer, or is rapid adaptation (few-shot) the only viable strategy?

6. **Compression-generation artifact entanglement**: As PLADA shows, compression artifacts and generation artifacts are easily confused. Disentangling them remains unsolved, especially when both vary across tasks in a CL setting.

7. **Adversarial robustness of CL detectors**: CL methods add complexity (replay buffers, regularization terms, adapter modules) that may introduce new attack surfaces. No work has studied adversarial robustness of CL-based detectors.

### 10.3 Promising Underexplored Directions

8. **INR-based forensic features**: Only one paper (NRR, ECCV 2024) explores INR for manipulation detection. The potential for INR fitting residuals as generator-agnostic, compression-robust features is largely untapped.

9. **Foundation model CL**: Combining CLIP/DINOv2 features with lightweight CL mechanisms (LoRA adapters, prompt tuning) for efficient incremental adaptation without full backbone retraining.

10. **Causal reasoning for detection**: CausalCLIP (2025) begins to address spurious correlations, but causal approaches to forensics (distinguishing causal generation artifacts from correlational ones) are nascent.

11. **Multi-modal CL**: Joint audio-visual deepfake detection with CL. Audio and video generators evolve on different timelines, requiring independent but coordinated incremental updates.

---

## 11. Recommendations for the AI-Generated Media Detector Project

Based on this survey and the project's current architecture (weighted ensemble of smoothing/texture/mode collapse detectors transitioning toward deep learning):

### Short-term (Phase 1-2 alignment)
1. **Add compression augmentation immediately**: Include JPEG (quality 30-95), WebP, and resize augmentation in all training pipelines. Use Montibeller et al.'s framework to estimate real platform parameters.
2. **Implement FIRE-style frequency analysis**: Mid-frequency reconstruction error is the current SOTA for compression-robust diffusion detection.
3. **Consider NPR features**: Neighboring pixel relationships are simple, effective, and complementary to current hand-crafted features.

### Medium-term (Phase 2-3 alignment)
4. **Use CLIP ViT-L/14 as backbone**: Provides strong zero-shot generalization foundation. Yermakov et al. show this works with minimal fine-tuning.
5. **Implement DevFD-style LoRA adapters**: When new generators appear, add orthogonal LoRA modules rather than retraining the full model. This is parameter-efficient and prevents forgetting.
6. **Build a replay buffer**: Maintain a curated set of examples from each encountered generator type. Even simple replay is the strongest CL baseline.

### Long-term (Phase 3+ alignment)
7. **Design for CL from the start**: The detector architecture should be modular, with a frozen foundation backbone, a set of LoRA adapters (one per generator family), and a lightweight fusion layer.
8. **Monitor for neural compression**: JPEG AI will change the forensic landscape. Build in codec identification and compression-aware normalization layers.
9. **Explore INR fitting residuals**: As a novel research contribution, INR reconstruction error as a forensic feature is promising and underexplored.

---

## 12. References

### Continual Learning for Deepfake Detection
1. Li, Y. et al. "A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials." WACV, 2023. https://arxiv.org/abs/2205.05467
2. Fontana, F. et al. "Revisiting Deepfake Detection: Chronological Continual Learning and the Limits of Generalization." arXiv preprint, 2025. https://arxiv.org/abs/2509.07993
3. Zhang, X. et al. "Generalization-Preserved Learning: Closing the Backdoor to Catastrophic Forgetting in Continual Deepfake Detection." ICCV, 2025. [CVF Open Access]
4. Cheng, J. et al. "Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection." CVPR, 2025. https://github.com/beautyremain/SUR-LID
5. Shen, H. et al. "When Generative Replay Meets Evolving Deepfakes: Domain-Aware Relative Weighting for Incremental Face Forgery Detection." arXiv preprint, 2025. https://arxiv.org/abs/2511.18436
6. DevFD authors. "DevFD: Developmental Face Forgery Detection by Learning Shared and Orthogonal LoRA Subspaces." NeurIPS, 2025. https://arxiv.org/abs/2509.19230
7. Tassone, F. et al. "Continuous fake media detection: adapting deepfake detectors to new generative techniques." Computer Vision and Image Understanding, 2024. https://arxiv.org/abs/2406.08171
8. "Continual Face Forgery Detection via Historical Distribution Preserving." IJCV, 2024.
9. "GAN-CNN Ensemble: A Robust Deepfake Detection Model Using Minimized Catastrophic Forgetting and Generative Replay." Procedia Computer Science, 2024.
10. "Continual Deepfake Detection Based on Multi-Perspective Sample Selection Mechanism." IEEE TNNLS, 2024.

### Audio Deepfake Continual Learning
11. Wani et al. "Audio Deepfake Detection: A Continual Approach with Feature Distillation and Dynamic Class Rebalancing." Preprint, 2025.
12. "Region-Based Optimization in Continual Learning for Audio Deepfake Detection." AAAI, 2025.
13. "Rehearsal with Auxiliary-Informed Sampling for Audio Deepfake Detection." 2025.
14. Xiao et al. "An Exemplar-Free Class Incremental Learning Method for Audio Deepfake Detection." Interspeech, 2025.

### Zero-Shot and Foundation Model Detection
15. Ojha, U. et al. "Towards Universal Fake Image Detectors that Generalize Across Generative Models." CVPR, 2023. https://arxiv.org/abs/2302.10174
16. Yermakov, A. et al. "Unlocking the Hidden Potential of CLIP in Generalizable Deepfake Detection." arXiv preprint, 2025. https://arxiv.org/abs/2503.19683
17. Pirogov, V. "Visual Language Models as Zero-Shot Deepfake Detectors." arXiv preprint, 2025. https://arxiv.org/abs/2507.22469
18. "Zero-Shot Visual Deepfake Detection: Can AI Predict and Prevent Fake Content Before It's Created?" arXiv preprint, 2025. https://arxiv.org/abs/2509.18461
19. "CausalCLIP: Causally-Informed Feature Disentanglement and Filtering." arXiv preprint, 2025. https://arxiv.org/abs/2512.13285
20. "Standing on the Shoulders of Giants: Reprogramming Visual-Language Model for General Deepfake Detection." 2024. https://arxiv.org/abs/2409.02664

### Generalization and Domain Adaptation
21. Shiohara, K. & Yamasaki, T. "Detecting Deepfakes with Self-Blended Images." CVPR, 2022. https://arxiv.org/abs/2204.08376
22. "DiffusionFake: Enhancing Generalization in Deepfake Detection via Guided Stable Diffusion." NeurIPS, 2024. https://arxiv.org/abs/2410.04372
23. Yang, Y. et al. "D3: Scaling Up Deepfake Detection by Learning from Discrepancy." CVPR, 2025. https://arxiv.org/abs/2404.04584
24. Tan, C. et al. "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection." CVPR, 2024. https://arxiv.org/abs/2312.10461
25. "Deepfake Detection that Generalizes Across Benchmarks." 2025. https://arxiv.org/abs/2508.06248

### Compression Robustness
26. Montibeller, A. et al. "Bridging the Gap: A Framework for Real-World Video Deepfake Detection via Social Network Compression Emulation." arXiv preprint, 2025. https://arxiv.org/abs/2508.08765
27. Li, M. et al. "Pay Less Attention to Deceptive Artifacts: Robust Detection of Compressed Deepfakes on Online Social Networks." arXiv preprint, 2025. https://arxiv.org/abs/2506.20548
28. Chu, B. et al. "FIRE: Robust Detection of Diffusion-Generated Images via Frequency-Guided Reconstruction Error." CVPR, 2025. https://arxiv.org/abs/2412.07140

### Neural Compression and Forensics
29. Cannas, E.D. et al. "Is JPEG AI going to change image forensics?" ICCV Workshop, 2025. https://arxiv.org/abs/2412.03261
30. "Forensic analysis of AI-compression traces in spatial and frequency domain." Pattern Recognition Letters, 2024.
31. "Forensic Recognition of Codec-Specific Image Compression Artefacts." ACM IH&MMSec, 2024.
32. "Implications of Neural Compression to Scientific Images." ACM IH&MMSec, 2025.

### Implicit Neural Representations
33. Zhang, Z. et al. "Image Manipulation Detection with Implicit Neural Representation and Limited Supervision." ECCV, 2024.
34. Sitzmann, V. et al. "Implicit Neural Representations with Periodic Activation Functions (SIREN)." NeurIPS, 2020.
35. "DS-NeRV: Implicit Neural Video Representation with Decomposed Static and Dynamic." CVPR, 2024.
36. "Bias for Action: Video Implicit Neural Representations with Bias Modulation." CVPR, 2025.
37. "On Quantizing Neural Representation for Compression." ICLR, 2025.

### Knowledge Distillation and Lightweight Detection
38. "DistilDIRE: A Small, Fast, Cheap and Lightweight Diffusion Synthesized Deepfake Detection." 2024. https://arxiv.org/abs/2406.00856
39. Wang, Z. et al. "DIRE for Diffusion-Generated Image Detection." ICCV, 2023. https://arxiv.org/abs/2303.09295
40. "DEEPDISTAL: Deepfake Dataset Distillation using Active Learning." CVPR Workshop, 2024.

### Proactive Defense
41. "A Survey on Proactive Deepfake Defense: Disruption and Watermarking." ACM Computing Surveys, 2025.
42. "FractalForensics: Proactive Deepfake Detection and Localization via Fractal Watermarks." 2025.
43. "Big Brother is Watching: Proactive Deepfake Detection via Learnable Hidden Face." 2025.

### Surveys
44. Liu et al. "A Review of Deepfake and Its Detection: From GANs to Diffusion Models." Int. J. Intelligent Systems, 2025.
45. "DeepFake detection in the AIGC era: A survey, benchmarks, and future perspectives." Information Fusion, 2025.
46. "Passive Deepfake Detection: A Comprehensive Survey across Multi-modalities." 2025. https://arxiv.org/abs/2411.17911
47. "Deepfake Media Forensics: Status and Future Challenges." PMC, 2025.

---

*Note: This report reflects the state of published and preprint literature as of March 2026. The field is evolving rapidly; papers from late 2025 and early 2026 may not be fully captured. ArXiv links are provided where available; some papers cited from conference proceedings may have updated versions on arXiv. Readers are encouraged to check for the most recent versions.*
