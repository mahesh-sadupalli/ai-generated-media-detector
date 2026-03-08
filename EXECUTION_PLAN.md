# DeepFake Detector — Product Execution Plan
### From Research Prototype to Production Product

---

## SLIDE 1: THE OPPORTUNITY

**Problem:** AI-generated content (deepfakes) is exploding — used in fraud, misinformation, identity theft, and scams. People and organizations need a reliable way to verify if content is real or AI-generated.

**Market Context:**
- Deepfake incidents grew 400%+ year-over-year
- Governments worldwide passing AI content regulation (EU AI Act, US DEFIANCE Act)
- Enterprise demand: media companies, banks, insurance, law enforcement, social platforms
- Consumer demand: journalists, content creators, everyday users

**Our Angle:** Detection powered by understanding *how* GANs generate artifacts — not just pattern matching, but explainable reasoning about *why* content appears fake.

---

## SLIDE 2: WHAT WE HAVE TODAY (Current State)

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT PROTOTYPE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✅ Three Hand-Crafted Detectors                            │
│     ├── Smoothing Detector (FFT, Sobel, texture variance)   │
│     ├── Texture Detector (LBP, GLCM, spectral entropy)     │
│     └── Mode Collapse Detector (symmetry, autocorrelation)  │
│                                                             │
│  ✅ Combined Classifier (weighted scoring + explanations)   │
│  ✅ Face Extraction Pipeline (OpenCV Haar Cascade)          │
│  ✅ Synthetic Test Data (60 generated samples)              │
│  ✅ Working Demo Script                                     │
│  ✅ Clean Modular Architecture                              │
│                                                             │
│  ❌ No trained ML model (hand-crafted features only)        │
│  ❌ No web interface or API                                 │
│  ❌ No real-world evaluation metrics                        │
│  ❌ Simulated artifacts (not real GAN outputs)              │
│  ❌ No user-facing product                                  │
│  ❌ No database, auth, or infrastructure                    │
│                                                             │
│  DETECTION ACCURACY (on synthetic data):                    │
│     Real faces:     ~0.617 score                            │
│     Generated:      ~0.759 score                            │
│     Gap:            ~0.14 (narrow — needs improvement)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Honest Assessment:** We have a working proof-of-concept with solid architecture, but the detection relies on hand-crafted features with narrow margins. To become a product, we need a trained deep learning model, real-world data, and a full web stack.

---

## SLIDE 3: PRODUCT VISION

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   User uploads image/video  ──→  Processing  ──→  Verdict        │
│                                                                  │
│   ┌──────────┐    ┌────────────────────┐    ┌────────────────┐   │
│   │          │    │  Face Detection    │    │  RESULT:       │   │
│   │  Upload  │───→│  Feature Extract   │───→│  ✅ REAL (92%) │   │
│   │  Image   │    │  Model Inference   │    │  or            │   │
│   │  /Video  │    │  Explanation Gen   │    │  ❌ FAKE (87%) │   │
│   │          │    │                    │    │                │   │
│   └──────────┘    └────────────────────┘    │  WHY:          │   │
│                                             │  • Smoothing   │   │
│                                             │    detected    │   │
│                                             │  • Texture     │   │
│                                             │    anomalies   │   │
│                                             │  • Heatmap     │   │
│                                             │    overlay     │   │
│                                             └────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Key Differentiator:** Not just "fake or real" — we tell users **WHY** with visual evidence (heatmaps, artifact breakdowns, confidence per region).

---

## SLIDE 4: EXECUTION ROADMAP (6 Phases)

```
 PHASE 1          PHASE 2          PHASE 3          PHASE 4          PHASE 5          PHASE 6
 Model &          Backend          Frontend         Integration      Testing &        Launch &
 Data             API              Web App          & Polish         Hardening        Scale

 Week 1-3         Week 3-4         Week 4-6         Week 6-7         Week 7-8         Week 8+
 ────────         ────────         ────────         ────────         ────────         ────────
 Dataset          FastAPI          React/Next.js    End-to-end       Load testing     Deploy
 Training         Endpoints        Upload UI        flow             Security         Monitor
 Evaluation       File handling    Results UI       Error handling   Edge cases       Iterate
 Model export     Queue system     Heatmaps         Responsive       CI/CD            Scale
```

---

## SLIDE 5: PHASE 1 — MODEL & DATA (Week 1-3)

**Goal:** Replace hand-crafted features with a trained deep learning model that achieves >90% accuracy on real-world deepfake datasets.

### 1A. Dataset Acquisition
```
┌─────────────────────────────────────────────────────────────┐
│  TRAINING DATA SOURCES                                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  FaceForensics++ (already partially set up)                 │
│  ├── 1,000 original videos                                  │
│  ├── 4,000 manipulated videos (4 methods)                   │
│  └── Standard benchmark for deepfake detection              │
│                                                             │
│  CelebDF-v2                                                 │
│  ├── 590 real + 5,639 deepfake videos                       │
│  └── Higher quality deepfakes                               │
│                                                             │
│  DFDC (Facebook Deepfake Detection Challenge)               │
│  ├── 100,000+ videos                                        │
│  └── Largest public deepfake dataset                        │
│                                                             │
│  WildDeepfake (optional — real-world scraped fakes)         │
│  └── Covers in-the-wild generation methods                  │
│                                                             │
│  TARGET: 50,000+ face images (balanced real/fake)           │
│  SPLIT:  70% train / 15% validation / 15% test             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1B. Model Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│  HYBRID MODEL: Deep Learning + Hand-Crafted Features            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Image (224x224)                                          │
│       │                                                         │
│       ├──→ [EfficientNet-B4 Backbone]  ──→ CNN Features (1792d) │
│       │    (pretrained ImageNet, fine-tuned)                     │
│       │                                                         │
│       ├──→ [Our Existing Detectors]    ──→ Hand-crafted (12d)   │
│       │    Smoothing (4) + Texture (4) + Collapse (4)           │
│       │                                                         │
│       └──→ [Frequency Analysis Head]   ──→ Frequency (256d)    │
│            (DCT/FFT spectral features)                          │
│       │                                                         │
│       ▼                                                         │
│  [Concatenate: 1792 + 12 + 256 = 2060d]                        │
│       │                                                         │
│       ▼                                                         │
│  [FC 2060→512→256→2]  ──→  Output: P(real), P(fake)            │
│                                                                 │
│  LOSS: Binary Cross-Entropy + Artifact Classification Aux Loss  │
│  OPTIMIZER: AdamW, lr=1e-4, cosine schedule                     │
│  REGULARIZATION: Dropout 0.3, label smoothing 0.1               │
│                                                                 │
│  WHY THIS ARCHITECTURE:                                         │
│  • EfficientNet-B4 is SOTA for image classification efficiency  │
│  • Our hand-crafted features add interpretability               │
│  • Frequency head catches spectral artifacts CNNs miss          │
│  • Hybrid approach = better than either alone                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1C. Training & Evaluation
```
Targets:
  • Accuracy:    > 92% on FaceForensics++ test set
  • AUC-ROC:     > 0.95
  • F1 Score:    > 0.90
  • False Pos:   < 5% (critical — don't accuse real content)
  • Latency:     < 500ms per image on GPU, < 2s on CPU

Deliverables:
  • Trained model weights (.pt file)
  • Evaluation report with confusion matrix, ROC curve
  • Cross-dataset generalization results
  • Model card documenting capabilities and limitations
```

---

## SLIDE 6: PHASE 2 — BACKEND API (Week 3-4)

**Goal:** Production-ready FastAPI backend that handles file uploads, runs inference, and returns structured results.

### API Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                      BACKEND STACK                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Framework:     FastAPI (async, high-performance)            │
│  Server:        Uvicorn + Gunicorn (multi-worker)           │
│  Task Queue:    Celery + Redis (async processing)           │
│  Database:      PostgreSQL (results, users, history)        │
│  Storage:       S3 / local (uploaded files, temp)           │
│  Cache:         Redis (model warm-up, rate limiting)        │
│  Auth:          JWT tokens (optional for MVP)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### API Endpoints
```
POST /api/v1/detect/image
  ├── Input:  multipart/form-data (image file, max 10MB)
  ├── Output: { verdict, confidence, artifacts[], heatmap_url, explanation }
  └── Latency target: < 3 seconds

POST /api/v1/detect/video
  ├── Input:  multipart/form-data (video file, max 100MB)
  ├── Output: { verdict, confidence, frame_results[], timeline }
  └── Processing: async with job ID → poll for results

GET /api/v1/results/{job_id}
  ├── Output: job status + results when complete
  └── Used for video processing polling

GET /api/v1/health
  └── Health check + model status
```

### Response Schema
```json
{
  "verdict": "FAKE",
  "confidence": 0.87,
  "processing_time_ms": 1240,
  "faces_detected": 1,
  "analysis": {
    "smoothing_score": 0.76,
    "texture_score": 0.42,
    "mode_collapse_score": 0.81,
    "frequency_anomaly_score": 0.69
  },
  "artifacts_detected": [
    {
      "type": "smoothing",
      "severity": "high",
      "description": "Unnatural smoothing detected around facial features",
      "regions": [{"x": 45, "y": 62, "w": 130, "h": 100}]
    }
  ],
  "heatmap_url": "/api/v1/heatmaps/abc123.png",
  "explanation": "This image shows signs of AI generation. Key indicators: over-smoothed skin texture (76% confidence), repetitive patterns suggesting mode collapse (81% confidence)."
}
```

---

## SLIDE 7: PHASE 3 — FRONTEND WEB APP (Week 4-6)

**Goal:** Clean, trustworthy web interface where users upload content and get clear, visual results.

### Tech Stack Decision
```
┌──────────────────────────────────────────────────────────┐
│  OPTION A (Recommended): Next.js Full-Stack              │
│  ├── Next.js 14 (App Router)                             │
│  ├── Tailwind CSS + shadcn/ui                            │
│  ├── Framer Motion (animations)                          │
│  ├── React Dropzone (file upload)                        │
│  └── Deployed on Vercel                                  │
│                                                          │
│  OPTION B: Separate SPA                                  │
│  ├── React + Vite                                        │
│  ├── Deployed separately from backend                    │
│  └── More flexible but more infrastructure               │
│                                                          │
│  OPTION C: Streamlit (fastest MVP)                       │
│  ├── Python-native, minimal frontend code                │
│  ├── Quick to build but limited customization            │
│  └── Good for initial demo, not for scale                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### UI Pages & Components
```
PAGE 1: LANDING / UPLOAD
┌──────────────────────────────────────────────────────────┐
│                                                          │
│              🔍 DeepFake Detector                        │
│         Verify if content is AI-generated                │
│                                                          │
│   ┌──────────────────────────────────────────────────┐   │
│   │                                                  │   │
│   │     Drag & drop your image or video here         │   │
│   │              or click to browse                   │   │
│   │                                                  │   │
│   │     Supports: JPG, PNG, MP4, MOV (max 100MB)     │   │
│   │                                                  │   │
│   └──────────────────────────────────────────────────┘   │
│                                                          │
│              [ Analyze Content ]                         │
│                                                          │
│   Recent: 50,000+ images analyzed | 94% accuracy         │
│                                                          │
└──────────────────────────────────────────────────────────┘

PAGE 2: RESULTS
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  VERDICT:  ❌ LIKELY AI-GENERATED  (87% confidence)      │
│                                                          │
│  ┌─────────────────┐  ┌────────────────────────────┐     │
│  │                 │  │ ARTIFACT BREAKDOWN          │     │
│  │   [Original]    │  │                            │     │
│  │   [Heatmap      │  │ Smoothing    ████████░░ 76%│     │
│  │    Overlay]     │  │ Texture      ████░░░░░░ 42%│     │
│  │                 │  │ Mode Collapse████████░░ 81%│     │
│  │                 │  │ Frequency    ██████░░░░ 69%│     │
│  └─────────────────┘  │                            │     │
│                        │ EXPLANATION:               │     │
│  Toggle:               │ • Skin texture appears     │     │
│  [Original] [Heatmap]  │   unnaturally smooth       │     │
│  [Artifacts]           │ • Repetitive patterns in   │     │
│                        │   facial structure          │     │
│                        │ • Frequency spectrum shows  │     │
│                        │   GAN fingerprint           │     │
│                        └────────────────────────────┘     │
│                                                          │
│  [ Download Report ]  [ Analyze Another ]                │
│                                                          │
└──────────────────────────────────────────────────────────┘

PAGE 3: ABOUT / HOW IT WORKS
  - Visual explanation of detection methodology
  - Accuracy metrics and limitations disclaimer
  - Research basis and methodology
```

### Heatmap Visualization
```
Key Feature: Visual evidence showing WHERE artifacts were detected

Implementation:
  1. Run detection on sliding windows / face regions
  2. Generate per-pixel anomaly scores
  3. Create color-coded overlay (green=normal, red=suspicious)
  4. Allow user to toggle between original and heatmap
  5. Highlight specific artifact regions with bounding boxes
```

---

## SLIDE 8: PHASE 4 — INTEGRATION & POLISH (Week 6-7)

```
┌─────────────────────────────────────────────────────────────┐
│  END-TO-END FLOW                                            │
│                                                             │
│  User → Upload → Validate → Queue → Detect → Store → Show  │
│                                                             │
│  Key Integration Work:                                      │
│                                                             │
│  1. File Validation Pipeline                                │
│     • File type verification (magic bytes, not extension)   │
│     • Size limits enforcement                               │
│     • Image corruption detection                            │
│     • Video codec compatibility                             │
│     • EXIF metadata extraction                              │
│                                                             │
│  2. Processing Pipeline                                     │
│     • Face detection → crop → resize → normalize            │
│     • Multi-face handling (analyze each face separately)     │
│     • Video: frame sampling strategy (not every frame)      │
│     • GPU batch processing for throughput                    │
│                                                             │
│  3. Results Generation                                      │
│     • Heatmap generation and caching                        │
│     • PDF report generation (downloadable)                  │
│     • Confidence calibration (scores → probabilities)       │
│                                                             │
│  4. Error Handling                                           │
│     • No face detected → clear message                      │
│     • Low quality image → warning + best effort             │
│     • Processing timeout → graceful failure                 │
│     • Unsupported format → specific guidance                │
│                                                             │
│  5. Mobile Responsiveness                                   │
│     • Touch-friendly upload                                 │
│     • Responsive results layout                             │
│     • Camera capture integration (mobile browsers)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 9: PHASE 5 — TESTING & HARDENING (Week 7-8)

```
┌──────────────────────────────────────────────────────────┐
│  TESTING STRATEGY                                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Unit Tests (pytest)                                     │
│  ├── Each detector independently                         │
│  ├── Classifier logic                                    │
│  ├── API endpoint contracts                              │
│  └── File validation                                     │
│                                                          │
│  Integration Tests                                       │
│  ├── Upload → Detection → Results (full pipeline)        │
│  ├── Video processing end-to-end                         │
│  └── Concurrent request handling                         │
│                                                          │
│  Model Tests                                             │
│  ├── Accuracy on held-out test set                       │
│  ├── Cross-dataset generalization                        │
│  ├── Adversarial robustness (compressed, cropped, etc)   │
│  └── Bias testing (across demographics)                  │
│                                                          │
│  Performance Tests                                       │
│  ├── Latency benchmarks (P50, P95, P99)                  │
│  ├── Concurrent users (target: 100 simultaneous)         │
│  ├── Memory profiling under load                         │
│  └── GPU utilization optimization                        │
│                                                          │
│  Security                                                │
│  ├── File upload sanitization (prevent malicious files)   │
│  ├── Rate limiting (prevent abuse)                       │
│  ├── Input validation (prevent injection)                │
│  └── CORS configuration                                  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## SLIDE 10: PHASE 6 — DEPLOYMENT & LAUNCH (Week 8+)

### Infrastructure
```
┌──────────────────────────────────────────────────────────────┐
│  DEPLOYMENT ARCHITECTURE                                     │
│                                                              │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────────┐   │
│  │          │    │           │    │                      │   │
│  │  Vercel  │───→│  FastAPI  │───→│  Model Server        │   │
│  │  (Next)  │    │  (API)    │    │  (GPU Instance)      │   │
│  │          │    │           │    │                      │   │
│  └──────────┘    └─────┬─────┘    └──────────────────────┘   │
│                        │                                     │
│                  ┌─────┴─────┐                               │
│                  │           │                               │
│                  │  Redis    │                               │
│                  │  (Queue   │                               │
│                  │  + Cache) │                               │
│                  │           │                               │
│                  └─────┬─────┘                               │
│                        │                                     │
│                  ┌─────┴─────┐                               │
│                  │PostgreSQL │                               │
│                  │(Results,  │                               │
│                  │ Analytics)│                               │
│                  └───────────┘                               │
│                                                              │
│  OPTION A: Cloud (AWS/GCP)                                   │
│  ├── EC2 g4dn.xlarge (T4 GPU) — ~$0.50/hr                   │
│  ├── RDS PostgreSQL                                          │
│  ├── ElastiCache Redis                                       │
│  ├── S3 for file storage                                     │
│  └── Estimated: $200-400/month at low traffic                │
│                                                              │
│  OPTION B: Budget-Friendly                                   │
│  ├── Railway / Render (API)                                  │
│  ├── Modal / Replicate (GPU inference)                       │
│  ├── Vercel (frontend)                                       │
│  ├── Supabase (database)                                     │
│  └── Estimated: $50-150/month at low traffic                 │
│                                                              │
│  OPTION C: Self-Hosted (if you have a GPU machine)           │
│  ├── Docker Compose on personal server                       │
│  ├── Cloudflare Tunnel for public access                     │
│  └── Estimated: electricity cost only                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## SLIDE 11: WHAT CHANGES IN CODEBASE (File-Level Plan)

```
KEEP & ENHANCE (existing files):
──────────────────────────────────
src/artifact_detectors/smoothing_detector.py     → Keep as feature extractor
src/artifact_detectors/texture_detector.py       → Keep as feature extractor
src/artifact_detectors/mode_collapse_detector.py → Keep as feature extractor
src/utils/simple_face_detection.py               → Replace with MTCNN/RetinaFace

MODIFY (existing files):
──────────────────────────────────
src/artifact_detectors/combined_artifact_classifier.py
  → Refactor to use trained model instead of hardcoded weights
requirements.txt
  → Add new dependencies (celery, redis, sqlalchemy, etc.)

CREATE (new files):
──────────────────────────────────
src/models/
  ├── hybrid_detector.py          # EfficientNet + feature fusion model
  ├── frequency_head.py           # DCT/FFT spectral analysis network
  └── train.py                    # Training pipeline

src/api/
  ├── main.py                     # FastAPI application entry point
  ├── routes/
  │   ├── detect.py               # /detect/image and /detect/video endpoints
  │   ├── results.py              # /results/{job_id} endpoint
  │   └── health.py               # /health endpoint
  ├── schemas.py                  # Pydantic request/response models
  ├── tasks.py                    # Celery async tasks
  └── config.py                   # Environment configuration

src/processing/
  ├── pipeline.py                 # Orchestrates detection pipeline
  ├── face_detector.py            # Modern face detection (MTCNN)
  ├── heatmap.py                  # Heatmap generation
  └── report.py                   # PDF report generation

frontend/                         # Next.js application
  ├── app/
  │   ├── page.tsx                # Landing + upload page
  │   ├── results/[id]/page.tsx   # Results page
  │   └── about/page.tsx          # How it works page
  ├── components/
  │   ├── FileUploader.tsx        # Drag-and-drop upload
  │   ├── ResultsDisplay.tsx      # Verdict + breakdown
  │   ├── HeatmapViewer.tsx       # Interactive heatmap overlay
  │   ├── ArtifactChart.tsx       # Score visualization
  │   └── ExplanationPanel.tsx    # Human-readable reasoning
  └── lib/
      └── api.ts                  # API client

tests/
  ├── test_detectors.py           # Unit tests for detectors
  ├── test_api.py                 # API endpoint tests
  ├── test_pipeline.py            # Integration tests
  └── test_model.py               # Model accuracy tests

docker-compose.yml                # Local development stack
Dockerfile                        # API container
Makefile                          # Common commands
.env.example                      # Environment template
```

---

## SLIDE 12: PRIORITY & EFFORT MATRIX

```
                        HIGH IMPACT
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         │  Train real      │  Build web UI    │
         │  ML model        │  with heatmaps   │
         │  [PHASE 1]       │  [PHASE 3]       │
         │  ~3 weeks        │  ~2 weeks        │
         │                  │                  │
HIGH ────┼──────────────────┼──────────────────┼──── LOW
EFFORT   │                  │                  │    EFFORT
         │  GPU deployment  │  FastAPI backend │
         │  infrastructure  │  [PHASE 2]       │
         │  [PHASE 6]       │  ~1 week         │
         │  ~1 week         │                  │
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                        LOW IMPACT

CRITICAL PATH:  Model Training → Backend API → Frontend
                (everything else can parallel)
```

---

## SLIDE 13: RISK & MITIGATION

```
┌────────────────────┬───────────────┬──────────────────────────────┐
│ RISK               │ LIKELIHOOD    │ MITIGATION                   │
├────────────────────┼───────────────┼──────────────────────────────┤
│ Model doesn't      │ Medium        │ Start with EfficientNet      │
│ generalize to      │               │ fine-tune (proven baseline). │
│ real-world fakes   │               │ Train on multiple datasets.  │
│                    │               │ Add "confidence too low"     │
│                    │               │ fallback.                    │
├────────────────────┼───────────────┼──────────────────────────────┤
│ High false          │ Medium        │ Calibrate thresholds on      │
│ positive rate       │               │ diverse real images.         │
│ (real flagged       │               │ Add "uncertain" category.    │
│ as fake)            │               │ Never say 100% fake.         │
├────────────────────┼───────────────┼──────────────────────────────┤
│ GPU costs too       │ Low-Medium    │ Use Modal/Replicate for      │
│ high at scale       │               │ pay-per-use GPU. Optimize    │
│                    │               │ with model quantization      │
│                    │               │ (INT8) and batching.         │
├────────────────────┼───────────────┼──────────────────────────────┤
│ New generation      │ High          │ Design for retraining.       │
│ methods bypass      │               │ Monitor detection rates.     │
│ our detectors       │               │ Continuous data collection.  │
│                    │               │ Modular architecture for     │
│                    │               │ swapping detection heads.    │
├────────────────────┼───────────────┼──────────────────────────────┤
│ Legal/ethical       │ Low           │ Clear disclaimers: "This is  │
│ liability from      │               │ an estimate, not proof."     │
│ wrong verdicts      │               │ Never claim 100% accuracy.   │
│                    │               │ Provide confidence ranges.   │
└────────────────────┴───────────────┴──────────────────────────────┘
```

---

## SLIDE 14: IMMEDIATE NEXT STEPS (What to do Monday)

```
STEP 1 (TODAY):  Decide on product direction
  ┌─────────────────────────────────────────────┐
  │  Decision A: Web app (Next.js + FastAPI)    │ ← Recommended
  │  Decision B: Streamlit MVP (fastest demo)   │
  │  Decision C: API-only (developer-focused)   │
  └─────────────────────────────────────────────┘

STEP 2 (DAY 1-2):  Set up training infrastructure
  • Download FaceForensics++ full dataset
  • Set up training script with EfficientNet-B4
  • Configure GPU environment (local / cloud)

STEP 3 (DAY 3-7):  Train and evaluate model
  • Train hybrid model
  • Evaluate on test set + cross-dataset
  • Iterate until >90% accuracy

STEP 4 (DAY 7-10):  Build FastAPI backend
  • Implement /detect/image endpoint
  • Add file upload handling
  • Integrate trained model

STEP 5 (DAY 10-14):  Build frontend
  • Upload interface
  • Results display with heatmaps
  • Deploy MVP

STEP 6 (DAY 14+):  Polish & launch
  • Error handling, edge cases
  • Performance optimization
  • Deploy to production
```

---

## SLIDE 15: SUCCESS METRICS

```
LAUNCH CRITERIA (MVP):
  ✓ >90% accuracy on FaceForensics++ test set
  ✓ <3 second response time for images
  ✓ <30 second response time for short videos
  ✓ Clean upload → result flow works end-to-end
  ✓ Heatmap visualization working
  ✓ Human-readable explanations generated
  ✓ Mobile-responsive design
  ✓ Handles edge cases gracefully

GROWTH METRICS (Post-Launch):
  • Daily active users
  • Images analyzed per day
  • Average confidence score distribution
  • False positive/negative rate (user feedback)
  • API response time P95
```

---

## SUMMARY

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   TODAY            →    2 WEEKS          →    6 WEEKS        │
│                                                              │
│   Research              Working               Production     │
│   Prototype             Model + API            Web App       │
│                                                              │
│   Hand-crafted          Trained DL model       Full UI       │
│   features              + API endpoints        + Heatmaps    │
│   60 test images        >90% accuracy          + Reports     │
│   CLI demo              Real-time inference    + Deployment  │
│                                                              │
│   BIGGEST UNLOCK: Training a real model on real data.        │
│   Everything else is engineering — which we can do fast.     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```
