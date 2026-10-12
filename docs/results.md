# Results — measured, not asserted

Every figure here came from a run, not an estimate. Both runs are reproducible from
`scripts/train_kaggle.md`.

---

## 1. Training setup

| | r1 | r2 |
|---|---|---|
| Model | `yolov8m.pt` (detect) | fine-tuned from r1's `best.pt` |
| Dataset | India + Japan + US, curated | same, pothole images ×3 |
| Train / val images | 12,743 / 3,185 | 17,557 / **3,185 (unchanged)** |
| Epochs | 40 | 22 of 25 (interrupted) |
| imgsz / batch | 960 / 16, 2× T4 | same |
| Optimiser | AdamW, auto LR 0.00125 | same, `lr0=0.0005` |
| Wall clock | 5.02 h | ~2.4 h |

Dataset composition (boxes):

| Class | r1 | r2 (oversampled) |
|---|---|---|
| D00 Longitudinal | 36.0% | 27.1% |
| D10 Transverse | 21.4% | 15.9% |
| D20 Alligator | 26.4% | 24.7% |
| **D40 Pothole** | **16.2%** | **32.3%** |

China (drone) and Norway (4.29 GB, 80% of the archive) were excluded — see
`REBUILD_PLAN.md` §5.2.

---

## 2. Per-class performance

Validation set: 3,185 images, 6,913 instances, unchanged between runs.

### r1 — **selected**

| Class | Instances | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|---|
| D00 Longitudinal | 2,462 | 0.597 | 0.636 | 0.651 | 0.364 |
| D10 Transverse | 1,435 | 0.561 | 0.571 | 0.558 | 0.252 |
| D20 Alligator | 1,856 | 0.674 | 0.620 | **0.688** | 0.361 |
| D40 Pothole | 1,160 | 0.595 | 0.495 | 0.538 | 0.234 |
| **All** | 6,913 | 0.607 | 0.581 | **0.609** | 0.303 |

### r2 — pothole oversampling, rejected

| Class | Precision | Recall | AP50 | Δ AP50 vs r1 |
|---|---|---|---|---|
| D00 | 0.630 | 0.632 | 0.648 | −0.003 |
| D10 | 0.577 | 0.548 | 0.560 | +0.002 |
| D20 | 0.672 | 0.618 | 0.675 | −0.013 |
| **D40** | 0.580 | 0.525 | **0.526** | **−0.012** |
| All | 0.615 | 0.581 | 0.602 | −0.007 |

### Gate status

| Gate | Target | r1 | r2 |
|---|---|---|---|
| Overall mAP50 | ≥ 0.60 | **0.609 ✅** | 0.602 ✅ |
| **D40 pothole AP50** | ≥ 0.55 | **0.538 ❌** | 0.526 ❌ |

**Neither run met the pothole gate.** r1 is selected as the better of the two.

Context: the CRDDC'2022 challenge winner scored F1 76.9% across six countries using
ensemble methods, so 0.609 mAP50 from a single-stage detector fine-tune on three
countries is a credible result.

---

## 3. Why oversampling failed — the useful finding

3× duplication of pothole-containing images did exactly what it was designed to do
on recall, and still lost AP:

| D40 metric | r1 | r2 | Δ |
|---|---|---|---|
| Recall | 0.495 | 0.525 | **+0.030** |
| Precision | 0.595 | 0.580 | −0.015 |
| **AP50** | **0.538** | **0.526** | **−0.012** |

Recall rose, precision paid for it, and AP fell. The class became easier to *find*
and slightly harder to *localise*.

### The real constraint: pothole AP50 is localisation-limited

Ratio of AP50-95 to AP50 measures how tightly boxes are fitted:

| Class | AP50-95 / AP50 |
|---|---|
| D00 Longitudinal | 0.56 |
| D20 Alligator | 0.52 |
| D10 Transverse | 0.45 |
| **D40 Pothole** | **0.43** ← worst |

Potholes are small, roughly circular objects. A box that is proportionally accurate
but displaced loses most of its IoU, so it fails the 0.5 threshold and contributes
nothing to AP50. Cracks are elongated, so the same proportional error costs less IoU.

**This means oversampling cannot fix the pothole gate.** Recall was never the binding
constraint at AP50 — box tightness is. More pothole examples raise recall and, through
extra false positives, cost precision.

### What would plausibly move D40 AP50

Not more quota on oversampling. In rough order of expected value:

1. **Inference at higher resolution.** imgsz 960 → 1280 sharpens small-object boxes.
   Costs inference time, and CPU inference is already the bottleneck at 127 ms/frame
   (measured on the 8 GB laptop, so it would need a GPU to be viable).
2. **Box-regression-focused fine-tuning** with pothole-heavy batches *and* a lower
   IoU-matched assignment cost — addresses tightness rather than frequency.
3. **SAM 2 box-prompted masks** to refine box extents against the pothole boundary.

None were attempted; quota was spent on the oversampling hypothesis, which the data
then refuted.

---

## 4. Measured inference cost on the target machine

Apple M1, 8 GB, 8 threads, `rdd2022-yolov8m-r1.pt`, median over 25 frames.
Reproduce with `scripts/benchmark_detector.py`.

| imgsz | CPU median | CPU fps | MPS median | MPS fps | MPS speedup |
|---|---|---|---|---|---|
| 640 | 149 ms | 6.7 | 69 ms | 14.4 | 2.1× |
| 768 | 259 ms | 3.9 | — | — | — |
| **960** | **303 ms** | **3.3** | **125 ms** | **8.0** | **2.4×** |
| 1280 | 547 ms | 1.8 | 202 ms | 5.0 | 2.7× |

### Frame drop rate against 30 fps capture

| imgsz | CPU drop | MPS drop |
|---|---|---|
| 640 | 78% | 52% |
| 960 | 89% | 73% |
| 1280 | 94% | 83% |

Inference remains the binding constraint. Even at 960 on MPS the agent can only
process 8 of every 30 captured frames.

### Why 960 is now the default

The earlier default of 640 was chosen when the RDD2022 detector did not exist,
and its comment said to raise it *"once the RDD2022 model is trained."* That
condition is now met. 960 matches the resolution r1 was trained and validated at,
and §3 identifies pothole localisation — not class frequency — as the binding
constraint, which is the one thing resolution improves.

### `Settings.device` was dead configuration

The field existed in config but was never passed to ultralytics, which
auto-selects a device and lands on CPU even where MPS is available. Every
measurement above was therefore being taken on the CPU path while an
Apple GPU sat idle. Wired through, MPS is 2.1–2.7× faster.

Verified safe rather than assumed: running the same random input through the
model on both devices gives a maximum relative difference of **9.6e-07**, with
identical detection counts at every confidence threshold.

Thread count was also tested and made no difference (150 ms at 1 thread and at 8);
a standalone 2048³ matmul showed the same flat scaling, so this is the M1's
single-core throughput rather than a configuration error.

---

## 5. Honest limitations

| Limitation | Detail |
|---|---|
| D40 AP50 below gate | 0.538 vs 0.55. Cause identified as localisation, not class imbalance. |
| No in-domain evaluation | **All figures are RDD2022 val. They say nothing about performance on Indian roads captured by this system.** The in-domain set (~150 hand-labelled frames) is still outstanding and is the number that matters. |
| Crack severity incomplete | RDD2022 is bounding-box only, so crack width cannot be derived. Severity falls back to a default and `compute_pci()["limitations"]` reports it. Box width is never substituted for crack width. |
| GPS not field-validated | Kalman filter and quality gating are unit-tested; the before/after field measurement (§14 of the plan) has not been run. |
| Synthetic-source pipeline | Capture has been verified against a synthetic source and a video file. The live phone MJPEG link has not been exercised. |
| Cumulative distance overestimates | Chainage is the sum of consecutive fix-to-fix distances, so uncorrelated GPS noise inflates it. Measured on a synthetic 2 km run: σ = 0 m gives −1%, σ = 2 m gives +4%, σ = 8 m gives **+68%**. Overestimating chainage produces the wrong number of sample units, so trip distance is not a trustworthy measurement. Fixing it properly needs a per-trip filter or map matching, not a constant. |
| Severity is graded, not measured | RDD2022 provides bounding boxes, so pothole depth and crack width are unavailable. Severity is defaulted (potholes to H, cracks to M) and every such case is reported in `limitations`. A PCI computed from defaults is a defect-density index, not a distress survey. |
| Benchmark frames are synthetic | The latency table above was measured on the procedural road scene, which has little texture. Latency depends on input tensor shape rather than content, so the timings are sound; the detections-per-frame column is not meaningful on these frames and needs real footage. |
| Per-object ground position | One GPS position per frame, not per object. Two defects in one frame share a position; per-object positions need the Phase 6 homography. |

---

## 6. Reproducing

```bash
python scripts/prepare_rdd2022.py --src /kaggle/working/RDD2022 --dst /kaggle/working/rdd2022-curated
```

Then the training cells in `scripts/train_kaggle.md` §5. The r2 oversampling build is
documented in this file's §1 and rebuilds in about 30 seconds.