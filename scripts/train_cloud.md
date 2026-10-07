# Cloud training runbook — RDD2022 fine-tune

The local machine is 8 GB RAM with no GPU. Training runs on Kaggle or Colab.

**Launch training before anything else is ready.** It is the only task in the
rebuild that cannot be compressed later, because it waits on external quota.

---

## 1. Get the dataset

Verified against figshare's API. **The archive is 12.65 GB** — this is the single
biggest practical obstacle, more than the training itself.

| | |
| --- | --- |
| Article | `21431547` — *RDD2022, released through CRDDC'2022* |
| DOI | `10.6084/m9.figshare.21431547.v1` |
| Archive | `RDD2022_released_through_CRDDC2022.zip` |
| File id | `38030910` |
| Direct URL | `https://ndownloader.figshare.com/files/38030910` |

```python
# In Colab / Kaggle. 12.65 GB - expect this to take a while and to risk
# session timeouts. Do it once, then cache the curated subset.
!wget -q --show-progress -O rdd2022.zip https://ndownloader.figshare.com/files/38030910
```

> **Do not guess the ndownloader file id.** A wrong id returns an HTML page
> silently, and `unzip` then fails with *"End-of-central-directory signature not
> found"* — which looks like a corrupt download rather than a bad URL. Check
> `ls -lh rdd2022.zip` first: it should be ~13G, not a few KB.

Expected layout — note `annotations/xmls`, **Pascal VOC XML, not COCO**:

```
RDD2022/
  India/            train/images/*.jpg   train/annotations/xmls/*.xml
  Japan/            train/images/*.jpg   train/annotations/xmls/*.xml
  United_States/    train/images/*.jpg   train/annotations/xmls/*.xml
  Norway/  Czech/  China_Drone/  China_MotorBike/
```

The authoritative label map (`label_map.pbtxt`, file id `38030820`) is:

```
D00 = 1,  D10 = 2,  D20 = 3,  D40 = 4
```

Those are **1-indexed TF Object Detection ids, not YOLO class indices**. COCO
re-uploads use different ids again (D40 as `category_id 7`). Never assume an id
ordering — resolve the D-code from the class *name*, which is what
`prepare_rdd2022.py` does.

**Disk.** Colab free gives ~78 GB, so 13 GB extracted fits, but download and
extraction are the fragile part. Mitigations:

- Curate immediately after extracting, then delete the raw zip.
- Save `rdd2022-curated/` to Drive so a reconnect does not force a re-download.
- Prefer Kaggle if a pre-converted RDD2022 exists there — but verify it (4
  classes, pothole mapped correctly) before trusting it.

---

## 2. Inspect before curating

```bash
python scripts/prepare_rdd2022.py --src /path/to/RDD2022 --inspect
```

This reports, per country, whether the annotations are VOC or COCO, how many
images carry target-class boxes, and the per-class box counts.

> **The traps this avoids:**
>
> 1. The official release ships **Pascal VOC XML**, not `coco.json`. A
>    COCO-only script finds nothing and reports an empty dataset.
> 2. `category_id` is **not** the class index. D40 (pothole) is 4 in
>    `label_map.pbtxt` but 7 in common COCO re-uploads. The script resolves the
>    D-code from the class *name* in both formats.

---

## 3. Curate

```bash
python scripts/prepare_rdd2022.py \
    --src /path/to/RDD2022 \
    --dst data/rdd2022-curated \
    --countries India Japan United_States \
    --val-fraction 0.2
```

China is excluded by default — its subset is drone/oblique imagery, a different
domain that degrades forward-facing vehicle-mounted inference. Add `Norway` if
you want more data; it is also vehicle-mounted.

Read the report. It flags rare classes. **If D10 lands below ~500 labels, expect
poor recall on transverse cracks** — India has only 68 in the raw dataset. This
is a known limitation, not a bug to hide; report per-class AP.

---

## 4. Train

```python
from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")  # segmentation: masks enable crack width measurement
model.train(
    data="data/rdd2022-curated/roadscope.yaml",
    epochs=80,
    imgsz=960,  # upscale from native 600-720; thin cracks vanish otherwise
    batch=8,
    device="cuda",
    patience=15,
    project="runs",
    name="rdd2022-roadscope-r1",
    # lighting varies enormously on Indian roads
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,  # horizontal only — roads have no vertical flip
    fliud=0.0,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,  # pothole oversampling
    erasing=0.2,
)
```

---

## 5. Evaluate — per class, never just mAP

```python
from ultralytics import YOLO

model = YOLO("runs/rdd2022-roadscope-r1/weights/best.pt")
metrics = model.val(data="data/rdd2022-curated/roadscope.yaml", split="val")

print(f"mAP50      {metrics.box.map50:.4f}")
print(f"mAP50-95   {metrics.box.map:.4f}")
for i, name in model.names.items():
    print(
        f"  {name}  AP50 {metrics.box.ap50[i]:.4f}  AP50-95 {metrics.box.ap[i - 1] if i else metrics.box.ap[i]:.4f}"
    )
```

Reference points:

| Metric | Gate | Note |
| --- | --- | --- |
| Overall mAP50 | ≥ 0.60 | CRDDC'2022 winner scored F1 76.9%, so this is realistic |
| **D40 AP50** | ≥ 0.55 | Pothole — the rarest class, the project's namesake |
| D40 recall | ≥ 0.70 | On the in-domain set, not RDD2022 |
| Lane-marking FP | 0 | Measured separately on paved-marking frames |

**Commit `metrics.json` for every run.** The point is the trend across runs, not
the final number alone.

---

## 6. Label an in-domain test set

RDD2022 mAP will *not* predict performance on your roads. Capture at least 150
frames of your own dashcam video and hand-label them.

```bash
python -m roadscope.agent --camera-url http://<phone>:8080/stream
```

Save frames, annotate in CVAT or LabelImg, evaluate against that set. This number
is the one you quote in the viva.

---

## 7. Publish

Upload `best.pt` to a GitHub Release. **Do not commit it to git** — that is how
the repository ended up 402 MB.

```bash
gh release create v0.1.0 runs/rdd2022-roadscope-r1/weights/best.pt \
    --title "RoadScope RDD2022 detector" --notes "mAP50 0.xx, D40 AP50 0.xx"
```

Consumers fetch it with `scripts/download_weights.sh`.

---

## Expected timeline

| Run | Days | Purpose |
| --- | --- | --- |
| r1 | 3–4 | Baseline. Reveals which class collapses. |
| r2 | 2–3 | Imbalance fixes driven by r1's per-class AP. |
| r3 | 2–3 | Best curated mix, final. |

If D40 is still short at r2, run r4 with pothole-only oversampling and stronger
`copy_paste`. **Budget for four runs** — pothole is the rarest class, and this is
the phase with zero schedule float.