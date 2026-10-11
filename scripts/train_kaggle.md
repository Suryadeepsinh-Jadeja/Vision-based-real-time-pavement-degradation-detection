# Kaggle training guide — RDD2022 fine-tune

Verified end to end against the real dataset (Oct 2026). Every byte range and file
size below was read out of the archive's ZIP64 central directory, and the ranges
were confirmed to produce valid, independently-unzipable archives.

Run everything **in the notebook**. You do not need the dataset on your laptop.

---

## 0. Why this is only 2 GB, not 13 GB

The figshare archive is **13.26 GB**, but it is not what it appears to be. Reading
its central directory shows it is a **container of seven nested per-country ZIPs**:

| Nested archive | Size | Use |
| --- | --- | --- |
| `China_Drone.zip` | 0.16 GB | excluded — drone imagery, wrong domain |
| `China_MotorBike.zip` | 0.19 GB | excluded — motorcycle imagery |
| `Czech.zip` | 0.26 GB | available (forward-facing) |
| **`India.zip`** | **0.53 GB** | **use** |
| **`Japan.zip`** | **1.07 GB** | **use** |
| `Norway.zip` | **10.61 GB** | available, but 80% of the archive |
| **`United_States.zip`** | **0.44 GB** | **use** |

**Norway alone is 10.61 GB — four fifths of the download — and is not needed.**
India + Japan + United_States is **2.04 GB**, an 85% saving.

Every member is **STORED** (no deflate), which means each member occupies an exact,
contiguous byte range of the container. You can therefore fetch just the ranges you
need and write them straight out as standalone ZIP files.

| Member | HTTP byte range | Size |
| --- | --- | --- |
| India.zip | `609341032-1136050223` | 527 MB |
| Japan.zip | `1136050224-2208649127` | 1,073 MB |
| United_States.zip | `12819836010-13264172022` | 444 MB |

> Ranges are **inclusive at both ends**. Verified: India's range begins with a valid
> `PK\x03\x04` local header naming `RDD2022/India.zip`, and ends with a well-formed
> EOCD declaring 17,377 entries. Each file unzips normally.

If ranged requests are blocked for any reason, fall back to §2b (full download).

---

## 1. Create the notebook

1. **New Notebook** → *Add Input* → skip (none needed; we fetch directly)
2. **Session options** (right sidebar) → **Accelerator: GPU T4 ×2** — both, DDP roughly halves wall clock
3. **Session options** → **Persistence: Variables and Files** — so `/kaggle/working` survives a
   session reset and you do not re-download 2 GB for each of runs r2/r3/r4
4. Confirm **Internet: On**

Measured on Kaggle (Oct 2026): `torch 2.11.0+cu128`, 2 × Tesla T4 at 15.6 GB each,
30 h/week GPU quota, and **only 20 GB on `/kaggle/working`**.

Cell 1 — confirm the environment:

```python
import subprocess, torch, shutil, pathlib
print("torch     :", torch.__version__)
print("cuda      :", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory/1e9:.1f} GB")
print("disk:")
print(subprocess.run(["df","-h","/kaggle/working"],capture_output=True,text=True).stdout)
```

### Disk budget — 20 GB is the binding constraint

| Stage | Cumulative |
| --- | --- |
| 3 nested zips downloaded | 2.04 GB |
| + extracted countries | ~4.1 GB |
| + curated copy (curation copies each image) | **~6.1 GB ← peak** |
| after `rm -rf` of raw data | ~2.0 GB |
| + runs / weights / plots | ~2.5 GB |

Peak is ~6.5 GB of 20 GB. The ranged download is what makes this comfortable — the
full-download fallback peaks near 17.3 GB and leaves almost nothing for training, so
prefer ranged requests if you hit a space error.

---

## 2. Download only what you need

### 2a. Ranged fetch (2.04 GB, recommended)

```python
import subprocess, pathlib

URL = "https://ndownloader.figshare.com/files/38030910"
RANGES = {
    "India.zip":         (609341032,   1136050223),
    "Japan.zip":         (1136050224,  2208649127),
    "United_States.zip": (12819836010, 13264172022),
}
out = pathlib.Path("/kaggle/working/nested"); out.mkdir(parents=True, exist_ok=True)

for name, (start, end) in RANGES.items():
    dest = out / name
    if dest.exists() and dest.stat().st_size == end - start + 1:
        print(f"skip {name} (already have it)"); continue
    print(f"fetching {name}: {end-start+1:,} bytes")
    subprocess.run(
        ["curl", "-sL", "--retry", "5", "--retry-delay", "5",
         "-H", f"Range: bytes={start}-{end}", URL, "-o", str(dest)],
        check=True,
    )
    got = dest.stat().st_size
    assert got == end - start + 1, f"{name}: expected {end-start+1} bytes, got {got}"
    # A STORED member must start with the ZIP local-file-header magic.
    assert dest.open("rb").read(4) == b"PK\x03\x04", f"{name} is not a ZIP"
    print(f"  ok {got:,} bytes, valid ZIP header")
```

The size assertion matters: a wrong range silently yields a truncated file that
fails much later with a confusing unzip error.

### 2b. Full download (fallback, 13.26 GB)

```python
!cd /kaggle/working && wget -q --show-progress -O rdd2022.zip https://ndownloader.figshare.com/files/38030910
!ls -lh /kaggle/working/rdd2022.zip    # must be ~13G; a few KB means the URL is wrong
```

Then extract only the members you need, and delete the container:

```python
import zipfile, pathlib
dest = pathlib.Path("/kaggle/working/RDD2022")
with zipfile.ZipFile("/kaggle/working/rdd2022.zip") as z:
    for m in ("India.zip", "Japan.zip", "United_States.zip"):
        z.extract(f"RDD2022/{m}", dest)
pathlib.Path("/kaggle/working/rdd2022.zip").unlink()   # free 13.26 GB immediately
```

---

## 3. Unpack the countries

```python
import zipfile, pathlib, shutil
base = pathlib.Path("/kaggle/working")
for zpath in sorted((base / "nested").glob("*.zip")):
    with zipfile.ZipFile(zpath) as z:
        z.extractall(base / "RDD2022")
    zpath.unlink()                     # free the nested zip
print(sorted(p.name for p in (base / "RDD2022").iterdir()))
```

You should see `India`, `Japan`, `United_States`. Each contains `train/` and
`test/`, with `train/annotations/xmls/*.xml` — **Pascal VOC XML, not COCO**.

---

## 4. Install and curate

```python
!pip install -q ultralytics
!git clone -q https://github.com/Suryadeepsinh-Jadeja/Vision-based-real-time-pavement-degradation-detection.git
%cd /kaggle/working/Vision-based-real-time-pavement-degradation-detection
```

Inspect first — this reports the annotation format and per-class counts:

```python
!python scripts/prepare_rdd2022.py --src /kaggle/working/RDD2022 --inspect
```

Then curate:

```python
!python scripts/prepare_rdd2022.py \
    --src /kaggle/working/RDD2022 \
    --dst /kaggle/working/rdd2022-curated \
    --countries India Japan United_States \
    --val-fraction 0.2
```

**Read the report.** It prints box counts per class and warns if D10 is thin. Do not
ignore that warning — it predicts the recall you will get.

Free the raw data now that curation is done:

```python
!rm -rf /kaggle/working/RDD2022 /kaggle/working/nested
```

---

## 5. Train

Two T4s via DDP:

```python
from ultralytics import YOLO
import json, pathlib, shutil

MODEL = "yolov8m-seg.pt"      # fall back to "yolov8m.pt" if seg training misbehaves
RUN = "r1"

model = YOLO(MODEL)
model.train(
    data="/kaggle/working/rdd2022-curated/roadscope.yaml",
    epochs=60,
    imgsz=960,
    batch=16,                 # total across both GPUs; divisible by 2 for DDP
    device=[0, 1],
    workers=4,
    project="/kaggle/working/runs",
    name=RUN,
    patience=15,
    cache="disk",             # "ram" is faster but 30 GB is tight next to a 2 GB dataset

    # lighting varies enormously on Indian roads
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,               # horizontal only — roads have no vertical flip
    fliud=0.0,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,           # pothole oversampling
    erasing=0.2,
)
```

**If it OOMs**, halve `batch` to 8. **If segmentation training errors out**, switch
`MODEL` to `yolov8m.pt` — detect-only trains reliably everywhere, and the rebuild can
ship with bounding-box severity until Phase 4 adds masks. Do not let segmentation
block the whole project.

Expected wall clock on 2× T4: **3–5 hours** for 60 epochs on ~24k images.

---

## 6. Evaluate per class — never just mAP

```python
best = f"/kaggle/working/runs/{RUN}/weights/best.pt"
val = YOLO(best).val(
    data="/kaggle/working/rdd2022-curated/roadscope.yaml",
    split="val", imgsz=960, batch=16, device=[0,1], plots=True,
)
names = val.model.names
metrics = {
    "run": RUN,
    "model": MODEL,
    "map50": float(val.box.map50),
    "map50_95": float(val.box.map),
    "per_class": {
        names[i]: {"ap50": float(val.box.ap50[i]),
                   "precision": float(val.box.p[i]),
                   "recall": float(val.box.r[i])}
        for i in range(len(names))
    },
}
print(json.dumps(metrics, indent=2))
pathlib.Path("/kaggle/working/metrics.json").write_text(json.dumps(metrics, indent=2))
```

Gates from the plan:

| Metric | Gate | Note |
| --- | --- | --- |
| Overall mAP50 | ≥ 0.60 | CRDDC'2022 winner scored F1 76.9%, so this is realistic |
| **D40 pothole AP50** | ≥ 0.55 | rarest class; the project's namesake |
| D40 recall | ≥ 0.70 | on your *own* footage, not RDD2022 — see §8 |

**Expect D10 (transverse crack) to be poor.** India has only 68 transverse labels in
the raw dataset. Report per-class AP and state the limitation. Do not hide it behind
an averaged mAP — an examiner will check, and honesty is the stronger position.

---

## 7. Publish the weights

Do **not** commit `best.pt` to git — that is how the repository once reached 402 MB.

```python
import pathlib
out = pathlib.Path("/kaggle/working/output"); out.mkdir(exist_ok=True)
shutil.copy(best, out / "best.pt")
shutil.copy("/kaggle/working/metrics.json", out / "metrics.json")
print([f"{p.name}: {p.stat().st_size/1e6:.1f} MB" for p in out.iterdir()])
```

Then in the Kaggle UI: **Save Version → Save & Run All** (or *Save Output*). The
output files become a downloadable dataset artifact.

To get them into this repository, either:

- download `best.pt` from the Kaggle output and drop it in `models/`, or
- create a GitHub Release and use `scripts/download_weights.sh`

Then set `ROADSCOPE_MODEL_NAME=best.pt` in `.env` (it is already first in the
candidate list).

---

## 8. The number that actually matters

RDD2022 mAP will **not** predict performance on your roads. Capture your own
dashcam video and hand-label at least **150 frames**. That in-domain set is the
figure to quote in a viva, and the only one that is actually about your system.

Capture on the laptop, not the notebook:

```bash
python -m roadscope.agent --camera-url http://<phone-ip>:8080/stream --seconds 300
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `assert ... expected N bytes, got M` | Range served partially, or proxy stripped `Range` | Retry that one file; if it persists use §2b |
| `Destination file already exists` / not a ZIP | Wrong byte range | Re-check against §0 table |
| unzip: "End-of-central-directory signature not found" | You downloaded HTML, or a truncated file | Verify magic bytes are `PK\x03\x04` |
| `No labels found in ...cache` | Paths in `roadscope.yaml` wrong | `--inspect`, then re-run curation |
| CUDA out of memory | batch too high | `batch=8` |
| DDP hang / NCCL error | GPU count mismatch | `device=[0,1]` requires exactly 2 visible GPUs |
| mAP50 far below 0.60 | China data included, or class imbalance | China excluded by default; check the curation report |
| Session disconnect mid-training | Free-tier timeout | Train fewer epochs per version and resume from `last.pt` |
| Disk full | Raw data not deleted | `rm -rf /kaggle/working/RDD2022` after curation |

---

## Expected timeline

| Run | Purpose |
| --- | --- |
| r1 | Baseline. Reveals which class collapses. |
| r2 | Imbalance fixes driven by r1's per-class AP. |
| r3 | Best curated mix. Final. |

**Budget four runs.** Pothole is the rarest class, and Phase 3 has zero schedule
float — the reason to start now rather than after the code work.