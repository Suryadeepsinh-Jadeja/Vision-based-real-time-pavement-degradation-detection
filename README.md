# RoadScope — Vision-Based Pavement Degradation Detection & GIS System

> ## ⚠️ Status: Under rebuild
>
> The prototype that used to live in this repository has been **deleted**. It did not work:
> the detector fabricated its own class labels, GPS was a hand-written CSV, there was no live
> video path, and nothing was ever persisted or transmitted.
>
> The replacement is specified in **[`REBUILD_PLAN.md`](REBUILD_PLAN.md)** — 7 gated phases
> across 3 weeks. This repository is currently at **Phase 0** (start of work).
>
> The old code is recoverable at commit `a4008379` or tag `pre-rebuild-cleanup`.
> Domain content worth keeping was salvaged to [`docs/legacy_reference.md`](docs/legacy_reference.md).

## Do not run anything yet

There is no runnable application in this repository right now. `run.sh`, `app/`, and the
`requirements.txt` files have been removed. Dependencies will be declared in `pyproject.toml`
during Phase 1.

## Current contents

| Path | Purpose |
| --- | --- |
| [`REBUILD_PLAN.md`](REBUILD_PLAN.md) | The rebuild specification — phases, gates, schema, tests, schedule |
| [`docs/legacy_reference.md`](docs/legacy_reference.md) | Civil/GIS domain content salvaged from the deleted prototype |
| `tests/fixtures/` | Sample GPS track and defect set, kept as test fixtures (not demo data) |
| `models/pothole_yolov8.pt` | Legacy single-class weights. Used only by Phase 2's vertical slice and as a fallback; replaced by the RDD2022 model in Phase 3 |
| `new.md` | Original system specification |
| `Review_2_Presentation.md` | Presentation outline and speaker notes |
| `VIVA_DEFENSE_GUIDE.md` | Review and viva preparation material |
| `PRESENTATION_CONTENT_GUIDE.md` | Presentation content guide |
| `smart ppt.pdf` | Earlier presentation |

## Team

| Member | Reg. no. | Pillar |
| --- | --- | --- |
| Jadeja Suryadeepsinh Bharatsinh | 23BEL1027 | System lead / integration — Phase 2, 5 |
| Vishnu Nair | 23BEL1028 | Computer science / AI — Phase 3 |
| Udayvardhan Singh Rathore | 23BEL1018 | Civil engineering / ASTM — Phase 3, 6 |
| Ainessh Kumar S | 23BEL1039 | Geospatial & GIS — Phase 4 |
| Shreyan Biswas | 23BEL1052 | Urban governance / UI — Phase 1, 6, 7 |

## Prerequisites

- Git
- Python 3.11 (validated; other versions untested with the CV/YOLO dependencies)
- Docker, for the local MQTT broker

`venv/` is retained from the old setup so the 1.4 GB of PyTorch/Ultralytics wheels are not
re-downloaded. It will be replaced by a clean environment during Phase 1.

---

*The previous README documented the deleted prototype. Recover it with
`git show a4008379:README.md`.*