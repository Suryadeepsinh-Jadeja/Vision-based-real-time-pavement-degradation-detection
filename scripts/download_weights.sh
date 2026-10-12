#!/usr/bin/env bash
# Fetch the trained detector weights.
#
# Weights are deliberately NOT committed to git: that is how the repository
# reached 402 MB. This script places them in models/, which .gitignore excludes.
#
# Current weights
# ---------------
#   rdd2022-yolov8m-r1.pt   YOLOv8m fine-tuned on RDD2022, 4 classes, 52 MB
#                           mAP50 0.609, D00 0.651 / D10 0.558 / D20 0.688 /
#                           D40 0.538.  See docs/results.md for the full
#                           per-class table and the pothole gate analysis.
#
#   Source: a Kaggle notebook that trains run `r1`. Reproduce it with
#   scripts/prepare_rdd2022.py then the cells in scripts/train_kaggle.md.
#   Weights are not on GitHub Releases -- there is no release to fetch from yet.
#   See the two options below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/models"
DEST="$MODELS_DIR/rdd2022-yolov8m-r1.pt"

EXPECTED_BYTES=52099026
EXPECTED_CLASSES=('D00' 'D10' 'D20' 'D40')

if [[ "${1:-}" == "--verify" ]]; then
    [[ -f "$DEST" ]] || { echo "Missing: $DEST" >&2; exit 1; }
    python3 "$ROOT/scripts/check_model.py" "$DEST"
    exit $?
fi

# --- Already present? -------------------------------------------------------
if [[ -f "$DEST" ]]; then
    echo "Weights already present: $DEST"
    echo "Delete the file to force a re-download."
    exit 0
fi

mkdir -p "$MODELS_DIR"

# --- Two ways to obtain the file --------------------------------------------
cat <<'EOF'
The weights are not published anywhere automated yet. Two ways to get them:

  1. Kaggle Output tab (one-off, ~1 minute)
     - Run scripts/train_kaggle.md, or open your existing run
     - Cell: copy runs/r1/weights/best.pt to /kaggle/working/output/
     - Save Version, then Output tab -> download r1-best.pt
     - Move it here, named rdd2022-yolov8m-r1.pt

  2. Kaggle CLI (repeatable)
     - Settings -> API -> Create New Token -> ~/.kaggle/kaggle.json (chmod 600)
     - kaggle kernels output <user>/<notebook-slug> -p models/ --force
     - Rename the downloaded best.pt to rdd2022-yolov8m-r1.pt

Once a GitHub Release carries the asset, this script can download it directly
by setting ROADSCOPE_RELEASE_TAG and ROADSCOPE_RELEASE_ASSET.

EOF
echo "Nothing downloaded. Place the file at:"
echo "  $DEST"
echo
echo "Then verify it:"
echo "  bash scripts/download_weights.sh --verify"
echo "  source venv/bin/activate && python -m roadscope.agent --check  # 4 classes"
