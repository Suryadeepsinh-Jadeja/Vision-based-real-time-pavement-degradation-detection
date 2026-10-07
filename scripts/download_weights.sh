#!/usr/bin/env bash
# Fetch trained detector weights from a GitHub Release.
#
# Weights are deliberately NOT committed to git: that is how the repository
# reached 402 MB. This script caches them into models/.
set -euo pipefail

REPO="${ROADSCOPE_REPO:-Suryadeepsinh-Jadeja/Vision-based-real-time-pavement-degradation-detection}"
TAG="${ROADSCOPE_RELEASE_TAG:-v0.1.0}"
ASSET="${ROADSCOPE_RELEASE_ASSET:-best.pt}"
DEST_NAME="${ROADSCOPE_MODEL_NAME:-roadscope-yolov8m-seg.pt}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/models"
DEST="$MODELS_DIR/$DEST_NAME"

if [[ -f "$DEST" ]]; then
    echo "Weights already present: $DEST"
    echo "Delete the file to force a re-download."
    exit 0
fi

mkdir -p "$MODELS_DIR"

URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"

echo "Downloading weights"
echo "  release : $TAG"
echo "  asset   : $ASSET"
echo "  url     : $URL"

if command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar "$URL" -o "$DEST.tmp"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$DEST.tmp" "$URL"
else
    echo "ERROR: neither curl nor wget is available." >&2
    exit 1
fi

if [[ ! -s "$DEST.tmp" ]]; then
    echo "ERROR: downloaded file is empty." >&2
    rm -f "$DEST.tmp"
    exit 1
fi

mv "$DEST.tmp" "$DEST"
echo "Saved to $DEST"

echo
echo "Note: no Release has been published yet for tag '$TAG'."
echo "Until then, the legacy single-class weights in models/ are used by"
echo "Phase 2's vertical slice only. See REBUILD_PLAN.md section 5."