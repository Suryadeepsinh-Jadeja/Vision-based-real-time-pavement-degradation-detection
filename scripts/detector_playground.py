#!/usr/bin/env python3
"""Launch the detector playground.

    streamlit run scripts/detector_playground.py

Upload an image or a video and see what the RDD2022 detector finds, frame by
frame for video. No GPS, no broker and no database required.

Streamlit runs the target as a top-level script with no package context, so
this launcher puts ``src`` on ``sys.path`` and imports absolutely. A fresh
checkout works without ``pip install -e .``.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from roadscope.ui.detector_playground import main  # noqa: E402

main()
