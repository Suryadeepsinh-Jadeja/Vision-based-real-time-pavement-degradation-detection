#!/usr/bin/env python3
"""Launch the RoadScope dashboard.

    streamlit run scripts/dashboard.py

Streamlit executes the target file as a top-level script with no package
context, so pointing it at ``src/roadscope/ui/dashboard.py`` directly fails with
``ImportError: attempted relative import with no known parent package``. This
launcher puts ``src`` on ``sys.path`` and imports the module absolutely, which
also means a fresh checkout works without ``pip install -e .``.

Reads ``ROADSCOPE_SERVER_DB`` to choose the survey store, so a demo database
can be shown without touching real survey data::

    ROADSCOPE_SERVER_DB=var/demo.sqlite streamlit run scripts/dashboard.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from roadscope.ui.dashboard import main  # noqa: E402

main()
