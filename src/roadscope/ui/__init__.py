"""RoadScope ui subsystem.

The dashboard is launched with ``streamlit run scripts/dashboard.py``.

It is not launched as ``streamlit run -m roadscope.ui.dashboard``: Streamlit
executes the file it is given as a top-level script with no package context, so
relative imports inside the module raise
``ImportError: attempted relative import with no known parent package``. The
launcher script imports it absolutely after putting ``src`` on ``sys.path``.
"""
