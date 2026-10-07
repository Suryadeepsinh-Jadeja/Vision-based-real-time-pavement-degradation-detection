"""Structured logging setup.

A failed demo should be diagnosable after the fact, so logs go to both the
console and a rotating file, with a consistent format.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_CONFIGURED = False

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)-24s %(filename)s:%(lineno)d %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    force: bool = False,
) -> None:
    """Configure root logging. Idempotent unless ``force`` is set."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        rotating.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(rotating)

    # Ultralytics and other libraries are extremely chatty at INFO.
    for noisy in ("ultralytics", "PIL", "matplotlib", "urllib3", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    setup_logging()
    return logging.getLogger(name)
