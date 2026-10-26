"""Detector wrapper.

The legacy prototype constructed ``RoadDamageDetector()`` twice per Streamlit
rerun (``app/dashboard.py:163`` and ``:194``), loading the 22 MB weights twice.
This module guarantees a single process-wide instance and loads weights lazily,
so importing RoadScope never pays the model-load cost.

Engine selection is explicit. If the requested weights are missing the detector
reports itself unavailable; it never silently substitutes a different engine,
which is how the old system appeared to work while running a weaker one.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import paths
from ..logging_setup import get_logger

log = get_logger(__name__)


def _torch():
    """Import torch lazily.

    Importing it eagerly costs ~1.5 s at ``import roadscope``, and torch is only
    needed once the detector actually loads weights. ``ultralytics`` pulls it in
    anyway, so this is purely to keep module import cheap and optional.
    """
    try:
        import torch
    except ImportError:
        return None
    return torch


CLASS_CODES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}


def _is_auto(device: object) -> bool:
    """True when a device setting means "choose for me".

    ``Settings.device`` defaults to the string ``"auto"``. Ultralytics treats
    every non-empty device string as a device name and raises
    ``Invalid CUDA 'device=auto' requested``, so the sentinel has to be
    translated to ``None`` at the boundary.
    """
    return device is None or not str(device).strip() or str(device).strip().lower() == "auto"
RDD_CLASS_NAME_MAP = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}


@dataclass(slots=True)
class Detection:
    """One detected distress in a single frame."""

    class_code: str
    confidence: float
    bbox: tuple[int, int, int, int]
    mask: Any = None  # np.ndarray | None; present for segmentation models
    class_id: int = -1

    @property
    def label(self) -> str:
        return f"{self.class_code} - {RDD_CLASS_NAME_MAP.get(self.class_code, 'Distress')}"


class DetectorUnavailable(RuntimeError):
    """Raised when no usable weights file can be found."""


class DefectDetector:
    """Single-instance YOLO wrapper."""

    _instance: DefectDetector | None = None
    _lock = threading.Lock()
    load_count = 0

    def __init__(self, model_path: Path | None = None, device: str | None = None) -> None:
        self.model_path = model_path
        #: Resolved on first use by :meth:`resolve_device`; None means "auto".
        #: The literal string "auto" is a settings value, not a torch device --
        #: handing it to ultralytics raises "Invalid CUDA 'device=auto'".
        self.device: str | None = None if _is_auto(device) else device
        self.model: Any = None
        self.names: dict[int, str] = {}
        self.engine_name = "unavailable"
        type(self).load_count += 1
        log.debug("Detector instantiated (load_count=%d)", type(self).load_count)

    @classmethod
    def instance(
        cls, model_path: Path | str | None = None, device: str | None = None
    ) -> DefectDetector:
        """Return the process-wide detector, constructing it on first call.

        ``model_path`` accepts a string as well as a :class:`~pathlib.Path`.
        A str previously reached ``_load`` unchanged and raised
        ``AttributeError: 'str' object has no attribute 'name'`` at the first
        ``self.model_path.name``. Every in-repo caller passed a Path, so the
        mismatch only surfaced for a caller passing a plain string.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(Path(model_path) if model_path else None, device)
                    cls._instance._load()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton. For tests."""
        with cls._lock:
            cls._instance = None
            cls.load_count = 0

    def _load(self) -> None:
        """Load weights, if available. Never raises for a missing file."""
        if self.model_path is None:
            self.model_path = paths.resolve_model()
        if self.model_path is None:
            log.warning(
                "No weights found under %s. Run scripts/download_weights.sh, or set "
                "ROADSCOPE_MODEL_PATH. Detector is unavailable.",
                paths.models_dir(),
            )
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            log.warning("ultralytics not installed. Install the 'yolo' extra.")
            return

        log.info("Loading weights from %s", self.model_path)
        self.model = YOLO(str(self.model_path))
        raw_names = getattr(self.model, "names", None) or {}
        # Normalise ultralytics' several possible name formats to {int: str}.
        self.names = {
            (int(k) if str(k).isdigit() else i): str(v)
            for i, (k, v) in enumerate(raw_names.items())
        }
        self.engine_name = f"YOLO ({self.model_path.name}, {len(self.names)} classes)"
        log.info("Detector ready: %s", self.engine_name)

    @property
    def available(self) -> bool:
        return self.model is not None

    def resolve_class(self, class_id: int) -> str:
        """Map a model class index to an RDD2022 code.

        Uses the model's own names. The legacy implementation guessed via
        substring matching and then overrode itself, which relabelled potholes
        as longitudinal cracks.
        """
        name = self.names.get(class_id, "").lower().replace(" ", "_")
        for token, code in (
            ("longitudinal", "D00"),
            ("transverse", "D10"),
            ("alligator", "D20"),
            ("fatigue", "D20"),
            ("pothole", "D40"),
        ):
            if token in name:
                return code
        if name in {"d00", "d10", "d20", "d40"}:
            return name.upper()
        # Single-class models (legacy weights) only ever detect potholes.
        if len(self.names) == 1:
            return "D40"
        return CLASS_CODES.get(class_id, "D40")

    def resolve_device(self, requested: str | None = None) -> str:
        """Pick the torch device, falling back when the request is unusable.

        ``Settings.device`` used to be ignored entirely -- it was never passed
        to ultralytics, which auto-selects and lands on CPU even on Apple
        Silicon where MPS is available and 2-3x faster. See
        ``scripts/benchmark_detector.py`` for the measurements.

        Silently downgrading a device the caller asked for would be worse than
        the original bug, so the fallback is logged.
        """
        torch = _torch()
        if torch is None:
            return "cpu"

        if requested and not _is_auto(requested):
            candidate = str(requested).lower()
        elif self.device is not None:
            return self.device
        elif torch.backends.mps.is_available():
            candidate = "mps"
        elif torch.cuda.is_available():
            candidate = "0"
        else:
            candidate = "cpu"

        try:
            torch.empty(0, device=candidate)
        except Exception as exc:
            log.warning(
                "device %r is unusable (%s); falling back to cpu. Set ROADSCOPE_DEVICE to "
                "override.",
                candidate,
                exc,
            )
            candidate = "cpu"

        self.device = candidate
        log.info("detector device: %s", candidate)
        return candidate

    def infer(
        self,
        frame: Any,
        conf: float = 0.35,
        iou: float = 0.50,
        imgsz: int = 960,
    ) -> list[Detection]:
        """Run inference on a BGR frame."""
        if self.model is None:
            raise DetectorUnavailable(
                "Detector has no weights loaded. Fetch weights or set ROADSCOPE_MODEL_PATH."
            )
        results = self.model(
            frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            device=self.resolve_device(),
        )
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            masks = getattr(result, "masks", None)
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                detections.append(
                    Detection(
                        class_code=self.resolve_class(cls_id),
                        confidence=float(box.conf[0]),
                        bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        mask=None if masks is None else masks.data[i].cpu().numpy(),
                        class_id=cls_id,
                    )
                )
        return detections
