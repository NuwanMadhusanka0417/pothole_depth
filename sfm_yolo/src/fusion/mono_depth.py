"""Monocular metric depth backend for Path B (DepthAnything V2).

Wraps a pretrained **metric** monocular depth model so the RGB+IMU pipeline can
obtain a per-pixel metric depth map from a single frame. The model is loaded
lazily; if the optional dependencies are missing we raise a clear, actionable
error instead of failing deep inside the pipeline.

Install (into the `road` env)::

    pip install transformers timm

The default checkpoint is the outdoor **metric** DepthAnything V2 model
(``depth-anything/Depth-Anything-V2-metric-vkitti-small-hf``). Metric models
return depth directly in metres, avoiding monocular scale ambiguity.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils.logging_utils import get_logger

_logger = get_logger("fusion.mono_depth")

_DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-metric-vkitti-small-hf"

_INSTALL_HINT = (
    "Monocular metric depth (Path B) needs 'transformers' and 'timm'. "
    "Install them into the road env:\n"
    "    C:\\Users\\22390013@students.ltu.edu.au\\.conda\\envs\\road\\python.exe "
    "-m pip install transformers timm"
)


def is_available() -> bool:
    """True if the optional depth-model dependencies are importable."""
    import importlib.util as u
    return all(u.find_spec(m) is not None for m in ("transformers", "torch"))


class MetricDepthModel:
    """Lazy wrapper around a HuggingFace metric depth-estimation pipeline."""

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device
        self._pipe = None

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(_INSTALL_HINT) from exc

        import torch
        device = self.device
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        _logger.info("Loading metric depth model %s (device=%s)", self.model_name, device)
        from transformers import pipeline
        self._pipe = pipeline("depth-estimation", model=self.model_name, device=device)

    def infer(self, rgb_bgr: np.ndarray) -> np.ndarray:
        """Return an (H, W) float32 **metric** depth map (metres) for a BGR frame."""
        self._ensure_loaded()
        from PIL import Image

        rgb = rgb_bgr[:, :, ::-1]  # BGR (OpenCV) -> RGB
        out = self._pipe(Image.fromarray(np.ascontiguousarray(rgb)))
        # HF returns {'predicted_depth': tensor, 'depth': PIL}. Prefer the tensor.
        pred = out.get("predicted_depth")
        if pred is not None:
            depth = pred.squeeze().detach().cpu().numpy().astype(np.float32)
        else:  # pragma: no cover - fallback
            depth = np.asarray(out["depth"], dtype=np.float32)
        # Resize to the input frame if the model returned a different size.
        h, w = rgb_bgr.shape[:2]
        if depth.shape[:2] != (h, w):
            import cv2
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth
