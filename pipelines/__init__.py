"""Các pipeline huấn luyện của LoRA-Talk, được nạp theo yêu cầu."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "LANDMARK_DEFAULTS": ("pipelines.trainer_landmark", "LANDMARK_DEFAULTS"),
    "LandmarkTrainer": ("pipelines.trainer_landmark", "LandmarkTrainer"),
    "apply_landmark_default_args": ("pipelines.trainer_landmark", "apply_landmark_default_args"),
    "PAARendererTrainer": ("pipelines.trainer_paa", "PAARendererTrainer"),
    "PAA_ALIGNMENT_LITE_DEFAULTS": ("pipelines.trainer_paa", "PAA_ALIGNMENT_LITE_DEFAULTS"),
    "PAA_RENDERER_DEFAULTS": ("pipelines.trainer_paa", "PAA_RENDERER_DEFAULTS"),
    "apply_default_args": ("pipelines.trainer_paa", "apply_default_args"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Chỉ import pipeline được sử dụng để tránh dependency chéo giữa hai stage."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} không có thuộc tính {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
