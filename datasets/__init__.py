"""Dataset của LoRA-Talk, được nạp theo stage cần dùng."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "LipLandmarkDataset": ("datasets.landmark_dataset", "LipLandmarkDataset"),
    "RendererFrameDataset": ("datasets.renderer_dataset", "RendererFrameDataset"),
    "FrameSample": ("datasets.renderer_dataset", "FrameSample"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Nạp dataset tương ứng khi được truy cập lần đầu."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} không có thuộc tính {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
