"""Các model của LoRA-Talk.

Module này dùng lazy import để Stage 1 Landmark không vô tình phụ thuộc vào
torchvision hay các thành phần của renderer ở Stage 2.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "build_split_landmark_generator": ("models.landmark_generator", "build_split_landmark_generator"),
    "PAARenderer": ("models.renderer", "PAARenderer"),
    "build_renderer_discriminator": ("models.discriminator", "build_renderer_discriminator"),
    "build_lora_landmark_model": ("models.peft_lora", "build_lora_landmark_model"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Nạp đúng thành phần khi nó được truy cập lần đầu."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} không có thuộc tính {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
