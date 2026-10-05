"""Các tiện ích dùng chung của LoRA-Talk.

Tránh import toàn bộ tiện ích ngay từ đầu: training Landmark không cần OpenCV,
và training Renderer không cần tải các hàm audio/landmark không liên quan.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "FACEMESH_LIPS": ("utils.landmark", "FACEMESH_LIPS"),
    "FACEMESH_CONNECTION": ("utils.landmark", "FACEMESH_CONNECTION"),
    "ALL_LANDMARK_IDX": ("utils.landmark", "ALL_LANDMARK_IDX"),
    "POSE_LANDMARK_IDX": ("utils.landmark", "POSE_LANDMARK_IDX"),
    "CONTENT_LANDMARK_IDX": ("utils.landmark", "CONTENT_LANDMARK_IDX"),
    "ORI_SEQUENCE_IDX": ("utils.landmark", "ORI_SEQUENCE_IDX"),
    "ORDER": ("utils.landmark", "ORDER"),
    "POSE_SEQUENCE_IDX": ("utils.landmark", "POSE_SEQUENCE_IDX"),
    "CONTENT_SEQUENCE_IDX": ("utils.landmark", "CONTENT_SEQUENCE_IDX"),
    "MOUTH_SLICE": ("utils.landmark", "MOUTH_SLICE"),
    "MOUTH_PRIOR_SLICE": ("utils.landmark", "MOUTH_PRIOR_SLICE"),
    "LOWER_SLICE": ("utils.landmark", "LOWER_SLICE"),
    "summarize_landmarks": ("utils.landmark", "summarize_landmarks"),
    "extract_f0_from_wav": ("utils.audio", "extract_f0_from_wav"),
    "normalize_checkpoint_state": ("utils.checkpoint", "normalize_checkpoint_state"),
    "load_pretrained_into_split_model": ("utils.checkpoint", "load_pretrained_into_split_model"),
    "split_state_to_iplap_state_dict": ("utils.checkpoint", "split_state_to_iplap_state_dict"),
    "env_value": ("utils.env", "env_value"),
    "read_rgb_image": ("utils.image", "read_rgb_image"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Nạp tiện ích theo yêu cầu để các dependency độc lập với nhau."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} không có thuộc tính {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
