"""Các tiện ích đọc và chuyển đổi ảnh."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def read_rgb_image(path: Path) -> np.ndarray:
    """Đọc ảnh bằng OpenCV và trả về mảng RGB uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
