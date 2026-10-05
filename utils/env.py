"""Các hàm đọc cấu hình đường dẫn từ args, biến môi trường và file ``.env``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent
_ENV_LOADED = False
_ENV_DIR: Optional[Path] = None


def _strip_quotes(value: str) -> str:
    """Bỏ cặp dấu nháy bao quanh giá trị trong file ``.env`` nếu có."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(env_path: Optional[Path] = None) -> None:
    """
    Đọc cặp key/value từ ``.env`` mà không cần thư viện bên ngoài.

    Biến môi trường hiện có luôn được ưu tiên hơn giá trị trong file.
    """
    global _ENV_LOADED, _ENV_DIR
    if _ENV_LOADED and env_path is None:
        return

    candidates = [Path(env_path)] if env_path is not None else [PACKAGE_DIR / ".env", PROJECT_DIR / ".env"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        _ENV_DIR = candidate.parent
        for line in candidate.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = _strip_quotes(value.strip())
            if key and key not in os.environ:
                os.environ[key] = value
        break

    if env_path is None:
        _ENV_LOADED = True


def env_path(name: str, default: Optional[str | Path] = None) -> Optional[Path]:
    """Lấy biến môi trường dưới dạng ``Path``, hỗ trợ đường dẫn tương đối với file .env."""
    load_env_file()
    value = os.getenv(name)
    if value is not None and value != "":
        path = Path(value).expanduser()
        if not path.is_absolute() and _ENV_DIR is not None:
            path = _ENV_DIR / path
        return path
    if default is None:
        return None
    return Path(default).expanduser()


def env_value(name: str, default: Optional[str] = None) -> Optional[str]:
    """Lấy chuỗi cấu hình từ môi trường, dùng default khi biến chưa được đặt."""
    load_env_file()
    value = os.getenv(name)
    if value is not None and value != "":
        return value
    return default


def arg_or_env_path(args: Any, attr: str, env_name: str, default: Optional[str | Path] = None) -> Optional[Path]:
    """Ưu tiên ``args.<attr>``, sau đó biến môi trường, cuối cùng là giá trị mặc định."""
    value = getattr(args, attr, None)
    if value is not None and str(value) != "":
        return Path(value).expanduser()
    return env_path(env_name, default)


def set_arg_path_from_env(args: Any, attr: str, env_name: str, default: Optional[str | Path] = None) -> Optional[Path]:
    """Lấy đường dẫn theo thứ tự ưu tiên và ghi ngược lại vào namespace args."""
    path = arg_or_env_path(args, attr, env_name, default)
    if path is not None:
        setattr(args, attr, path)
    return path


def require_path(path: Optional[Path], description: str, env_name: str) -> Path:
    """Bắt buộc có đường dẫn; ném lỗi kèm tên biến môi trường cần cấu hình."""
    if path is None:
        raise ValueError(f"Missing {description}. Set args value or {env_name} in .env.")
    return path
