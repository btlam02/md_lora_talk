"""Discriminator GAN của IP-LAP dùng trong Stage 2 PAA Renderer."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
import torch
import torch.nn as nn

from utils.env import env_path

IP_LAP_PATH = env_path("LORATALK_IP_LAP_DIR", Path(__file__).resolve().parents[2] / "third_party" / "IP_LAP")
if str(IP_LAP_PATH) not in sys.path:
    sys.path.insert(0, str(IP_LAP_PATH))


def _load_iplap_define_d():
    """Nạp factory PatchGAN đa tỉ lệ từ source IP-LAP."""
    module_path = IP_LAP_PATH / "models" / "pix2pixHD_disc.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy discriminator IP-LAP: {module_path}")
    spec = importlib.util.spec_from_file_location("_lora_talk_iplap_pix2pixhd_disc", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load IP-LAP discriminator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.define_D


define_D = _load_iplap_define_d()


def build_renderer_discriminator(device: torch.device | str = "cuda") -> nn.Module:
    """Tạo PatchGAN đa tỉ lệ, nhận cặp ảnh thật/giả ghép theo channel."""
    netD = define_D(
        input_nc=6,
        ndf=64,
        n_layers_D=3,
        norm="instance",
        use_sigmoid=False,
        num_D=2,
        getIntermFeat=True,
    )
    return netD.to(device)
