"""Nạp và chuyển đổi checkpoint giữa attention gộp của IP-LAP và attention tách LoRA."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn


def normalize_checkpoint_state(ckpt_or_state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Lấy state dict và bỏ đúng tiền tố ``module.`` do DataParallel tạo ra."""
    state_dict = ckpt_or_state.get("state_dict", ckpt_or_state)
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_pretrained_into_split_model(model: nn.Module, ckpt_path: Path) -> Tuple[List[str], List[str]]:
    """Nạp checkpoint IP-LAP và tách ``in_proj`` thành Q/K/V cho model LoRA-ready."""
    state_dict = normalize_checkpoint_state(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    new_sd = {}
    old_pre = "fusion_transformer.transformer_encoder.layers."
    new_pre = "fusion_transformer.layers."
    for key, value in state_dict.items():
        if not key.startswith(old_pre):
            new_sd[key] = value
            continue
        rest = key[len(old_pre) :]
        layer_idx, sub = rest.split(".", 1)
        if sub == "self_attn.in_proj_weight":
            qw, kw, vw = value.chunk(3, dim=0)
            new_sd[f"{new_pre}{layer_idx}.self_attn.q_proj.weight"] = qw
            new_sd[f"{new_pre}{layer_idx}.self_attn.k_proj.weight"] = kw
            new_sd[f"{new_pre}{layer_idx}.self_attn.v_proj.weight"] = vw
        elif sub == "self_attn.in_proj_bias":
            qb, kb, vb = value.chunk(3, dim=0)
            new_sd[f"{new_pre}{layer_idx}.self_attn.q_proj.bias"] = qb
            new_sd[f"{new_pre}{layer_idx}.self_attn.k_proj.bias"] = kb
            new_sd[f"{new_pre}{layer_idx}.self_attn.v_proj.bias"] = vb
        elif sub.startswith("self_attn.out_proj."):
            new_sd[f"{new_pre}{layer_idx}.{sub}"] = value
        elif sub in {
            "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias",
            "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
        }:
            new_sd[f"{new_pre}{layer_idx}.{sub}"] = value
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    return missing, unexpected


def split_state_to_iplap_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Đổi state dict attention tách về định dạng checkpoint IP-LAP gốc.

    Inference IP-LAP cần ``self_attn.in_proj_*`` trong khi LoRA huấn luyện trên
    các projection Q/K/V riêng lẻ, nên ba tensor phải được ghép lại theo thứ tự Q-K-V.
    """
    converted: Dict[str, torch.Tensor] = {}
    consumed = set()
    layer_prefix = "fusion_transformer.layers."
    old_prefix = "fusion_transformer.transformer_encoder.layers."

    layer_ids = sorted(
        {
            int(key[len(layer_prefix) :].split(".", 1)[0])
            for key in state_dict
            if key.startswith(layer_prefix) and key[len(layer_prefix) :].split(".", 1)[0].isdigit()
        }
    )
    for layer_idx in layer_ids:
        src = f"{layer_prefix}{layer_idx}."
        dst = f"{old_prefix}{layer_idx}."
        q_w = state_dict.get(src + "self_attn.q_proj.weight")
        k_w = state_dict.get(src + "self_attn.k_proj.weight")
        v_w = state_dict.get(src + "self_attn.v_proj.weight")
        if q_w is not None and k_w is not None and v_w is not None:
            converted[dst + "self_attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
            consumed.update(
                {
                    src + "self_attn.q_proj.weight",
                    src + "self_attn.k_proj.weight",
                    src + "self_attn.v_proj.weight",
                }
            )
        q_b = state_dict.get(src + "self_attn.q_proj.bias")
        k_b = state_dict.get(src + "self_attn.k_proj.bias")
        v_b = state_dict.get(src + "self_attn.v_proj.bias")
        if q_b is not None and k_b is not None and v_b is not None:
            converted[dst + "self_attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)
            consumed.update(
                {
                    src + "self_attn.q_proj.bias",
                    src + "self_attn.k_proj.bias",
                    src + "self_attn.v_proj.bias",
                }
            )

    for key, value in state_dict.items():
        if key in consumed:
            continue
        if key.startswith(layer_prefix):
            rest = key[len(layer_prefix) :]
            layer_idx, sub = rest.split(".", 1)
            converted[f"{old_prefix}{layer_idx}.{sub}"] = value
        else:
            converted[key] = value
    return converted
