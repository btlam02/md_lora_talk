"""Các hàm cấu hình PEFT/LoRA cho Landmark Generator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model


DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "linear1", "linear2"]


def reset_lora_weights(model: nn.Module) -> None:
    """Khởi tạo lại adapter LoRA: A ngẫu nhiên, B bằng 0 để không đổi base model ban đầu."""
    for module in model.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for adapter in module.lora_A.keys():
                nn.init.kaiming_uniform_(module.lora_A[adapter].weight, a=math.sqrt(5))
                nn.init.zeros_(module.lora_B[adapter].weight)



def set_extra_modules_trainable(model: nn.Module, extra_modules: Optional[List[str]] = None) -> Dict[str, int]:
    """Mở gradient cho các module ngoài LoRA được yêu cầu train thêm."""
    if not extra_modules:
        return {}
    counts = {}
    for mod_name in extra_modules:
        count = 0
        for parameter_name, param in model.named_parameters():
            if mod_name in parameter_name:
                param.requires_grad = True
                count += param.numel()
        counts[mod_name] = count
    return counts


def _get_split_transformer_layer_ids(model: nn.Module) -> List[int]:
    """Trả về chỉ số các fusion layer có trong model, kể cả khi được PEFT bọc."""
    prefix = "fusion_transformer.layers."
    layer_ids = set()
    for parameter_name, _ in model.named_parameters():
        if prefix not in parameter_name:
            continue
        suffix = parameter_name.split(prefix, 1)[1]
        layer_id, _, _ = suffix.partition(".")
        if layer_id.isdigit():
            layer_ids.add(int(layer_id))
    return sorted(layer_ids)


def unfreeze_transformer_layers(model: nn.Module, n_layers: int = 0) -> int:
    """Mở gradient cho N fusion layer cuối, không giả định model luôn có 4 layer."""
    if n_layers <= 0:
        return 0

    layer_ids = _get_split_transformer_layer_ids(model)
    selected_ids = set(layer_ids[-n_layers:])
    count = 0
    prefix = "fusion_transformer.layers."
    for parameter_name, param in model.named_parameters():
        if prefix not in parameter_name:
            continue
        layer_id, _, _ = parameter_name.split(prefix, 1)[1].partition(".")
        if layer_id.isdigit() and int(layer_id) in selected_ids:
            param.requires_grad = True
            count += param.numel()
    return count


def build_lora_landmark_model(
    base_model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    resume_lora_dir: Optional[Path] = None,
    extra_train_modules: Optional[List[str]] = None,
    unfreeze_layers: int = 0,
) -> nn.Module:
    """Bọc base model bằng PEFT LoRA và cấu hình các tham số được phép cập nhật."""
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    if resume_lora_dir is not None and resume_lora_dir.exists():
        # Khi resume, PEFT đọc cấu hình adapter từ thư mục đã lưu.
        model = PeftModel.from_pretrained(base_model, str(resume_lora_dir), is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            modules_to_save=extra_train_modules,
        )
        model = get_peft_model(base_model, peft_config)

    set_extra_modules_trainable(model, extra_train_modules)
    unfreeze_transformer_layers(model, unfreeze_layers)
    return model
