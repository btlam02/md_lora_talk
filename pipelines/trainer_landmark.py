"""Pipeline huấn luyện LoRA cho Landmark Generator của IP-LAP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from peft import PeftModel
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.landmark_dataset import LipLandmarkDataset
from losses.landmark_losses import (
    l1_loss_landmark,
    lmd_loss,
    mouth_aperture_profile_loss,
    mouth_open_delta_loss,
    velocity_loss_iplap,
)
from models.landmark_generator import build_split_landmark_generator
from models.peft_lora import build_lora_landmark_model
from utils.checkpoint import load_pretrained_into_split_model, split_state_to_iplap_state_dict
from utils.env import arg_or_env_path, require_path


LANDMARK_DEFAULTS = {
    "device": "cuda",
    "t_frames": 5,
    "nl": 15,
    "d_model": 512,
    "n_layers": 4,
    "n_head": 4,
    "dim_ff": 1024,
    "dropout": 0.1,
    "rank": 16,
    "alpha": 32.0,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "linear1", "linear2"],
    "extra_train_modules": None,
    "unfreeze_transformer_layers": 0,
    "audio_offset": -2,
    "open_focus_prob": 0.0,
    "open_focus_trials": 8,
    "epochs": 20,
    "max_steps": 0,
    "batch_size": 16,
    "val_batch_size": 16,
    "lr": 2e-4,
    "weight_decay": 0.01,
    "num_workers": 4,
    "save_interval": 500,
    "w_l1": 1.0,
    "w_lmd": 2.0,
    "w_velocity": 1.0,
    "w_mouth_delta": 1.5,
    "w_mouth_profile": 0.5,
    "final_checkpoint_name": "loratalk_landmark.pth",
}


def apply_landmark_default_args(args: Any, defaults: Optional[Dict[str, Any]] = None) -> Any:
    """Bổ sung các giá trị mặc định còn thiếu vào namespace cấu hình."""
    for name, value in (defaults or LANDMARK_DEFAULTS).items():
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def move_landmark_batch_to_device(batch: Tuple[torch.Tensor, ...], device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Chuyển toàn bộ tensor của một batch Landmark sang thiết bị đích."""
    return tuple(item.to(device, non_blocking=True) for item in batch)


class LandmarkTrainer:
    """
    Fine-tune Landmark bằng PEFT LoRA và xuất checkpoint đã merge cho IP-LAP.

    Quy trình: nạp checkpoint gốc → tách Q/K/V → train adapter → merge adapter
    → ghép Q/K/V về ``in_proj`` để inference IP-LAP dùng trực tiếp.
    """

    def __init__(self, args: Any):
        """Khởi tạo lần lượt đường dẫn, model, dataloader và optimizer."""
        self.args = apply_landmark_default_args(args)
        self.device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.setup_paths()
        self.setup_model()
        self.setup_dataloaders()
        self.setup_optimizer()

    def setup_paths(self) -> None:
        """Đọc đường dẫn từ args/.env và chuẩn bị thư mục đầu ra."""
        self.pretrained_ckpt = require_path(
            arg_or_env_path(self.args, "pretrained_ckpt", "LORATALK_PRETRAINED_LANDMARK_CKPT"),
            "pretrained landmark checkpoint",
            "LORATALK_PRETRAINED_LANDMARK_CKPT",
        )
        self.train_filelist = require_path(
            arg_or_env_path(self.args, "train_filelist", "LORATALK_TRAIN_FILELIST"),
            "landmark train filelist",
            "LORATALK_TRAIN_FILELIST",
        )
        self.val_filelist = require_path(
            arg_or_env_path(self.args, "val_filelist", "LORATALK_VAL_FILELIST"),
            "landmark validation filelist",
            "LORATALK_VAL_FILELIST",
        )
        self.landmark_root = require_path(
            arg_or_env_path(self.args, "landmark_root", "LORATALK_LANDMARK_ROOT"),
            "landmark root",
            "LORATALK_LANDMARK_ROOT",
        )
        self.audio_root = require_path(
            arg_or_env_path(self.args, "audio_root", "LORATALK_AUDIO_ROOT"),
            "audio root",
            "LORATALK_AUDIO_ROOT",
        )
        self.out_dir = require_path(
            arg_or_env_path(self.args, "output_dir", "LORATALK_LORA_OUTPUT_DIR", "outputs/stage01_loratalk_landmark"),
            "landmark output directory",
            "LORATALK_LORA_OUTPUT_DIR",
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_dir = self.out_dir / "adapter"
        self.export_dir = self.out_dir / "exported_pth"
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def build_base_model(self) -> torch.nn.Module:
        """Tạo model LoRA-ready và nạp toàn bộ trọng số pretrained IP-LAP."""
        model = build_split_landmark_generator(
            t_frames=self.args.t_frames,
            d_model=self.args.d_model,
            n_layers=self.args.n_layers,
            n_head=self.args.n_head,
            dim_ff=self.args.dim_ff,
            dropout=self.args.dropout,
        )
        load_pretrained_into_split_model(model, self.pretrained_ckpt)
        return model

    def setup_model(self) -> None:
        """Bọc base model bằng LoRA rồi báo tỷ lệ tham số thực sự được train."""
        base_model = self.build_base_model()
        self.model = build_lora_landmark_model(
            base_model,
            rank=self.args.rank,
            alpha=self.args.alpha,
            dropout=self.args.lora_dropout,
            target_modules=self.args.target_modules,
            resume_lora_dir=arg_or_env_path(self.args, "resume_lora_dir", "LORATALK_RESUME_LORA_DIR"),
            extra_train_modules=self.args.extra_train_modules,
            unfreeze_layers=self.args.unfreeze_transformer_layers,
        ).to(self.device)
        trainable = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        total = sum(param.numel() for param in self.model.parameters())
        print(f"LoRA-Talk Landmark trainable params: {trainable:,}/{total:,} ({100.0 * trainable / total:.2f}%)")

    def setup_dataloaders(self) -> None:
        """Tạo dataloader train ngẫu nhiên và validation tái lập được."""
        train_ds = LipLandmarkDataset(
            filelist_path=self.train_filelist,
            landmark_root=self.landmark_root,
            audio_root=self.audio_root,
            t_frames=self.args.t_frames,
            nl=self.args.nl,
            deterministic=False,
            open_focus_prob=self.args.open_focus_prob,
            open_focus_trials=self.args.open_focus_trials,
            audio_offset=self.args.audio_offset,
        )
        val_ds = LipLandmarkDataset(
            filelist_path=self.val_filelist,
            landmark_root=self.landmark_root,
            audio_root=self.audio_root,
            t_frames=self.args.t_frames,
            nl=self.args.nl,
            deterministic=True,
            audio_offset=self.args.audio_offset,
        )
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.args.val_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def setup_optimizer(self) -> None:
        """Chỉ tối ưu các tham số được đánh dấu ``requires_grad``."""
        params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.weight_decay)

    def compute_losses(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward Landmark và tính loss hình học/động học cùng các metric log."""
        t_mels, t_pose, t_content, nl_pose, nl_content = batch
        pred = self.model(t_mels, t_pose, nl_pose, nl_content)
        gt = t_content.reshape(-1, 2, t_content.size(-1))
        loss_l1 = l1_loss_landmark(pred, gt)
        loss_lmd = lmd_loss(pred, gt)
        loss_velocity = velocity_loss_iplap(pred, gt, self.args.t_frames)
        loss_mouth_delta = mouth_open_delta_loss(pred, gt, t_frames=self.args.t_frames)
        loss_mouth_profile = mouth_aperture_profile_loss(pred, gt)
        total = (
            loss_l1 * self.args.w_l1
            + loss_lmd * self.args.w_lmd
            + loss_velocity * self.args.w_velocity
            + loss_mouth_delta * self.args.w_mouth_delta
            + loss_mouth_profile * self.args.w_mouth_profile
        )
        metrics = {
            "loss": float(total.detach().cpu()),
            "l1": float(loss_l1.detach().cpu()),
            "lmd": float(loss_lmd.detach().cpu()),
            "velocity": float(loss_velocity.detach().cpu()),
            "mouth_delta": float(loss_mouth_delta.detach().cpu()),
            "mouth_profile": float(loss_mouth_profile.detach().cpu()),
        }
        return total, metrics

    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Thực hiện một bước backward và cập nhật adapter LoRA."""
        self.model.train()
        batch = move_landmark_batch_to_device(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.compute_losses(batch)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Đánh giá toàn bộ validation loader, có trọng số theo batch size."""
        self.model.eval()
        totals: Dict[str, float] = {}
        count = 0
        for batch in self.val_loader:
            batch = move_landmark_batch_to_device(batch, self.device)
            _, metrics = self.compute_losses(batch)
            batch_size = batch[0].size(0)
            count += batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
        return {key: value / max(count, 1) for key, value in totals.items()}

    def save_adapter(self, name: str) -> Path:
        """Lưu riêng PEFT adapter và metadata cần thiết để resume/xuất model."""
        adapter_dir = self.out_dir / name
        adapter_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(adapter_dir))
        metadata = {
            "global_step": self.global_step,
            "pretrained_ckpt": str(self.pretrained_ckpt),
            "t_frames": self.args.t_frames,
            "nl": self.args.nl,
            "rank": self.args.rank,
            "alpha": self.args.alpha,
            "target_modules": self.args.target_modules,
        }
        (adapter_dir / "trainer_state.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return adapter_dir

    def export_merged_checkpoint(self, adapter_dir: Path, out_path: Path) -> Path:
        """Merge adapter vào base model và xuất state dict tương thích IP-LAP."""
        base_model = self.build_base_model()
        merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False).merge_and_unload()
        state = split_state_to_iplap_state_dict(merged_model.state_dict())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, out_path)
        return out_path

    def train(self) -> None:
        """Huấn luyện theo epoch, lưu adapter tốt nhất và checkpoint cuối cùng."""
        best_lmd = float("inf")
        best_adapter: Optional[Path] = None
        for epoch in range(self.args.epochs):
            pbar = tqdm(self.train_loader, desc=f"Landmark epoch {epoch + 1}/{self.args.epochs}")
            for batch in pbar:
                metrics = self.train_step(batch)
                pbar.set_postfix({"loss": f"{metrics['loss']:.4f}", "lmd": f"{metrics['lmd']:.4f}"})
                if self.args.save_interval > 0 and self.global_step % self.args.save_interval == 0:
                    self.save_adapter(f"adapter_step{self.global_step:09d}")
                if self.args.max_steps and self.global_step >= self.args.max_steps:
                    break

            val_metrics = self.validate()
            print(f"Validation: {val_metrics}")
            if val_metrics["lmd"] < best_lmd:
                best_lmd = val_metrics["lmd"]
                best_adapter = self.save_adapter("adapter_best")
                merged_path = self.export_dir / self.args.final_checkpoint_name
                self.export_merged_checkpoint(best_adapter, merged_path)
                print(f"Exported LoRA-Talk Landmark checkpoint to {merged_path}")
            if self.args.max_steps and self.global_step >= self.args.max_steps:
                break

        final_adapter = self.save_adapter("adapter_final")
        self.export_merged_checkpoint(final_adapter, self.export_dir / self.args.final_checkpoint_name)
