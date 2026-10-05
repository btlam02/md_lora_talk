"""Pipeline fine-tune Stage 2 cho PAA Renderer."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.renderer_dataset import RendererFrameDataset
from models.renderer import PAARenderer
from models.discriminator import build_renderer_discriminator
from losses.renderer_losses import (
    mouth_l1_loss,
    lower_face_l1_loss,
    masked_l1,
    sketch_mouth_mask,
    mouth_edge_l1,
    temporal_l1,
    mouth_temporal_l1,
    alignment_temporal_l1,
    warp_temporal_l1,
    sketch_aperture_proxy,
    image_mouth_activity,
    derivative_cosine_loss,
)
from utils.env import arg_or_env_path, env_value, require_path


PAA_RENDERER_DEFAULTS = {
    "device": "cuda",
    "adaptation_mode": "translation",
    "max_epochs": 1,
    "max_steps": 300,
    "batch_size": 4,
    "val_batch_size": 4,
    "num_workers": 4,
    "lr": 1e-6,
    "disc_lr": 5e-6,
    "train_bn_stats": False,
    "max_train_clips": 300,
    "max_val_clips": 80,
    "train_stride": 3,
    "val_stride": 8,
    "random_refs": False,
    "sequence_length": 5,
    "audio_offset_frames": 0,
    "sketch_root_name": "04_sketch_loratalk",
    "mouth_mask_threshold": 0.03,
    "mouth_mask_dilation": 15,
    "w_warp": 0.0,
    "w_gen": 4.0,
    "w_l1": 0.0,
    "w_mouth": 0.0,
    "w_lower": 0.0,
    "w_mouth_mask": 1.0,
    "w_mouth_edge": 0.2,
    "w_temporal": 0.0,
    "w_mouth_temporal": 0.03,
    "w_aux_sketch_alignment": 0.05,
    "w_alignment_flow_temporal": 0.0,
    "w_alignment_weight_temporal": 0.0,
    "w_warp_temporal": 0.0,
    "w_gan": 0.25,
    "w_gan_feat": 1.0,
    "gan_start_step": 50,
    "save_interval": 100,
}


PAA_ALIGNMENT_LITE_DEFAULTS = {
    **PAA_RENDERER_DEFAULTS,
    "adaptation_mode": "alignment_lite",
    "lr": 5e-7,
    "disc_lr": 0.0,
    "w_warp": 1.0,
    "w_gen": 2.0,
    "w_mouth_mask": 0.0,
    "w_mouth_edge": 0.0,
    "w_mouth_temporal": 0.0,
    "w_aux_sketch_alignment": 0.0,
    "w_alignment_flow_temporal": 0.01,
    "w_alignment_weight_temporal": 0.01,
    "w_warp_temporal": 0.05,
    "w_gan": 0.0,
    "w_gan_feat": 0.0,
}


def apply_default_args(args: Any, defaults: Optional[Dict[str, Any]] = None) -> Any:
    """Điền cấu hình PAA còn thiếu và ưu tiên sketch root từ biến môi trường."""
    has_sketch_arg = hasattr(args, "sketch_root_name") and getattr(args, "sketch_root_name") not in {None, ""}
    for name, value in (defaults or PAA_RENDERER_DEFAULTS).items():
        if not hasattr(args, name):
            setattr(args, name, value)
    if not has_sketch_arg:
        args.sketch_root_name = env_value("LORATALK_RENDERER_SKETCH_ROOT", PAA_RENDERER_DEFAULTS["sketch_root_name"])
    return args


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Chuyển các tensor trong batch dictionary sang thiết bị đích."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def normalize_renderer_state_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    """Chuẩn hóa prefix checkpoint cũ về tên module của PAARenderer hiện tại."""
    clean_state = {}
    for key, value in state.items():
        new_key = key.removeprefix("module.")
        if new_key.startswith("flow_"):
            new_key = "flow_module." + new_key[len("flow_") :]
        clean_state[new_key] = value
    return clean_state


def flatten_image_batch(images: torch.Tensor) -> torch.Tensor:
    """Gộp batch và chiều thời gian khi input có dạng ``(B, T, C, H, W)``."""
    if images.ndim == 5:
        return images.reshape(-1, *images.shape[-3:])
    return images


def freeze_bn_running_stats(module: torch.nn.Module) -> None:
    """Giữ BatchNorm ở eval để không cập nhật running statistics khi fine-tune."""
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()


def set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    """Bật hoặc tắt gradient cho toàn bộ tham số của một module."""
    for param in module.parameters():
        param.requires_grad = value


def sequence_training_enabled(args: Any) -> bool:
    """Chỉ bật input sequence khi có ít nhất một temporal/alignment loss hoạt động."""
    return any(
        getattr(args, name, 0.0) > 0
        for name in (
            "w_temporal",
            "w_mouth_temporal",
            "w_alignment_flow_temporal",
            "w_alignment_weight_temporal",
            "w_warp_temporal",
            "w_aux_sketch_alignment",
        )
    )


def flatten_sequence_batch(batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Đổi dữ liệu sequence sang batch phẳng, lặp reference cho từng frame đích."""
    targets = batch["target_sequence"]
    sketches = batch["sketch_sequence"]
    mels = batch["mel_sequence"]
    batch_size, seq_len = targets.shape[:2]
    target_flat = targets.reshape(batch_size * seq_len, 1, *targets.shape[2:])
    sketch_flat = sketches.reshape(batch_size * seq_len, sketches.shape[2], *sketches.shape[3:])
    mel_flat = mels.reshape(batch_size * seq_len, *mels.shape[2:])
    ref_faces = batch["ref_faces"].unsqueeze(1).expand(-1, seq_len, -1, -1, -1, -1).reshape(
        batch_size * seq_len, *batch["ref_faces"].shape[1:]
    )
    ref_sketches = batch["ref_sketches"].unsqueeze(1).expand(-1, seq_len, -1, -1, -1, -1).reshape(
        batch_size * seq_len, *batch["ref_sketches"].shape[1:]
    )
    return target_flat, sketch_flat, ref_faces, ref_sketches, mel_flat


def discriminator_scores(prediction: Any) -> list[torch.Tensor]:
    """Chuẩn hóa output discriminator đa tỉ lệ về danh sách score cuối mỗi nhánh."""
    if isinstance(prediction, torch.Tensor):
        return [prediction]
    scores = []
    for item in prediction:
        if isinstance(item, (list, tuple)):
            scores.append(item[-1])
        else:
            scores.append(item)
    return scores


def lsgan_loss(prediction: Any, target_is_real: bool) -> torch.Tensor:
    """Tính LSGAN loss cho output đơn hoặc đa tỉ lệ của discriminator."""
    scores = discriminator_scores(prediction)
    loss = scores[0].new_tensor(0.0)
    for score in scores:
        target = torch.ones_like(score) if target_is_real else torch.zeros_like(score)
        loss = loss + F.mse_loss(score, target)
    return loss / max(len(scores), 1)


def discriminator_feature_matching_loss(fake_prediction: Any, real_prediction: Any) -> torch.Tensor:
    """So sánh các feature trung gian của discriminator, không lan gradient vào nhánh thật."""
    if not isinstance(fake_prediction, (list, tuple)) or not isinstance(real_prediction, (list, tuple)):
        return discriminator_scores(fake_prediction)[0].new_tensor(0.0)
    loss = discriminator_scores(fake_prediction)[0].new_tensor(0.0)
    count = 0
    for fake_scale, real_scale in zip(fake_prediction, real_prediction):
        if not isinstance(fake_scale, (list, tuple)) or not isinstance(real_scale, (list, tuple)):
            continue
        for fake_feat, real_feat in zip(fake_scale[:-1], real_scale[:-1]):
            loss = loss + F.l1_loss(fake_feat, real_feat.detach())
            count += 1
    return loss / max(count, 1)


class PAARendererTrainer:
    """
    Quản lý PAA adaptation, discriminator GAN, validation và lưu checkpoint.
    """

    def __init__(self, args: Any):
        """Khởi tạo cấu hình, model, dữ liệu và optimizer cho Stage 2."""
        self.args = apply_default_args(args)
        self.global_step = 0
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.setup_paths()
        self.setup_models()
        self.setup_dataloaders()
        self.setup_optimizers()

    def setup_paths(self) -> None:
        """Đọc/kiểm tra các đường dẫn checkpoint, dataset và thư mục output."""
        self.pretrained_renderer = arg_or_env_path(
            self.args,
            "pretrained_renderer",
            "LORATALK_PRETRAINED_RENDERER",
        )
        self.data_root = arg_or_env_path(self.args, "data_root", "LORATALK_RENDERER_DATA_ROOT")
        self.train_filelist = arg_or_env_path(
            self.args,
            "train_filelist",
            "LORATALK_RENDERER_TRAIN_FILELIST",
            arg_or_env_path(self.args, "train_filelist", "LORATALK_TRAIN_FILELIST"),
        )
        self.val_filelist = arg_or_env_path(
            self.args,
            "val_filelist",
            "LORATALK_RENDERER_VAL_FILELIST",
            arg_or_env_path(self.args, "val_filelist", "LORATALK_VAL_FILELIST"),
        )
        self.out_dir = require_path(
            arg_or_env_path(self.args, "out_dir", "LORATALK_RENDERER_OUTPUT_DIR", "outputs/renderer"),
            "renderer output directory",
            "LORATALK_RENDERER_OUTPUT_DIR",
        )
        self.data_root = require_path(self.data_root, "renderer data root", "LORATALK_RENDERER_DATA_ROOT")
        self.train_filelist = require_path(
            self.train_filelist,
            "renderer train filelist",
            "LORATALK_RENDERER_TRAIN_FILELIST",
        )
        self.val_filelist = require_path(
            self.val_filelist,
            "renderer validation filelist",
            "LORATALK_RENDERER_VAL_FILELIST",
        )

    def setup_models(self) -> None:
        """Khởi tạo renderer, nạp checkpoint và đặt adaptation mode tương ứng."""
        print("Initializing PAARenderer...")
        self.model = PAARenderer().to(self.device)
        if self.pretrained_renderer and self.pretrained_renderer.exists():
            checkpoint = torch.load(self.pretrained_renderer, map_location="cpu")
            state = checkpoint.get("state_dict", checkpoint)
            clean_state = normalize_renderer_state_dict(state)
            self.model.load_state_dict(clean_state, strict=True)
            print(f"Loaded pretrained renderer: {self.pretrained_renderer}")

        self.model.configure_adaptation(self.args.adaptation_mode)
        summary = self.model.trainable_summary()
        print(
            "Trainable params: "
            f"{summary['trainable_params']:,}/{summary['total_params']:,} "
            f"({summary['trainable_pct']:.2f}%)"
        )
        self.discriminator = build_renderer_discriminator(self.device)

    def setup_dataloaders(self) -> None:
        """Tạo train/validation loader với sequence khi temporal loss được bật."""
        print("Setting up renderer datasets...")
        train_ds = RendererFrameDataset(
            data_root=self.data_root,
            filelist=self.train_filelist,
            split="train",
            max_clips=self.args.max_train_clips,
            sample_stride=self.args.train_stride,
            random_refs=self.args.random_refs,
            sequence_length=self.args.sequence_length if sequence_training_enabled(self.args) else 1,
            audio_offset_frames=self.args.audio_offset_frames,
            sketch_root_name=self.args.sketch_root_name,
        )
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_ds = RendererFrameDataset(
            data_root=self.data_root,
            filelist=self.val_filelist,
            split="val",
            max_clips=self.args.max_val_clips,
            sample_stride=self.args.val_stride,
            random_refs=False,
            sequence_length=self.args.sequence_length if sequence_training_enabled(self.args) else 1,
            audio_offset_frames=self.args.audio_offset_frames,
            sketch_root_name=self.args.sketch_root_name,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.args.val_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def setup_optimizers(self) -> None:
        """Tạo optimizer generator và tùy chọn optimizer discriminator."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer_g = optim.Adam(trainable_params, lr=self.args.lr, betas=(0.5, 0.999))
        self.optimizer_d = None
        if self.args.w_gan > 0 and self.args.disc_lr > 0:
            self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.args.disc_lr, betas=(0.5, 0.999))

    def train_step(self, batch: Dict[str, Any]) -> Tuple[float, float, float]:
        """Cập nhật generator, sau đó discriminator nếu GAN loss đang được kích hoạt."""
        self.model.train()
        if not self.args.train_bn_stats:
            freeze_bn_running_stats(self.model)
        batch = move_batch_to_device(batch, self.device)

        use_sequence = sequence_training_enabled(self.args)
        if use_sequence:
            target_input, sketch_input, ref_faces, ref_sketches, mel_input = flatten_sequence_batch(batch)
            batch_size, seq_len = batch["target_sequence"].shape[:2]
        else:
            target_input = batch["target"]
            sketch_input = batch["target_sketches"]
            ref_faces = batch["ref_faces"]
            ref_sketches = batch["ref_sketches"]
            mel_input = batch["mel"]
            batch_size, seq_len = target_input.size(0), 1

        self.optimizer_g.zero_grad()
        fake, wrapped_ref, warp_loss, gen_loss = self.model(target_input, sketch_input, ref_faces, ref_sketches, mel_input)
        fake = flatten_image_batch(fake)
        gt = flatten_image_batch(target_input)

        mouth_mask = sketch_mouth_mask(
            sketch_input,
            threshold=self.args.mouth_mask_threshold,
            dilation=self.args.mouth_mask_dilation,
        )
        loss_warp = warp_loss.sum()
        loss_gen = gen_loss.sum()
        loss_l1 = torch.nn.functional.l1_loss(fake, gt)
        loss_mouth = mouth_l1_loss(fake, gt)
        loss_lower = lower_face_l1_loss(fake, gt)
        loss_mouth_mask = masked_l1(fake, gt, mouth_mask)
        loss_mouth_edge = mouth_edge_l1(fake, gt, mouth_mask)

        loss_temporal = fake.new_tensor(0.0)
        loss_mouth_temporal = fake.new_tensor(0.0)
        loss_alignment_flow_temporal = fake.new_tensor(0.0)
        loss_alignment_weight_temporal = fake.new_tensor(0.0)
        loss_warp_temporal = fake.new_tensor(0.0)
        loss_aux_sketch_alignment = fake.new_tensor(0.0)
        if use_sequence:
            fake_seq = fake.view(batch_size, seq_len, *fake.shape[1:])
            gt_seq = gt.view(batch_size, seq_len, *gt.shape[1:])
            mask_seq = mouth_mask.view(batch_size, seq_len, 1, *mouth_mask.shape[-2:])
            if self.args.w_temporal > 0:
                loss_temporal = temporal_l1(fake_seq, gt_seq)
            if self.args.w_mouth_temporal > 0:
                loss_mouth_temporal = mouth_temporal_l1(fake_seq, gt_seq, mask_seq)
            if self.args.w_alignment_flow_temporal > 0:
                loss_alignment_flow_temporal = alignment_temporal_l1(
                    getattr(self.model.flow_module, "last_output_flows", None), batch_size, seq_len
                )
            if self.args.w_alignment_weight_temporal > 0:
                loss_alignment_weight_temporal = alignment_temporal_l1(
                    getattr(self.model.flow_module, "last_output_weights", None), batch_size, seq_len
                )
            if self.args.w_warp_temporal > 0:
                loss_warp_temporal = warp_temporal_l1(wrapped_ref, gt, batch_size, seq_len)
            if self.args.w_aux_sketch_alignment > 0:
                image_activity = image_mouth_activity(fake_seq, mask_seq)
                sketch_activity = sketch_aperture_proxy(batch["sketch_sequence"], self.args.mouth_mask_threshold)
                loss_aux_sketch_alignment = derivative_cosine_loss(image_activity, sketch_activity)

        loss_g = (
            loss_warp * self.args.w_warp
            + loss_gen * self.args.w_gen
            + loss_l1 * self.args.w_l1
            + loss_mouth * self.args.w_mouth
            + loss_lower * self.args.w_lower
            + loss_mouth_mask * self.args.w_mouth_mask
            + loss_mouth_edge * self.args.w_mouth_edge
            + loss_temporal * self.args.w_temporal
            + loss_mouth_temporal * self.args.w_mouth_temporal
            + loss_alignment_flow_temporal * self.args.w_alignment_flow_temporal
            + loss_alignment_weight_temporal * self.args.w_alignment_weight_temporal
            + loss_warp_temporal * self.args.w_warp_temporal
            + loss_aux_sketch_alignment * self.args.w_aux_sketch_alignment
        )
        use_gan = self.args.w_gan > 0 and self.global_step >= self.args.gan_start_step
        loss_d = fake.new_tensor(0.0)
        if use_gan:
            set_requires_grad(self.discriminator, False)
            fake_pair = torch.cat([sketch_input[:, 2], fake], dim=1)
            real_pair = torch.cat([sketch_input[:, 2], gt], dim=1)
            fake_pred_for_g = self.discriminator(fake_pair)
            real_pred_for_feat = self.discriminator(real_pair.detach())
            loss_gan = lsgan_loss(fake_pred_for_g, True)
            loss_gan_feat = discriminator_feature_matching_loss(fake_pred_for_g, real_pred_for_feat)
            loss_g = loss_g + loss_gan * self.args.w_gan + loss_gan_feat * self.args.w_gan_feat

        loss_g.backward()
        self.optimizer_g.step()

        if use_gan and self.optimizer_d is not None:
            set_requires_grad(self.discriminator, True)
            self.optimizer_d.zero_grad()
            fake_pair = torch.cat([sketch_input[:, 2], fake.detach()], dim=1)
            real_pair = torch.cat([sketch_input[:, 2], gt], dim=1)
            loss_d_fake = lsgan_loss(self.discriminator(fake_pair), False)
            loss_d_real = lsgan_loss(self.discriminator(real_pair), True)
            loss_d = 0.5 * (loss_d_fake + loss_d_real)
            loss_d.backward()
            self.optimizer_d.step()

        self.global_step += 1
        return loss_g.item(), loss_mouth_mask.item(), loss_d.item()

    def checkpoint_payload(self, epoch: int, metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Đóng gói trạng thái cần thiết để resume Stage 2."""
        return {
            "state_dict": self.model.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict() if self.optimizer_d is not None else None,
            "epoch": epoch,
            "global_step": getattr(self, "global_step", 0),
            "metrics": metrics or {},
        }

    def train(self) -> None:
        """Chạy vòng lặp epoch/step và lưu checkpoint theo chu kỳ cấu hình."""
        best_score = float("inf")
        out_dir = self.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.args.max_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.max_epochs}")
            for batch in pbar:
                loss_g, loss_mouth, loss_d = self.train_step(batch)
                pbar.set_postfix(
                    {"loss_g": f"{loss_g:.4f}", "loss_d": f"{loss_d:.4f}", "mouth_mask": f"{loss_mouth:.4f}"}
                )
                if self.args.save_interval > 0 and self.global_step % self.args.save_interval == 0:
                    ckpt_path = out_dir / f"renderer_paa_v2_step{self.global_step:09d}.pth"
                    torch.save(
                        self.checkpoint_payload(
                            epoch,
                            {"loss_g": loss_g, "loss_d": loss_d, "mouth_mask": loss_mouth},
                        ),
                        ckpt_path,
                    )
                    print(f"Saved checkpoint to {ckpt_path}")
                if self.args.max_steps and self.global_step >= self.args.max_steps:
                    break

            # Checkpoint save
            ckpt_path = out_dir / f"renderer_paa_epoch_{epoch+1}.pth"
            torch.save(self.checkpoint_payload(epoch), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
            if self.args.max_steps and self.global_step >= self.args.max_steps:
                break
