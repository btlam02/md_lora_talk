"""Các hàm loss hình học và động học cho Stage 1 Landmark LoRA."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F

T_DEFAULT = 5


def l1_loss_landmark(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Sai số L1 theo từng tọa độ landmark."""
    return F.l1_loss(pred, gt)


def lmd_loss(pred: torch.Tensor, gt: torch.Tensor, point_slice: Optional[slice] = None) -> torch.Tensor:
    """Landmark distance: khoảng cách Euclid trung bình theo từng điểm."""
    if point_slice is not None:
        pred = pred[:, :, point_slice]
        gt = gt[:, :, point_slice]
    return torch.sqrt(((pred - gt) ** 2).sum(dim=1) + 1e-9).mean()


def _reshape_landmark_sequence(landmarks: torch.Tensor, t_frames: int) -> torch.Tensor:
    """Đổi ``(B*T, 2, P)`` sang ``(B, T, 2, P)`` và kiểm tra tính hợp lệ."""
    if t_frames < 2:
        raise ValueError("t_frames phải lớn hơn hoặc bằng 2 để tính sai khác theo thời gian")
    batch_times_frames, _, point_count = landmarks.shape
    if batch_times_frames % t_frames != 0:
        raise ValueError(f"BT={batch_times_frames} không chia hết cho T={t_frames}")
    return landmarks.reshape(batch_times_frames // t_frames, t_frames, 2, point_count)


def velocity_loss_iplap(pred: torch.Tensor, gt: torch.Tensor, t_frames: int = T_DEFAULT) -> torch.Tensor:
    """Phạt chênh lệch vận tốc landmark giữa các frame liên tiếp."""
    pred_seq = _reshape_landmark_sequence(pred, t_frames)
    gt_seq = _reshape_landmark_sequence(gt, t_frames)
    batch = pred_seq.size(0)
    # Giữ thứ tự [x0, y0, x1, y1, ...] trước khi so sánh vector vận tốc.
    pred_flat = pred_seq.permute(0, 1, 3, 2).flatten(start_dim=2)
    gt_flat = gt_seq.permute(0, 1, 3, 2).flatten(start_dim=2)
    pred_v = pred_flat[:, 1:, :] - pred_flat[:, :-1, :]
    gt_v = gt_flat[:, 1:, :] - gt_flat[:, :-1, :]
    return F.pairwise_distance(pred_v.reshape(batch * (t_frames - 1), -1), gt_v.reshape(batch * (t_frames - 1), -1)).mean()


def mouth_open_inner_torch(lm: torch.Tensor) -> torch.Tensor:
    """Đo độ mở môi trong bằng biên độ trục y của 20 điểm môi trong."""
    inner_y = lm[:, 1, 37:57]
    return inner_y.max(dim=-1)[0] - inner_y.min(dim=-1)[0]


def mouth_width_outer_torch(lm: torch.Tensor) -> torch.Tensor:
    """Đo bề rộng môi ngoài trên trục x; dùng cho phân tích/metric phụ trợ."""
    outer_x = lm[:, 0, 17:37]
    return outer_x.max(dim=-1)[0] - outer_x.min(dim=-1)[0]


def mouth_open_delta_loss(pred: torch.Tensor, gt: torch.Tensor, order: int = 1, t_frames: int = T_DEFAULT) -> torch.Tensor:
    """Ép đạo hàm theo thời gian của độ mở miệng khớp với ground truth."""
    if order < 1 or order >= t_frames:
        raise ValueError("order phải trong khoảng [1, t_frames - 1]")
    pred_open = mouth_open_inner_torch(_reshape_landmark_sequence(pred, t_frames).flatten(0, 1)).reshape(-1, t_frames)
    gt_open = mouth_open_inner_torch(_reshape_landmark_sequence(gt, t_frames).flatten(0, 1)).reshape(-1, t_frames)
    for _ in range(order):
        pred_open = pred_open[:, 1:] - pred_open[:, :-1]
        gt_open = gt_open[:, 1:] - gt_open[:, :-1]
    return F.l1_loss(pred_open, gt_open)


def mouth_aperture_profile_torch(lm: torch.Tensor) -> torch.Tensor:
    """Tạo profile khe hở của môi ngoài và môi trong tại các cặp điểm đối diện."""
    outer_upper_y = lm[:, 1, 18:27]
    outer_lower_y = torch.flip(lm[:, 1, 28:37], dims=[1])
    inner_upper_y = lm[:, 1, 38:47]
    inner_lower_y = torch.flip(lm[:, 1, 48:57], dims=[1])
    outer_profile = outer_lower_y - outer_upper_y
    inner_profile = inner_lower_y - inner_upper_y
    return torch.cat([outer_profile, inner_profile], dim=1)


def mouth_aperture_profile_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """So sánh profile khe hở miệng bằng Smooth L1 để ổn định với outlier."""
    return F.smooth_l1_loss(mouth_aperture_profile_torch(pred), mouth_aperture_profile_torch(gt))
