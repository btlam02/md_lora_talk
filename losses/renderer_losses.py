"""Các loss ảnh, vùng miệng và temporal cho Stage 2 PAA Renderer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from utils.landmark import IMG_SIZE, MOUTH_SLICE, MOUTH_PRIOR_SLICE, LOWER_SLICE


def slice_l1(pred: torch.Tensor, gt: torch.Tensor, region_slice: tuple[slice, slice]) -> torch.Tensor:
    """Tính L1 trên một vùng ảnh theo lát cắt y/x."""
    return F.l1_loss(pred[:, :, region_slice[0], region_slice[1]], gt[:, :, region_slice[0], region_slice[1]])


def mouth_l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 giới hạn trong vùng miệng."""
    return slice_l1(pred, gt, MOUTH_SLICE)


def lower_face_l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 cho nửa dưới khuôn mặt."""
    return slice_l1(pred, gt, LOWER_SLICE)


def masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 chuẩn hóa theo số pixel mask hợp lệ và số channel."""
    channel_dim = 1 if pred.dim() == 4 else 2
    diff = torch.abs(pred - gt) * mask.to(dtype=pred.dtype)
    denom = mask.sum().clamp_min(1.0) * pred.size(channel_dim)
    return diff.sum() / denom


def mouth_prior_mask(reference: torch.Tensor) -> torch.Tensor:
    """Tạo mask prior hình chữ nhật quanh miệng theo kích thước ảnh đầu vào."""
    mask = reference.new_zeros((reference.size(0), 1, reference.size(2), reference.size(3)))
    ys, xs = MOUTH_PRIOR_SLICE
    mask[:, :, ys, xs] = 1.0
    return mask


def sketch_mouth_mask(sketch: torch.Tensor, threshold: float = 0.03, dilation: int = 15) -> torch.Tensor:
    """Suy ra mouth mask từ nét sketch; fallback về prior nếu sketch quá thưa."""
    center_sketch = sketch[:, sketch.size(1) // 2] if sketch.dim() == 5 else sketch
    line_mask = center_sketch.amax(dim=1, keepdim=True).gt(threshold).to(center_sketch.dtype)
    prior = mouth_prior_mask(center_sketch)
    mask = line_mask * prior
    if dilation > 1:
        if dilation % 2 == 0:
            dilation += 1
        mask = F.max_pool2d(mask, kernel_size=dilation, stride=1, padding=dilation // 2) * prior
    valid = mask.flatten(1).sum(dim=1).view(-1, 1, 1, 1) > 8.0
    return torch.where(valid, mask, prior)


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Tính biên độ gradient Sobel độc lập cho từng kênh ảnh."""
    kernel_x = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    channels = x.size(1)
    grad_x = F.conv2d(x, kernel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    grad_y = F.conv2d(x, kernel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)


def mouth_edge_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """So sánh biên Sobel trong mask miệng."""
    return masked_l1(sobel_edges(pred), sobel_edges(gt), mask)


def temporal_l1(fake_seq: torch.Tensor, gt_seq: torch.Tensor) -> torch.Tensor:
    """So sánh biến thiên ảnh giữa các frame liên tiếp."""
    if fake_seq.size(1) <= 1:
        return fake_seq.new_tensor(0.0)
    fake_diff = fake_seq[:, 1:] - fake_seq[:, :-1]
    gt_diff = gt_seq[:, 1:] - gt_seq[:, :-1]
    return F.l1_loss(fake_diff, gt_diff)


def mouth_temporal_l1(fake_seq: torch.Tensor, gt_seq: torch.Tensor, mouth_mask_seq: torch.Tensor) -> torch.Tensor:
    """Temporal L1 chỉ ở các pixel miệng xuất hiện trong hai frame liên tiếp."""
    if fake_seq.size(1) <= 1:
        return fake_seq.new_tensor(0.0)
    fake_diff = fake_seq[:, 1:] - fake_seq[:, :-1]
    gt_diff = gt_seq[:, 1:] - gt_seq[:, :-1]
    mask_pairs = (mouth_mask_seq[:, 1:] * mouth_mask_seq[:, :-1])
    return masked_l1(fake_diff, gt_diff, mask_pairs)


def alignment_temporal_l1(signal_flat: torch.Tensor | None, batch_size: int, seq_len: int) -> torch.Tensor:
    """Ép tín hiệu alignment thay đổi mượt theo thời gian."""
    if signal_flat is None or seq_len < 2:
        raise ValueError("Alignment temporal loss requires sequence outputs.")
    signal_seq = signal_flat.view(batch_size, seq_len, *signal_flat.shape[1:])
    delta = signal_seq[:, 1:] - signal_seq[:, :-1]
    return F.l1_loss(delta, torch.zeros_like(delta))


def warp_temporal_l1(wrapped_ref: torch.Tensor, gt: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """So sánh biến thiên temporal của warped reference với ground truth."""
    wrapped_seq = wrapped_ref.view(batch_size, seq_len, *wrapped_ref.shape[1:])
    gt_seq = gt.view(batch_size, seq_len, *gt.shape[1:])
    return F.l1_loss(wrapped_seq[:, 1:] - wrapped_seq[:, :-1], gt_seq[:, 1:] - gt_seq[:, :-1])


def sketch_aperture_proxy(sketch_seq: torch.Tensor, threshold: float) -> torch.Tensor:
    """Ước lượng độ mở miệng từ độ phân tán theo trục y của nét sketch."""
    batch_size, seq_len = sketch_seq.shape[:2]
    center_sketch = sketch_seq[:, :, sketch_seq.size(2) // 2].reshape(batch_size * seq_len, 3, IMG_SIZE, IMG_SIZE)
    prior = mouth_prior_mask(center_sketch)
    line_mask = center_sketch.amax(dim=1, keepdim=True).gt(threshold).to(center_sketch.dtype) * prior
    y_coords = torch.linspace(0.0, 1.0, IMG_SIZE, device=sketch_seq.device, dtype=sketch_seq.dtype).view(1, 1, IMG_SIZE, 1)
    mass = line_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    y_mean = (line_mask * y_coords).sum(dim=(1, 2, 3)) / mass
    y_var = (line_mask * (y_coords - y_mean.view(-1, 1, 1, 1)).square()).sum(dim=(1, 2, 3)) / mass
    return torch.sqrt(y_var.clamp_min(1e-8)).view(batch_size, seq_len)


def image_mouth_activity(fake_seq: torch.Tensor, mask_seq: torch.Tensor) -> torch.Tensor:
    """Đo hoạt động miệng từ biên và độ tối của ảnh sinh trong mouth mask."""
    gray = fake_seq.mean(dim=2, keepdim=True)
    edge = sobel_edges(gray.reshape(-1, 1, IMG_SIZE, IMG_SIZE)).view_as(gray)
    darkness = 1.0 - gray
    signal = 0.5 * edge + 0.5 * darkness
    denom = mask_seq.sum(dim=(2, 3, 4)).clamp_min(1.0)
    return (signal * mask_seq).sum(dim=(2, 3, 4)) / denom


def derivative_cosine_loss(pred_signal: torch.Tensor, target_signal: torch.Tensor) -> torch.Tensor:
    """So sánh hướng đạo hàm temporal của hai tín hiệu bằng cosine distance."""
    pred_delta = pred_signal[:, 1:] - pred_signal[:, :-1]
    target_delta = target_signal[:, 1:] - target_signal[:, :-1]
    pred_delta = pred_delta - pred_delta.mean(dim=1, keepdim=True)
    target_delta = target_delta - target_delta.mean(dim=1, keepdim=True)
    return 1.0 - F.cosine_similarity(pred_delta, target_delta, dim=1, eps=1e-6).mean()
