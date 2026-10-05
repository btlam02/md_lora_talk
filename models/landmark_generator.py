"""Landmark Generator IP-LAP với Transformer tách projection để gắn LoRA."""

from __future__ import annotations

import math
import sys
import importlib.util
import types
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.env import env_path

IPLAP_DIR = env_path("LORATALK_IP_LAP_DIR", Path(__file__).resolve().parents[2] / "third_party" / "IP_LAP")


def bootstrap_iplap_imports() -> None:
    """Chuẩn bị dependency tối thiểu và đường dẫn để import mã IP-LAP gốc."""
    if "lws" not in sys.modules:
        fake_lws = types.ModuleType("lws")
        fake_lws.lws = lambda *args, **kwargs: None
        sys.modules["lws"] = fake_lws
    if str(IPLAP_DIR) not in sys.path:
        sys.path.insert(0, str(IPLAP_DIR))


bootstrap_iplap_imports()


def _load_iplap_landmark_symbols():
    """Lấy các lớp gốc từ IP-LAP mà không sao chép lại toàn bộ kiến trúc."""
    module_path = IPLAP_DIR / "models" / "landmark_generator.py"
    spec = importlib.util.spec_from_file_location("_lora_talk_iplap_landmark_generator", module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy landmark generator IP-LAP: {module_path}")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load IP-LAP landmark generator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Landmark_generator, module.PositionalEmbedding


Landmark_generator, PositionalEmbedding = _load_iplap_landmark_symbols()

class MultiheadAttentionSplit(nn.Module):
    """
    Attention nhiều head có Q/K/V tách riêng.

    ``nn.MultiheadAttention`` gốc gộp Q/K/V trong ``in_proj_weight`` nên PEFT
    không thể target từng projection. Lớp này giữ phép tính attention tương đương
    nhưng biểu diễn chúng bằng các ``nn.Linear`` độc lập.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False):
        """Tính scaled dot-product attention cho tensor batch-first ``(B, T, D)``."""
        batch, q_len, dim = query.shape
        heads, head_dim = self.nhead, self.head_dim
        q = self.q_proj(query).view(batch, q_len, heads, head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch, -1, heads, head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch, -1, heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if attn_mask is not None:
            # Quy ước giống PyTorch: bool=True là vị trí không được attention.
            scores = scores.masked_fill(attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else scores + attn_mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :].bool(), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, q_len, dim)
        return self.out_proj(out), attn if need_weights else None


class TransformerEncoderLayerSplit(nn.Module):
    """
    Một encoder layer theo thứ tự post-norm của ``TransformerEncoderLayer`` IP-LAP.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiheadAttentionSplit(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Áp dụng self-attention, residual và FFN theo kiến trúc Transformer gốc."""
        attn_out, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout2(ff))


class FusionTransformerEncoderSplit(nn.Module):
    """
    Ghép token reference, audio và pose rồi đưa qua các encoder layer tách LoRA.
    """
    def __init__(self, t_frames: int, d_model: int, nlayers: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.T = t_frames
        self.position_v = PositionalEmbedding(d_model=d_model)
        self.position_a = PositionalEmbedding(d_model=d_model)
        self.modality = nn.Embedding(4, d_model, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayerSplit(d_model, nhead, dim_feedforward, dropout) for _ in range(nlayers)]
        )

    def forward(self, ref_embedding: torch.Tensor, mel_embedding: torch.Tensor, pose_embedding: torch.Tensor) -> torch.Tensor:
        """Hợp nhất ba loại embedding thành chuỗi token đã được attention."""
        position_v_encoding = self.position_v(pose_embedding)
        position_a_encoding = self.position_a(mel_embedding)
        device = pose_embedding.device
        modality_v = self.modality(torch.ones((pose_embedding.size(0), self.T), dtype=torch.long, device=device))
        modality_a = self.modality(2 * torch.ones((mel_embedding.size(0), self.T), dtype=torch.long, device=device))
        pose_tokens = pose_embedding + position_v_encoding + modality_v
        audio_tokens = mel_embedding + position_a_encoding + modality_a
        ref_tokens = ref_embedding + self.modality(
            3 * torch.ones((ref_embedding.size(0), ref_embedding.size(1)), dtype=torch.long, device=device)
        )
        x = torch.cat((ref_tokens, audio_tokens, pose_tokens), dim=1)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x


def build_split_landmark_generator(
    t_frames: int = 5,
    d_model: int = 512,
    n_layers: int = 4,
    n_head: int = 4,
    dim_ff: int = 1024,
    dropout: float = 0.1
) -> nn.Module:
    """Tạo Landmark Generator IP-LAP và thay fusion transformer bằng bản LoRA-ready."""
    if d_model != 512:
        raise ValueError("Landmark_generator IP-LAP hiện chỉ hỗ trợ d_model=512")
    model = Landmark_generator(
        T=t_frames,
        d_model=d_model,
        nlayers=n_layers,
        nhead=n_head,
        dim_feedforward=dim_ff,
        dropout=dropout,
    )
    model.fusion_transformer = FusionTransformerEncoderSplit(
        t_frames=t_frames,
        d_model=d_model,
        nlayers=n_layers,
        nhead=n_head,
        dim_feedforward=dim_ff,
        dropout=dropout,
    )
    return model
