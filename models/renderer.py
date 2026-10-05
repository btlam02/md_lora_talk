"""PAARenderer: giữ kiến trúc IP-LAP và kiểm soát phần tham số được fine-tune."""

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


def _load_iplap_renderer():
    """Nạp lớp Renderer gốc trực tiếp từ source IP-LAP đã cấu hình."""
    module_path = IP_LAP_PATH / "models" / "video_renderer.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy renderer IP-LAP: {module_path}")
    spec = importlib.util.spec_from_file_location("_lora_talk_iplap_video_renderer", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load IP-LAP renderer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Renderer


Renderer = _load_iplap_renderer()


class PAARenderer(Renderer):
    """
    Renderer IP-LAP có các adaptation mode rõ ràng.

    Mode PAA đóng băng phần sinh không gian (flow, convolution, SPADE) và chỉ
    cập nhật audio encoder/AdaIN; các mode khác mở đúng phần cần fine-tune.
    """
    def __init__(self):
        """Khởi tạo renderer gốc và trạng thái các module cần giữ eval."""
        super().__init__()
        self._freeze_spatial = False
        self._eval_when_training: list[nn.Module] = []

    @staticmethod
    def set_requires_grad(module: nn.Module, value: bool) -> None:
        """Bật/tắt gradient cho mọi tham số của module được chỉ định."""
        for param in module.parameters():
            param.requires_grad = value

    def freeze_spatial_layers(self) -> None:
        """Đóng băng flow và phần spatial của translation, chỉ giữ audio/AdaIN trainable."""
        self._freeze_spatial = True

        # 1. Đóng băng DenseFlowNetwork hoàn toàn.
        for param in self.flow_module.parameters():
            param.requires_grad = False

        # 2. Translation chỉ giữ đường audio và điều chế AdaIN được cập nhật.
        for name, param in self.translation.named_parameters():
            if "audio_encoder" in name or "adain_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.set_frozen_modules_eval()
        print("Đã đóng băng spatial layers PAA; chỉ train audio_encoder và AdaIN.")

    def configure_adaptation(self, mode: str) -> None:
        """
        Thiết lập phần renderer được phép fine-tune theo adaptation mode.

        - ``alignment_lite``: warm-up alignment cuối flow.
        - ``translation``: đóng băng flow, train toàn bộ translation.
        - ``paa``/``adain``: chỉ audio encoder và AdaIN.
        - ``paa_plus``: PAA cộng thêm các block texture miệng cuối.
        - ``flow``: train toàn bộ dense-flow alignment.
        - ``full``: train toàn bộ renderer.
        """
        self._freeze_spatial = False
        self._eval_when_training = []
        if mode == "full":
            self.set_requires_grad(self, True)
            return
        if mode == "flow":
            self.set_requires_grad(self, False)
            self.set_requires_grad(self.flow_module, True)
            return
        if mode == "alignment_lite":
            self.set_requires_grad(self, False)
            for name in ["spade_layer_4", "conv_4", "conv_5"]:
                self.set_requires_grad(getattr(self.flow_module, name), True)
            self._eval_when_training = [self.translation]
            self.translation.eval()
            return
        if mode == "translation":
            self.set_requires_grad(self, False)
            self.set_requires_grad(self.translation, True)
            self._eval_when_training = [self.flow_module]
            self.flow_module.eval()
            return

        self.freeze_spatial_layers()
        if mode in {"paa", "adain"}:
            return
        if mode == "paa_plus":
            for name in ["spade_2", "adain_2", "spade_4", "conv_last"]:
                self.set_requires_grad(getattr(self.translation, name), True)
            return
        raise ValueError(f"Unsupported adaptation mode: {mode}")

    def trainable_summary(self) -> dict[str, float]:
        """Trả về số và tỷ lệ tham số đang được phép train."""
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total if total else 0.0,
        }

    def set_frozen_modules_eval(self) -> None:
        """Giữ module frozen ở eval để BatchNorm không cập nhật running statistics."""
        if not self._freeze_spatial:
            return
        self.flow_module.eval()
        for name, module in self.translation.named_modules():
            if not name:
                continue
            if "audio_encoder" in name or "adain_" in name:
                continue
            module.eval()

    def train(self, mode: bool = True):
        """Gọi train chuẩn rồi khôi phục eval cho các nhánh đã được đóng băng."""
        super().train(mode)
        if mode:
            self.set_frozen_modules_eval()
            for module in self._eval_when_training:
                module.eval()
        return self
