"""Dataset Stage 1: chuẩn bị audio mel và landmark cho Landmark LoRA."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.landmark import ORDER

T_DEFAULT = 5
NL_DEFAULT = 15
MEL_STEP_DEFAULT = 16
FPS_DEFAULT = 25
POSE_POINTS = 74
CONTENT_POINTS = 57


def get_frame_ids(lm_dir: Path) -> List[int]:
    """Đọc các file ``<frame_id>.npy`` và trả về ID frame đã sắp xếp."""
    frame_ids = []
    for landmark_path in lm_dir.glob("*.npy"):
        try:
            frame_ids.append(int(landmark_path.stem))
        except ValueError:
            # Bỏ qua file .npy không đại diện cho một frame.
            continue
    return sorted(frame_ids)


def _read_filelist(filelist_path: Path) -> List[str]:
    """Đọc danh sách clip và loại bỏ dòng rỗng."""
    return [line.strip() for line in filelist_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def precompute_val_frames(
    filelist_path: Path,
    landmark_root: Path,
    audio_root: Path,
    t_frames: int = T_DEFAULT,
    nl: int = NL_DEFAULT,
    audio_offset: int = -2,
) -> Tuple[List[str], List[Optional[Tuple[str, List[int], List[int]]]]]:
    """Lập plan validation cố định để mỗi epoch đánh giá cùng các frame."""
    samples = _read_filelist(filelist_path)

    frame_plan = []
    for i, sample in enumerate(samples):
        rng = random.Random(i * 31337)
        lm_dir = landmark_root / sample
        audio_npy = audio_root / sample / "audio.npy"
        if not (lm_dir.is_dir() and audio_npy.exists()):
            frame_plan.append(None)
            continue

        frame_ids = get_frame_ids(lm_dir)
        vid_len = len(frame_ids)
        if vid_len < t_frames + nl + 5:
            frame_plan.append(None)
            continue
        try:
            lo = max(2, -audio_offset)
            hi = vid_len - t_frames - 2
            start_pos = rng.randint(lo, hi)
        except ValueError:
            frame_plan.append(None)
            continue

        t_idxs = frame_ids[start_pos : start_pos + t_frames]
        available = [fid for fid in frame_ids if fid not in set(t_idxs)]
        if len(available) < nl:
            frame_plan.append(None)
            continue
        nl_idxs = rng.sample(available, nl)
        frame_plan.append((sample, t_idxs, nl_idxs))
    return samples, frame_plan


class LipLandmarkDataset(Dataset):
    """
    Dataset landmark cho IP-LAP/LoRA.

    Mỗi phần tử trả về ``(t_mels, t_pose, t_content, nl_pose, nl_content)``.
    Validation dùng frame plan cố định; train chọn frame ngẫu nhiên.
    """
    def __init__(
        self,
        filelist_path: Path,
        landmark_root: Path,
        audio_root: Path,
        t_frames: int = T_DEFAULT,
        nl: int = NL_DEFAULT,
        mel_step: int = MEL_STEP_DEFAULT,
        deterministic: bool = False,
        open_focus_prob: float = 0.0,
        open_focus_trials: int = 8,
        audio_offset: int = -2,
    ):
        self.landmark_root = Path(landmark_root)
        self.audio_root = Path(audio_root)
        self.t_frames = t_frames
        self.nl = nl
        self.mel_step = mel_step
        self.deterministic = deterministic
        self.open_focus_prob = float(open_focus_prob)
        self.open_focus_trials = int(open_focus_trials)
        self.audio_offset = int(audio_offset)
        self._frame_ids_cache: Dict[str, List[int]] = {}

        self.samples = _read_filelist(filelist_path)

        if deterministic:
            _, self._frame_plan = precompute_val_frames(
                filelist_path, landmark_root, audio_root, t_frames, nl, audio_offset=self.audio_offset
            )
            self._valid_plans = [(i, plan) for i, plan in enumerate(self._frame_plan) if plan is not None]
        else:
            self._frame_plan = None
            self._valid_plans = None

    def __len__(self) -> int:
        """Trả số plan hợp lệ ở validation hoặc số clip ở train."""
        if self.deterministic:
            return len(self._valid_plans)
        return len(self.samples)


    def _load_landmarks(self, lm_dir: Path, t_idxs: List[int], nl_idxs: List[int]):
        """Nạp landmark dict của frame đích và các frame reference."""
        t_lms = [np.load(lm_dir / f"{frame_id}.npy", allow_pickle=True).item() for frame_id in t_idxs]
        nl_lms = [np.load(lm_dir / f"{frame_id}.npy", allow_pickle=True).item() for frame_id in nl_idxs]
        return t_lms, nl_lms

    def _get_frame_ids_for(self, lm_dir: Path) -> List[int]:
        """Lưu cache ID frame để không quét lại thư mục trong mỗi lần lấy mẫu."""
        key = str(lm_dir)
        if key not in self._frame_ids_cache:
            self._frame_ids_cache[key] = get_frame_ids(lm_dir)
        return self._frame_ids_cache[key]

    def _stack(self, lm_list: List[Dict[str, Any]], key: str, n_pts: int) -> torch.Tensor:
        """Sắp landmark theo thứ tự IP-LAP và tạo tensor có dạng ``(N, 2, P)``."""
        arr = torch.zeros((len(lm_list), 2, n_pts), dtype=torch.float32)
        for i, data in enumerate(lm_list):
            lm = sorted(data[key], key=lambda item: ORDER[int(item[0])])
            arr[i, 0, :] = torch.FloatTensor([float(item[1]) for item in lm])
            arr[i, 1, :] = torch.FloatTensor([float(item[2]) for item in lm])
        return arr

    def _build_mels(self, audio_npy_path: Path, t_idxs: List[int]) -> Optional[torch.Tensor]:
        """Cắt mel cho từng frame đích và zero-pad khi đoạn cuối audio chưa đủ."""
        audio_mel = np.load(audio_npy_path)
        t_mels = []
        for frame_idx in t_idxs:
            mel_start = frame_idx + self.audio_offset
            if mel_start < 0:
                return None
            start_idx = int(80.0 * (mel_start / float(FPS_DEFAULT)))
            mel = audio_mel[start_idx : start_idx + self.mel_step, :]
            if mel.shape[0] < self.mel_step:
                pad = np.zeros((self.mel_step - mel.shape[0], mel.shape[1]), dtype=mel.dtype)
                mel = np.concatenate([mel, pad], axis=0)
            t_mels.append(mel.T)
        return torch.FloatTensor(np.asarray(t_mels)).unsqueeze(1)

    def _build_batch_from_ids(
        self,
        sample: str,
        target_ids: List[int],
        reference_ids: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chuyển ID frame thành đủ năm tensor mà Landmark Generator cần."""
        lm_dir = self.landmark_root / sample
        audio_npy = self.audio_root / sample / "audio.npy"
        target_landmarks, reference_landmarks = self._load_landmarks(lm_dir, target_ids, reference_ids)
        t_pose = self._stack(target_landmarks, "pose_landmarks", POSE_POINTS)
        t_content = self._stack(target_landmarks, "content_landmarks", CONTENT_POINTS)
        nl_pose = self._stack(reference_landmarks, "pose_landmarks", POSE_POINTS)
        nl_content = self._stack(reference_landmarks, "content_landmarks", CONTENT_POINTS)
        t_mels = self._build_mels(audio_npy, target_ids)
        if t_mels is None:
            raise RuntimeError(f"Không thể tạo mel cho {sample}, target_ids={target_ids}")
        return t_mels, t_pose, t_content, nl_pose, nl_content

    def _getitem_deterministic(self, idx: int):
        """Tạo một batch validation từ frame plan đã được lập sẵn."""
        _, (sample, t_idxs, nl_idxs) = self._valid_plans[idx]
        return self._build_batch_from_ids(sample, t_idxs, nl_idxs)

    def __getitem__(self, idx: int):
        """Lấy một batch; train thử lại tối đa 20 lần nếu clip/frame bị lỗi."""
        if self.deterministic:
            return self._getitem_deterministic(idx)

        for _ in range(20):
            sample = self.samples[idx]
            lm_dir = self.landmark_root / sample
            audio_npy = self.audio_root / sample / "audio.npy"
            if not (lm_dir.is_dir() and audio_npy.exists()):
                idx = random.randint(0, len(self.samples) - 1)
                continue

            frame_ids = self._get_frame_ids_for(lm_dir)
            vid_len = len(frame_ids)
            if vid_len < self.t_frames + self.nl + 5:
                idx = random.randint(0, len(self.samples) - 1)
                continue

            lo = max(2, -self.audio_offset)
            hi = vid_len - self.t_frames - 2
            if hi < lo:
                idx = random.randint(0, len(self.samples) - 1)
                continue
            start_pos = random.randint(lo, hi)

            t_idxs = frame_ids[start_pos : start_pos + self.t_frames]
            available = [fid for fid in frame_ids if fid not in set(t_idxs)]
            if len(available) < self.nl:
                idx = random.randint(0, len(self.samples) - 1)
                continue
            nl_idxs = random.sample(available, self.nl)

            try:
                return self._build_batch_from_ids(sample, t_idxs, nl_idxs)
            except Exception:
                idx = random.randint(0, len(self.samples) - 1)
                continue

        raise RuntimeError("Không tìm được sample Landmark hợp lệ sau 20 lần thử")
