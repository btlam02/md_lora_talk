"""Dataset Stage 2: frame, sketch, reference và mel cho PAA Renderer."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.landmark import IMG_SIZE
from utils.image import read_rgb_image

FPS = 25
MEL_STEP_SIZE = 16
WINDOW_T = 5
REF_N = 3


@dataclass(frozen=True)
class FrameSample:
    """Metadata index frame của một sample PAA đã được kiểm tra tính hợp lệ."""
    key: str
    face_dir: Path
    sketch_dir: Path
    audio_npy: Path
    frame_ids: Tuple[int, ...]
    ref_frame_ids: Tuple[int, ...]
    center_frame_id: int
    sequence_centers: Tuple[int, ...]


def frame_ids(frame_dir: Path) -> List[int]:
    """Đọc ID số từ tên các file PNG trong thư mục frame."""
    ids = []
    for path in frame_dir.glob("*.png"):
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    return sorted(ids)


def read_rgb(path: Path) -> np.ndarray:
    """Đọc ảnh, resize về kích thước model và chuyển BGR sang RGB."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(images: List[np.ndarray]) -> torch.Tensor:
    """Đổi danh sách ảnh HWC uint8 thành tensor NCHW trong khoảng [0, 1]."""
    arr = np.asarray(images, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(arr)


class RendererFrameDataset(Dataset):
    """
    Dataset frame cho Stage 2 PAA Renderer.

    Mỗi sample gồm target, cửa sổ sketch, reference face/sketch và mel audio.
    """
    def __init__(
        self,
        data_root: Path,
        filelist: Path,
        split: str,
        max_clips: Optional[int] = None,
        sample_stride: int = 1,
        random_refs: bool = False,
        sequence_length: int = 1,
        audio_offset_frames: int = -2,
        sketch_root_name: str = "04_sketch",
    ) -> None:
        """Lập chỉ mục các frame đủ cửa sổ sketch và reference hợp lệ."""
        self.data_root = Path(data_root)
        self.filelist = Path(filelist)
        self.split = split
        self.random_refs = random_refs
        self.sequence_length = sequence_length
        self.audio_offset_frames = audio_offset_frames
        self.sketch_root_name = sketch_root_name
        self.samples: List[FrameSample] = []

        keys = [line.strip() for line in filelist.read_text(encoding="utf-8").splitlines() if line.strip()]
        if max_clips is not None:
            keys = keys[:max_clips]

        for key in tqdm(keys, desc=f"Indexing {split}"):
            face_dir = data_root / "03_face" / key
            sketch_dir = data_root / sketch_root_name / key
            audio_npy = data_root / "06_audio" / key / "audio.npy"
            if not face_dir.is_dir() or not sketch_dir.is_dir() or not audio_npy.exists():
                continue
            ids = frame_ids(face_dir)
            id_set = set(ids)
            sketch_ids = frame_ids(sketch_dir)
            sketch_id_set = set(sketch_ids)
            ref_ids = tuple(fid for fid in ids if fid in sketch_id_set)
            if len(ids) < WINDOW_T:
                continue
            if len(ref_ids) < REF_N:
                continue
            radius = sequence_length // 2
            for center in ids[2 + radius : -2 - radius : sample_stride]:
                sequence_centers = tuple(range(center - radius, center + radius + 1))
                required_ids = set()
                for seq_center in sequence_centers:
                    required_ids.update(range(seq_center - 2, seq_center + 3))
                if not all(fid in id_set for fid in required_ids):
                    continue
                if not required_ids.issubset(sketch_id_set):
                    continue
                self.samples.append(
                    FrameSample(key, face_dir, sketch_dir, audio_npy, tuple(ids), ref_ids, center, sequence_centers)
                )

        if not self.samples:
            raise ValueError(f"No valid {split} samples from {filelist}")
        print(f"{split}: {len(self.samples)} frame samples")

    def __len__(self) -> int:
        """Trả về số frame center/sequence hợp lệ đã lập chỉ mục."""
        return len(self.samples)

    def _ref_ids(self, sample: FrameSample) -> List[int]:
        """Chọn ba frame reference ngẫu nhiên khi train hoặc cố định khi validation."""
        ids = list(sample.ref_frame_ids)
        if self.random_refs and self.split == "train":
            return random.sample(ids, REF_N)
        positions = [0, len(ids) // 2, len(ids) - 1]
        return [ids[pos] for pos in positions]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Nạp ảnh/mel cho một sample và đóng gói theo schema trainer sử dụng."""
        sample = self.samples[idx]
        ref_ids = self._ref_ids(sample)

        target_faces = [read_rgb(sample.face_dir / f"{fid}.png") for fid in sample.sequence_centers]
        sketch_windows = [
            [read_rgb(sample.sketch_dir / f"{fid}.png") for fid in range(center - 2, center + 3)]
            for center in sample.sequence_centers
        ]
        ref_faces = [read_rgb(sample.face_dir / f"{fid}.png") for fid in ref_ids]
        ref_sketches = [read_rgb(sample.sketch_dir / f"{fid}.png") for fid in ref_ids]

        audio_mel = np.load(sample.audio_npy)
        mels = []
        for center in sample.sequence_centers:
            mel_start_frame_idx = max(center + self.audio_offset_frames, 0)
            start_idx = int(80.0 * (mel_start_frame_idx / float(FPS)))
            mel = audio_mel[start_idx : start_idx + MEL_STEP_SIZE, :]
            if mel.shape[0] != MEL_STEP_SIZE:
                pad = np.zeros((MEL_STEP_SIZE, audio_mel.shape[1]), dtype=audio_mel.dtype)
                pad[: mel.shape[0]] = mel
                mel = pad
            mels.append(torch.FloatTensor(mel.T).unsqueeze(0).unsqueeze(0))

        target_sequence = to_tensor(target_faces)
        sketch_sequence = torch.stack([to_tensor(window) for window in sketch_windows], dim=0)
        mel_sequence = torch.stack(mels, dim=0)
        center_idx = len(sample.sequence_centers) // 2

        return {
            "key": sample.key,
            "center_frame_id": sample.center_frame_id,
            "target": target_sequence[center_idx : center_idx + 1],
            "target_sketches": sketch_sequence[center_idx],
            "target_sequence": target_sequence,
            "sketch_sequence": sketch_sequence,
            "ref_faces": to_tensor(ref_faces),
            "ref_sketches": to_tensor(ref_sketches),
            "mel": mel_sequence[center_idx],
            "mel_sequence": mel_sequence,
        }
