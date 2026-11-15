from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

try:
    import av
except ImportError:
    av = None


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_dataset_dir(data_root: Path) -> Path:
    candidates = [p for p in data_root.iterdir() if p.is_dir()]
    dataset_dirs = [p for p in candidates if any(p.glob("1727742600*"))]
    if not dataset_dirs:
        raise FileNotFoundError(f"Could not find dataset directory under {data_root}")
    dataset_dirs.sort()
    return dataset_dirs[0]


def resolve_media_path(dataset_dir: Path, raw_path: str) -> Path | None:
    raw_path = str(raw_path).strip()
    if not raw_path:
        return None

    parts = Path(raw_path.lstrip("/\\")).parts
    if parts and parts[0] == "1727742600":
        parts = parts[1:]
    if not parts:
        return None

    direct = dataset_dir.joinpath(*parts)
    if direct.exists():
        return direct

    if len(parts) < 2:
        return None

    patient_id, visit_date, *rest = parts
    for machine_dir in dataset_dir.glob("1727742600*"):
        candidate = machine_dir / patient_id / visit_date
        if rest:
            candidate = candidate.joinpath(*rest)
        if candidate.exists():
            return candidate
    return None


def build_video_level_table(csv_path: Path, dataset_dir: Path, video_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        label = int(row["label"])
        patient_id = str(row["patient_id"])
        source_excel = str(row.get("source_excel", ""))
        source_row = int(row.get("source_row", -1))
        fold = int(row.get("fold", -1)) if "fold" in row else -1

        for column in video_columns:
            raw_path = row.get(column, "")
            if pd.isna(raw_path) or not str(raw_path).strip():
                continue
            resolved = resolve_media_path(dataset_dir, str(raw_path))
            if resolved is None or resolved.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            rows.append(
                {
                    "patient_id": patient_id,
                    "label": label,
                    "video_path": str(resolved),
                    "video_column": column,
                    "source_excel": source_excel,
                    "source_row": source_row,
                    "fold": fold,
                }
            )

    if not rows:
        raise RuntimeError(f"No video samples were built from {csv_path}")
    return pd.DataFrame(rows)


def build_single_column_table(csv_path: Path, dataset_dir: Path, video_column: str) -> pd.DataFrame:
    return build_video_level_table(csv_path, dataset_dir, [video_column])


def sample_frame_indices(num_total_frames: int, num_sampled_frames: int) -> np.ndarray:
    if num_total_frames <= 0:
        raise ValueError("num_total_frames must be positive")
    if num_total_frames >= num_sampled_frames:
        return np.linspace(0, num_total_frames - 1, num_sampled_frames, dtype=np.int32)
    indices = np.linspace(0, num_total_frames - 1, num_total_frames, dtype=np.int32)
    pad = np.full((num_sampled_frames - num_total_frames,), num_total_frames - 1, dtype=np.int32)
    return np.concatenate([indices, pad], axis=0)


def load_video_frames(video_path: str, num_frames: int, image_size: int) -> torch.Tensor:
    if av is not None:
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            total_frames = int(stream.frames)
            if total_frames <= 0:
                container.close()
                raise RuntimeError(f"Could not read frames from {video_path}")

            indices = set(sample_frame_indices(total_frames, num_frames).tolist())
            frames = []
            last_index = max(indices)
            for frame_id, frame in enumerate(container.decode(video=0)):
                if frame_id in indices:
                    image = frame.to_ndarray(format="rgb24")
                    image = cv2.resize(
                        image,
                        (image_size, image_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    frames.append(image)
                if frame_id >= last_index:
                    break
            container.close()

            if frames:
                while len(frames) < num_frames:
                    frames.append(frames[-1].copy())
                frames = np.asarray(frames[:num_frames], dtype=np.float32) / 255.0
                tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
                return torch.stack(
                    [normalize(frame, IMAGENET_MEAN, IMAGENET_STD) for frame in tensor],
                    dim=0,
                )
        except Exception:
            if "container" in locals():
                container.close()

    # OpenCV fallback for environments without PyAV or unsupported codecs.
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frames from {video_path}")

    indices = set(sample_frame_indices(total_frames, num_frames).tolist())
    frames = []
    frame_id = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        frame_id += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"No sampled frames were read from {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    frames = np.asarray(frames[:num_frames], dtype=np.float32) / 255.0
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
    tensor = torch.stack([normalize(frame, IMAGENET_MEAN, IMAGENET_STD) for frame in tensor], dim=0)
    return tensor


class AirwayVideoDataset(Dataset):
    def __init__(self, table: pd.DataFrame, num_frames: int, image_size: int):
        self.table = table.reset_index(drop=True)
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        video = load_video_frames(
            video_path=row["video_path"],
            num_frames=self.num_frames,
            image_size=self.image_size,
        )
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return {
            "pixel_values": video,
            "label": label,
            "patient_id": row["patient_id"],
            "video_path": row["video_path"],
            "video_column": row["video_column"],
        }
