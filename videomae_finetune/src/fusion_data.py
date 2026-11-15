from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .data import VIDEO_EXTENSIONS, load_video_frames, resolve_media_path


def build_pair_table(
    csv_path: Path,
    dataset_dir: Path,
    video_columns: list[str],
) -> pd.DataFrame:
    if len(video_columns) != 2:
        raise ValueError(f"Exactly two video columns are required, got {video_columns}")

    df = pd.read_csv(csv_path)
    rows = []
    first_column, second_column = video_columns

    for _, row in df.iterrows():
        first_raw = row.get(first_column, "")
        second_raw = row.get(second_column, "")
        if pd.isna(first_raw) or pd.isna(second_raw):
            continue
        if not str(first_raw).strip() or not str(second_raw).strip():
            continue

        first_path = resolve_media_path(dataset_dir, str(first_raw))
        second_path = resolve_media_path(dataset_dir, str(second_raw))
        if first_path is None or second_path is None:
            continue
        if first_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if second_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        rows.append(
            {
                "patient_id": str(row["patient_id"]),
                "label": int(row["label"]),
                "video_path_1": str(first_path),
                "video_path_2": str(second_path),
                "video_column_1": first_column,
                "video_column_2": second_column,
                "source_excel": str(row.get("source_excel", "")),
                "source_row": int(row.get("source_row", -1)),
                "fold": int(row.get("fold", -1)),
            }
        )

    if not rows:
        raise RuntimeError(f"No paired video samples were built from {csv_path}")
    return pd.DataFrame(rows)


class PairedAirwayVideoDataset(Dataset):
    def __init__(
        self,
        table: pd.DataFrame,
        num_frames: int,
        image_size: int,
        video_view: int,
    ):
        if video_view not in (1, 2):
            raise ValueError(f"video_view must be 1 or 2; got {video_view}")
        self.table = table.reset_index(drop=True)
        self.num_frames = num_frames
        self.image_size = image_size
        self.video_view = video_view

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        video = load_video_frames(
            video_path=row[f"video_path_{self.video_view}"],
            num_frames=self.num_frames,
            image_size=self.image_size,
        )
        return {
            "pixel_values": video,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "patient_id": row["patient_id"],
            "video_path_1": row["video_path_1"],
            "video_path_2": row["video_path_2"],
        }
