from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import IMAGE_EXTENSIONS, build_transforms, resolve_media_path


def build_pair_table(
    csv_path: Path,
    dataset_dir: Path,
    image_columns: list[str],
) -> pd.DataFrame:
    if len(image_columns) < 2:
        raise ValueError(
            f"At least two image columns are required, got {image_columns}"
        )

    df = pd.read_csv(csv_path)
    rows = []

    for _, row in df.iterrows():
        image_paths = []
        valid_sample = True
        for image_column in image_columns:
            raw_path = row.get(image_column, "")
            if pd.isna(raw_path) or not str(raw_path).strip():
                valid_sample = False
                break

            resolved = resolve_media_path(dataset_dir, str(raw_path))
            if resolved is None or resolved.suffix.lower() not in IMAGE_EXTENSIONS:
                valid_sample = False
                break
            image_paths.append(resolved)

        # Multi-image fusion uses complete cases so each feature position has
        # the same meaning for every sample.
        if not valid_sample:
            continue

        sample = {
            "patient_id": str(row["patient_id"]),
            "label": int(row["label"]),
            "source_excel": str(row.get("source_excel", "")),
            "source_row": int(row.get("source_row", -1)),
            "fold": int(row.get("fold", -1)),
        }
        for image_view, (image_column, image_path) in enumerate(
            zip(image_columns, image_paths), start=1
        ):
            sample[f"image_path_{image_view}"] = str(image_path)
            sample[f"image_column_{image_view}"] = image_column
        rows.append(sample)

    if not rows:
        raise RuntimeError(f"No paired image samples were built from {csv_path}")
    return pd.DataFrame(rows)


class PairedAirwayImageDataset(Dataset):
    def __init__(
        self,
        table: pd.DataFrame,
        image_size: int,
        image_view: int | None = None,
    ):
        self.table = table.reset_index(drop=True)
        self.transform = build_transforms(image_size=image_size, is_train=False)
        self.image_view = image_view
        self.num_views = len(
            [column for column in self.table.columns if column.startswith("image_path_")]
        )
        if self.num_views < 2:
            raise ValueError("At least two image views are required")
        if image_view is not None and not 1 <= image_view <= self.num_views:
            raise ValueError(
                f"image_view must be between 1 and {self.num_views}; got {image_view}"
            )

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        sample = {
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "patient_id": row["patient_id"],
        }
        if self.image_view is None:
            for image_view in range(1, self.num_views + 1):
                image = Image.open(row[f"image_path_{image_view}"]).convert("RGB")
                sample[f"pixel_values_{image_view}"] = self.transform(image)
            return sample

        image_path = row[f"image_path_{self.image_view}"]
        image = Image.open(image_path).convert("RGB")
        sample["pixel_values"] = self.transform(image)
        sample["image_path"] = image_path
        return sample
