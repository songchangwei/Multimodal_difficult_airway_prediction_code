from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def build_image_level_table(csv_path: Path, dataset_dir: Path, image_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        label = int(row["label"])
        patient_id = str(row["patient_id"])
        source_excel = str(row.get("source_excel", ""))
        source_row = int(row.get("source_row", -1))
        fold = int(row.get("fold", -1)) if "fold" in row else -1

        for column in image_columns:
            raw_path = row.get(column, "")
            if pd.isna(raw_path) or not str(raw_path).strip():
                continue
            resolved = resolve_media_path(dataset_dir, str(raw_path))
            if resolved is None or resolved.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rows.append(
                {
                    "patient_id": patient_id,
                    "label": label,
                    "image_path": str(resolved),
                    "image_column": column,
                    "source_excel": source_excel,
                    "source_row": source_row,
                    "fold": fold,
                }
            )

    if not rows:
        raise RuntimeError(f"No image samples were built from {csv_path}")
    return pd.DataFrame(rows)


def build_single_column_table(csv_path: Path, dataset_dir: Path, image_column: str) -> pd.DataFrame:
    return build_image_level_table(
        csv_path=csv_path,
        dataset_dir=dataset_dir,
        image_columns=[image_column],
    )


def build_transforms(image_size: int, is_train: bool):
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class AirwayImageDataset(Dataset):
    def __init__(self, table: pd.DataFrame, image_size: int, is_train: bool):
        self.table = table.reset_index(drop=True)
        self.transform = build_transforms(image_size=image_size, is_train=is_train)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return {
            "pixel_values": image,
            "label": label,
            "patient_id": row["patient_id"],
            "image_path": row["image_path"],
            "image_column": row["image_column"],
        }
