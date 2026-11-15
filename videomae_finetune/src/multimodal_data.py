from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset

import torch

from .data import VIDEO_EXTENSIONS, load_video_frames, resolve_media_path


def build_multivideo_table(
    csv_path: Path,
    dataset_dir: Path,
    video_columns: list[str],
) -> pd.DataFrame:
    """保留结构化字段，并只保留六路视频全部存在的患者样本。"""
    if not video_columns:
        raise ValueError("video_columns must contain at least one video column")

    df = pd.read_csv(csv_path)
    rows = []

    for _, row in df.iterrows():
        video_paths = []
        valid_sample = True
        for video_column in video_columns:
            raw_path = row.get(video_column, "")
            if pd.isna(raw_path) or not str(raw_path).strip():
                valid_sample = False
                break

            resolved = resolve_media_path(dataset_dir, str(raw_path))
            if (
                resolved is None
                or resolved.suffix.lower() not in VIDEO_EXTENSIONS
            ):
                valid_sample = False
                break
            video_paths.append(resolved)

        if not valid_sample:
            continue

        sample = row.to_dict()
        sample["patient_id"] = str(row["patient_id"])
        sample["label"] = int(row["label"])
        sample["source_excel"] = str(row.get("source_excel", ""))
        sample["source_row"] = int(row.get("source_row", -1))
        sample["fold"] = int(row.get("fold", -1))
        for video_view, (video_column, video_path) in enumerate(
            zip(video_columns, video_paths),
            start=1,
        ):
            sample[f"video_path_{video_view}"] = str(video_path)
            sample[f"video_column_{video_view}"] = video_column
        rows.append(sample)

    if not rows:
        raise RuntimeError(
            f"No complete multi-video samples were built from {csv_path}; "
            f"required columns={video_columns}"
        )
    return pd.DataFrame(rows)


class MultiVideoDataset(Dataset):
    """读取同一患者指定视角的视频，供单路冻结特征提取器使用。"""

    def __init__(
        self,
        table: pd.DataFrame,
        num_frames: int,
        image_size: int,
        video_view: int,
    ):
        if video_view < 1:
            raise ValueError(f"video_view must be positive, got {video_view}")
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
            "label": torch.tensor(
                float(row["label"]),
                dtype=torch.float32,
            ),
            "patient_id": row["patient_id"],
        }


def prepare_structured_features(
    df: pd.DataFrame,
    categorical_columns: list[str],
    numerical_columns: list[str],
):
    """生成结构化变量；编码、填补和标准化在训练折外部单独拟合。"""
    data = df.copy()
    required_source_columns = ["性别", "年龄", "身高", "体重", "种族"]
    missing_source_columns = [
        column
        for column in required_source_columns
        if column not in data.columns
    ]
    if missing_source_columns:
        raise KeyError(
            f"Missing structured source columns: {missing_source_columns}"
        )

    age = pd.to_numeric(data["年龄"], errors="coerce")
    height = pd.to_numeric(data["身高"], errors="coerce")
    weight = pd.to_numeric(data["体重"], errors="coerce")

    data["age"] = age
    data["height_cm"] = height
    data["weight_kg"] = weight
    data["BMI"] = weight / (height / 100.0) ** 2
    data["BMI"] = data["BMI"].replace([np.inf, -np.inf], np.nan)
    data["age_squared"] = data["age"] ** 2
    data["bmi_squared"] = data["BMI"] ** 2
    data["age_bmi"] = data["age"] * data["BMI"]
    data["age_group"] = pd.cut(
        data["age"],
        bins=[0, 40, 60, 120],
        labels=["<40", "40-60", ">60"],
        include_lowest=True,
    ).astype("object")
    data["bmi_group"] = pd.cut(
        data["BMI"],
        bins=[0, 18.5, 24, 28, 100],
        labels=["underweight", "normal", "overweight", "obese"],
        include_lowest=True,
    ).astype("object")

    feature_columns = list(categorical_columns) + list(numerical_columns)
    missing_feature_columns = [
        column for column in feature_columns if column not in data.columns
    ]
    if missing_feature_columns:
        raise KeyError(
            f"Missing structured feature columns: {missing_feature_columns}"
        )

    x = data[feature_columns].copy()
    y = pd.to_numeric(data["label"], errors="raise").astype(int)
    return x, y, data


def build_structured_preprocessor(
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
            ("numerical", numerical_pipeline, numerical_columns),
        ]
    )
