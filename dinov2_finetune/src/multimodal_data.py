from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import IMAGE_EXTENSIONS, resolve_media_path


def build_multimodal_pair_table(
    csv_path: Path,
    dataset_dir: Path,
    image_columns: list[str],
) -> pd.DataFrame:
    """Keep structured columns while requiring a complete image set."""
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

        if not valid_sample:
            continue

        sample = row.to_dict()
        sample["patient_id"] = str(row["patient_id"])
        sample["label"] = int(row["label"])
        sample["source_excel"] = str(row.get("source_excel", ""))
        sample["source_row"] = int(row.get("source_row", -1))
        sample["fold"] = int(row.get("fold", -1))
        for image_view, (image_column, image_path) in enumerate(
            zip(image_columns, image_paths), start=1
        ):
            sample[f"image_path_{image_view}"] = str(image_path)
            sample[f"image_column_{image_view}"] = image_column
        rows.append(sample)

    if not rows:
        raise RuntimeError(f"No multimodal samples were built from {csv_path}")
    return pd.DataFrame(rows)


def prepare_structured_features(
    df: pd.DataFrame,
    categorical_columns: list[str],
    numerical_columns: list[str],
):
    """Create the same leakage-safe structured variables used by the baseline."""
    data = df.copy()

    required_source_columns = ["性别", "年龄", "身高", "体重", "种族"]
    missing_source_columns = [
        column for column in required_source_columns if column not in data.columns
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
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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
            ("categorical", categorical_pipeline, categorical_columns),
            ("numerical", numerical_pipeline, numerical_columns),
        ]
    )
