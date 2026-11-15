from pathlib import Path

import numpy as np
import pandas as pd


def load_fold_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fold file not found: {path}")
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame, categorical_columns, numerical_columns):
    data = df.copy()

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
    missing_columns = [column for column in feature_columns if column not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing feature columns: {missing_columns}")

    x = data[feature_columns].copy()
    y = pd.to_numeric(data["label"], errors="raise").astype(int)
    return x, y, data
