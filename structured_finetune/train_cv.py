import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import dump

from src.data import load_fold_csv, prepare_features
from src.logger import setup_logger, write_json
from src.metrics import binary_metrics, find_best_threshold
from src.models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def train_one_fold(config, model_name, fold_idx):
    seed = config["seed"] + fold_idx
    set_seed(seed)

    output_dir = Path(config["output_dir"]) / model_name / f"fold_{fold_idx}"
    logger = setup_logger(output_dir)

    train_df = load_fold_csv(
        Path(config["folds_dir"]) / f"fold_{fold_idx}_train.csv"
    )
    val_df = load_fold_csv(
        Path(config["folds_dir"]) / f"fold_{fold_idx}_val.csv"
    )

    categorical_columns = config["features"]["categorical"]
    numerical_columns = config["features"]["numerical"]

    x_train, y_train, train_raw = prepare_features(
        train_df, categorical_columns, numerical_columns
    )
    x_val, y_val, val_raw = prepare_features(
        val_df, categorical_columns, numerical_columns
    )

    model = build_model(
        name=model_name,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        seed=seed,
    )

    logger.info(
        f"model={model_name} fold={fold_idx} "
        f"train_samples={len(x_train)} val_samples={len(x_val)}"
    )
    logger.info(f"features={categorical_columns + numerical_columns}")
    logger.info(
        f"train_positive={int(y_train.sum())} "
        f"val_positive={int(y_val.sum())}"
    )

    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    sample_weight = y_train.map({0: 1.0, 1: negative / max(1, positive)}).to_numpy()

    fit_kwargs = {}
    if model_name in {
        "hist_gradient_boosting",
        "gradient_boosting",
        "adaboost",
        "knn",
    }:
        if model_name != "knn":
            fit_kwargs["model__sample_weight"] = sample_weight

    model.fit(x_train, y_train, **fit_kwargs)
    train_probabilities = model.predict_proba(x_train)[:, 1]
    if config.get("threshold_strategy", "fixed") == "fixed":
        threshold = config["decision_threshold"]
    else:
        threshold = find_best_threshold(
            y_train,
            train_probabilities,
            strategy=config.get("threshold_strategy", "train_youden").replace("train_", ""),
        )

    probabilities = model.predict_proba(x_val)[:, 1]
    metrics = binary_metrics(
        y_val,
        probabilities,
        threshold=threshold,
    )
    metrics.update(
        {
            "model": model_name,
            "fold": fold_idx,
            "train_samples": len(x_train),
            "val_samples": len(x_val),
        }
    )

    logger.info(
        f"auc={metrics['auc']:.4f} f1={metrics['f1']:.4f} "
        f"sensitivity={metrics['sensitivity']:.4f} "
        f"specificity={metrics['specificity']:.4f} threshold={threshold:.4f}"
    )

    predictions = val_raw[
        ["patient_id", "label", "性别", "年龄", "身高", "体重", "种族", "BMI"]
    ].copy()
    predictions["probability"] = probabilities
    predictions["prediction"] = (
        probabilities >= threshold
    ).astype(int)
    predictions.to_csv(
        output_dir / "val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dump(model, output_dir / "model.joblib")
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    all_results = []
    for model_name in config["models"]:
        for fold_idx in range(config["num_folds"]):
            all_results.append(train_one_fold(config, model_name, fold_idx))

    summary = pd.DataFrame(all_results)
    summary.to_csv(
        Path(config["output_dir"]) / "cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    grouped = []
    for model_name, group in summary.groupby("model"):
        grouped.append(
            {
                "model": model_name,
                "mean_auc": float(group["auc"].mean()),
                "std_auc": float(group["auc"].std()),
                "mean_f1": float(group["f1"].mean()),
                "mean_sensitivity": float(group["sensitivity"].mean()),
                "mean_specificity": float(group["specificity"].mean()),
                "mean_accuracy": float(group["accuracy"].mean()),
            }
        )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
