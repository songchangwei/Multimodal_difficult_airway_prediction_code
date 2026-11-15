import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.train import sanitize_name, train_one_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    all_column_summaries = []
    per_column_results = []

    for video_column in config.data.train_video_columns:
        fold_summaries = []
        for fold_idx in range(config.num_folds):
            summary = train_one_fold(config, fold_idx, video_column)
            fold_summaries.append(summary)
            all_column_summaries.append(summary)

        valid_aucs = [item["best_auc"] for item in fold_summaries if item["best_auc"] >= 0]
        column_result = {
            "video_column": video_column,
            "output_subdir": sanitize_name(video_column),
            "folds": fold_summaries,
            "mean_auc": float(np.mean(valid_aucs)) if valid_aucs else None,
            "std_auc": float(np.std(valid_aucs)) if valid_aucs else None,
        }
        per_column_results.append(column_result)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(per_column_results, f, ensure_ascii=False, indent=2)

    summary_rows = []
    for item in all_column_summaries:
        row = {
            "video_column": item["video_column"],
            "fold": item["fold"],
            "best_epoch": item["best_epoch"],
            "best_auc": item["best_auc"],
            "train_videos": item["train_videos"],
            "val_videos": item["val_videos"],
            "device": item["device"],
        }
        metrics = item.get("best_metrics") or {}
        for key, value in metrics.items():
            row[f"best_{key}"] = value
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
