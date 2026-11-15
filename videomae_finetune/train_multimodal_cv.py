import argparse

from src.multimodal_config import load_video_structured_config
from src.multimodal_train import (
    train_one_multimodal_fold,
    write_cv_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train an all-video plus structured-data classifier. "
            "Existing single-video VideoMAE backbones are frozen."
        )
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_video_structured_config(args.config)
    fold_summaries = [
        train_one_multimodal_fold(config, fold_idx)
        for fold_idx in range(config.num_folds)
    ]
    summary = write_cv_summary(config, fold_summaries)
    print(
        "5-fold video multimodal training finished: "
        f"mean_auc={summary['mean_auc']}, std_auc={summary['std_auc']}"
    )


if __name__ == "__main__":
    main()
