import argparse

from src.fusion_config import load_fusion_config
from src.fusion_train import train_one_fusion_fold, write_cv_summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a two-video fusion classifier from frozen features extracted "
            "from existing single-video VideoMAE checkpoints."
        )
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_fusion_config(args.config)
    fold_summaries = [
        train_one_fusion_fold(config, fold_idx)
        for fold_idx in range(config.num_folds)
    ]
    summary = write_cv_summary(config, fold_summaries)
    print(
        f"5-fold video fusion training finished: mean_auc={summary['mean_auc']}, "
        f"std_auc={summary['std_auc']}"
    )


if __name__ == "__main__":
    main()
