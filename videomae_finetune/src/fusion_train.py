import math
import random
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import find_dataset_dir
from .fusion_config import FusionAppConfig
from .fusion_data import PairedAirwayVideoDataset, build_pair_table
from .fusion_model import FrozenVideoMAEBackbone, TwoVideoFusionClassifier
from .logger import append_metrics_csv, setup_logger, write_json
from .metrics import binary_classification_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_name(name: str) -> str:
    return re.sub(r'[<>:"/\\\\|?*]+', "_", name).strip()


def fusion_output_dir(config: FusionAppConfig) -> Path:
    return Path(config.output_dir) / sanitize_name(
        "_".join(config.data.video_columns) + "_frozen_fusion"
    )


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def extract_table_features(
    extractor: FrozenVideoMAEBackbone,
    table: pd.DataFrame,
    config: FusionAppConfig,
    device: torch.device,
    video_view: int,
) -> torch.Tensor:
    dataset = PairedAirwayVideoDataset(
        table=table,
        num_frames=config.data.num_frames,
        image_size=config.data.image_size,
        video_view=video_view,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.data.feature_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )

    features = []
    extractor.eval()
    with torch.inference_mode():
        for batch in tqdm(
            loader,
            leave=False,
            desc=f"extract video {video_view}",
        ):
            pixel_values = batch["pixel_values"].to(
                device,
                non_blocking=True,
            )
            with torch.amp.autocast(
                "cuda",
                enabled=config.train.use_amp and device.type == "cuda",
            ):
                batch_features = extractor(pixel_values)
            features.append(batch_features.float().cpu())
    return torch.cat(features, dim=0)


def load_features_for_fold(
    config: FusionAppConfig,
    train_table: pd.DataFrame,
    val_table: pd.DataFrame,
    fold_idx: int,
    device: torch.device,
    logger,
):
    feature_tensors = []
    feature_dim = None

    for video_view, video_column in enumerate(config.data.video_columns, start=1):
        checkpoint_path = Path(
            config.model.checkpoint_template.format(
                video_column=video_column,
                fold=fold_idx,
            )
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Existing single-video checkpoint was not found: {checkpoint_path}"
            )

        logger.info(
            f"loading frozen feature extractor: video_view={video_view} "
            f"video_column={video_column} checkpoint={checkpoint_path}"
        )
        extractor = FrozenVideoMAEBackbone(
            model_name=config.model.model_name,
            checkpoint_path=checkpoint_path,
        ).to(device)
        if feature_dim is None:
            feature_dim = extractor.hidden_size
        elif feature_dim != extractor.hidden_size:
            raise ValueError("The two frozen video extractors have different dimensions")

        train_features = extract_table_features(
            extractor,
            train_table,
            config,
            device,
            video_view,
        )
        val_features = extract_table_features(
            extractor,
            val_table,
            config,
            device,
            video_view,
        )
        feature_tensors.append((train_features, val_features))

        del extractor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_features = torch.cat([item[0] for item in feature_tensors], dim=1)
    val_features = torch.cat([item[1] for item in feature_tensors], dim=1)
    return train_features, val_features, feature_dim


def run_head_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    scaler,
    config,
    logger,
    epoch,
    is_train,
):
    model.train(is_train)
    running_loss = 0.0
    labels_all = []
    probs_all = []

    context = torch.enable_grad if is_train else torch.no_grad
    with context():
        for step, (features, labels) in enumerate(loader, start=1):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feature_1, feature_2 = features.chunk(2, dim=1)

            with torch.amp.autocast(
                "cuda",
                enabled=config.train.use_amp and device.type == "cuda",
            ):
                logits = model(feature_1, feature_2)
                loss = criterion(logits, labels)
                loss_for_backward = (
                    loss / config.train.grad_accum_steps if is_train else loss
                )

            if is_train:
                scaler.scale(loss_for_backward).backward()
                should_step = (
                    step % config.train.grad_accum_steps == 0
                    or step == len(loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            probs_all.extend(probs.tolist())
            labels_all.extend(labels_np.tolist())
            running_loss += float(loss.item())

            if is_train and step % config.train.log_interval == 0:
                logger.info(
                    f"epoch={epoch} step={step}/{len(loader)} "
                    f"loss={running_loss / step:.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.8f}"
                )

    metrics = binary_classification_metrics(
        labels=labels_all,
        probs=probs_all,
        threshold=config.train.decision_threshold,
    )
    metrics["loss"] = running_loss / max(1, len(loader))
    return metrics, labels_all, probs_all


def save_predictions(path: Path, table: pd.DataFrame, probs: list[float]):
    output = table.copy().reset_index(drop=True)
    output["prob"] = probs
    output.to_csv(path, index=False, encoding="utf-8-sig")


def train_one_fusion_fold(config: FusionAppConfig, fold_idx: int):
    set_seed(config.seed + fold_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = fusion_output_dir(config) / f"fold_{fold_idx}"
    logger = setup_logger(output_dir)

    folds_dir = Path(config.folds_dir)
    dataset_dir = find_dataset_dir(Path(config.data.data_root))
    train_table = build_pair_table(
        folds_dir / f"fold_{fold_idx}_train.csv",
        dataset_dir,
        config.data.video_columns,
    )
    val_table = build_pair_table(
        folds_dir / f"fold_{fold_idx}_val.csv",
        dataset_dir,
        config.data.video_columns,
    )
    logger.info(
        f"video_columns={config.data.video_columns} dataset_dir={dataset_dir} "
        f"train_pairs={len(train_table)} val_pairs={len(val_table)}"
    )

    train_features, val_features, feature_dim = load_features_for_fold(
        config=config,
        train_table=train_table,
        val_table=val_table,
        fold_idx=fold_idx,
        device=device,
        logger=logger,
    )
    train_labels = torch.tensor(
        train_table["label"].to_numpy(dtype=np.float32)
    )
    val_labels = torch.tensor(val_table["label"].to_numpy(dtype=np.float32))

    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=config.data.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(val_features, val_labels),
        batch_size=config.data.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = TwoVideoFusionClassifier(
        feature_dim=feature_dim,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )
    total_steps = (
        math.ceil(len(train_loader) / config.train.grad_accum_steps)
        * config.optim.epochs
    )
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=config.optim.warmup_ratio,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.train.use_amp and device.type == "cuda",
    )
    optimizer.zero_grad(set_to_none=True)

    best_auc = -1.0
    best_epoch = -1
    patience = 0
    best_metrics = None

    for epoch in range(1, config.optim.epochs + 1):
        train_metrics, _, _ = run_head_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            scaler,
            config,
            logger,
            epoch,
            True,
        )
        val_metrics, _, val_probs = run_head_epoch(
            model,
            val_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            scaler,
            config,
            logger,
            epoch,
            False,
        )

        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        append_metrics_csv(output_dir / "metrics.csv", row)
        logger.info(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"train_auc={train_metrics['auc']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        metric_for_selection = val_metrics["auc"]
        if np.isnan(metric_for_selection):
            metric_for_selection = -1.0

        if metric_for_selection > best_auc:
            best_auc = metric_for_selection
            best_epoch = epoch
            patience = 0
            best_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "feature_dim": feature_dim,
                    "video_columns": config.data.video_columns,
                    "frozen_checkpoint_template": config.model.checkpoint_template,
                    "config": asdict(config),
                },
                output_dir / "best.pt",
            )
            save_predictions(
                output_dir / "val_predictions.csv",
                val_table,
                val_probs,
            )
        else:
            patience += 1

        if patience >= config.train.early_stop_patience:
            logger.info(f"early stop at epoch={epoch}")
            break

    summary = {
        "video_columns": config.data.video_columns,
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_auc": best_auc,
        "best_metrics": best_metrics,
        "train_pairs": len(train_table),
        "val_pairs": len(val_table),
        "feature_dim_per_video": feature_dim,
        "device": str(device),
    }
    write_json(output_dir / "metrics.json", summary)
    return summary


def write_cv_summary(config: FusionAppConfig, fold_summaries: list[dict]):
    output_dir = fusion_output_dir(config)
    valid_aucs = [
        item["best_auc"]
        for item in fold_summaries
        if item["best_auc"] >= 0
    ]
    summary = {
        "video_columns": config.data.video_columns,
        "folds": fold_summaries,
        "mean_auc": float(np.mean(valid_aucs)) if valid_aucs else None,
        "std_auc": float(np.std(valid_aucs)) if valid_aucs else None,
    }
    write_json(output_dir / "cv_summary.json", summary)

    rows = []
    for item in fold_summaries:
        row = {
            "fold": item["fold"],
            "best_epoch": item["best_epoch"],
            "best_auc": item["best_auc"],
            "train_pairs": item["train_pairs"],
            "val_pairs": item["val_pairs"],
            "feature_dim_per_video": item["feature_dim_per_video"],
            "device": item["device"],
        }
        for key, value in (item.get("best_metrics") or {}).items():
            row[f"best_{key}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        output_dir / "cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return summary
