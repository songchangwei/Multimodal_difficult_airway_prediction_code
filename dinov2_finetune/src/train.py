import math
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import AirwayImageDataset, build_single_column_table, find_dataset_dir
from .logger import append_metrics_csv, setup_logger, write_json
from .metrics import binary_classification_metrics
from .model import DinoV2BinaryClassifier, build_criterion


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, optimizer, scheduler, criterion, device, scaler, config, logger, epoch, is_train):
    model.train(is_train)
    running_loss = 0.0
    labels_all = []
    probs_all = []

    context = torch.enable_grad if is_train else torch.no_grad
    with context():
        for step, batch in enumerate(tqdm(loader, leave=False), start=1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=config.train.use_amp and device.type == "cuda"):
                logits = model(pixel_values)
                loss = criterion(logits, labels)
                if is_train:
                    loss = loss / config.train.grad_accum_steps

            if is_train:
                scaler.scale(loss).backward()
                if step % config.train.grad_accum_steps == 0:
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
            running_loss += float(loss.item()) * (config.train.grad_accum_steps if is_train else 1.0)

            if is_train and step % config.train.log_interval == 0:
                logger.info(
                    f"epoch={epoch} step={step}/{len(loader)} "
                    f"loss={running_loss / step:.6f} lr={scheduler.get_last_lr()[0]:.8f}"
                )

    metrics = binary_classification_metrics(
        labels=labels_all,
        probs=probs_all,
        threshold=config.train.decision_threshold,
    )
    metrics["loss"] = running_loss / max(1, len(loader))
    return metrics, labels_all, probs_all


def save_predictions(path: Path, table: pd.DataFrame, probs: list[float]):
    out = table.copy().reset_index(drop=True)
    out["prob"] = probs
    out.to_csv(path, index=False, encoding="utf-8-sig")


def sanitize_name(name: str) -> str:
    return re.sub(r'[<>:"/\\\\|?*]+', "_", name).strip()


def train_one_fold(config, fold_idx: int, image_column: str):
    set_seed(config.seed + fold_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds_dir = Path(config.folds_dir)
    output_dir = Path(config.output_dir) / sanitize_name(image_column) / f"fold_{fold_idx}"
    logger = setup_logger(output_dir)

    dataset_dir = find_dataset_dir(Path("data"))
    train_csv = folds_dir / f"fold_{fold_idx}_train.csv"
    val_csv = folds_dir / f"fold_{fold_idx}_val.csv"

    train_table = build_single_column_table(train_csv, dataset_dir, image_column)
    val_table = build_single_column_table(val_csv, dataset_dir, image_column)
    logger.info(
        f"image_column={image_column} dataset_dir={dataset_dir} "
        f"train_images={len(train_table)} val_images={len(val_table)}"
    )

    train_ds = AirwayImageDataset(train_table, image_size=config.data.image_size, is_train=True)
    val_ds = AirwayImageDataset(val_table, image_size=config.data.image_size, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.data.valid_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    model = DinoV2BinaryClassifier(
        model_name=config.model.model_name,
        dropout=config.model.dropout,
        freeze_backbone=config.model.freeze_backbone,
    ).to(device)
    criterion = build_criterion()
    optimizer = AdamW(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    total_steps = math.ceil(len(train_loader) / config.train.grad_accum_steps) * config.optim.epochs
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_ratio=config.optim.warmup_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=config.train.use_amp and device.type == "cuda")

    best_auc = -1.0
    best_epoch = -1
    patience = 0
    best_metrics = None

    for epoch in range(1, config.optim.epochs + 1):
        train_metrics, _, _ = run_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler, config, logger, epoch, True
        )
        val_metrics, val_labels, val_probs = run_epoch(
            model, val_loader, optimizer, scheduler, criterion, device, scaler, config, logger, epoch, False
        )

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        append_metrics_csv(output_dir / "metrics.csv", row)
        logger.info(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} train_auc={train_metrics['auc']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} val_auc={val_metrics['auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_sens={val_metrics['sensitivity']:.4f} "
            f"val_spec={val_metrics['specificity']:.4f}"
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
                    "config": config.__dict__,
                },
                output_dir / "best.pt",
            )
            save_predictions(output_dir / "val_predictions.csv", val_table, val_probs)
        else:
            patience += 1

        if patience >= config.train.early_stop_patience:
            logger.info(f"early stop at epoch={epoch}")
            break

    summary = {
        "image_column": image_column,
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_auc": best_auc,
        "best_metrics": best_metrics,
        "train_images": len(train_table),
        "val_images": len(val_table),
        "device": str(device),
    }
    write_json(output_dir / "metrics.json", summary)
    return summary
