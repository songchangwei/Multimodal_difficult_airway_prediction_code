"""Five-fold fusion of all configured images and videos without structured data."""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from dinov2_finetune.src.data import IMAGE_EXTENSIONS, find_dataset_dir, resolve_media_path
from dinov2_finetune.src.fusion_model import FrozenDinoV2Backbone
from videomae_finetune.src.data import VIDEO_EXTENSIONS, load_video_frames
from videomae_finetune.src.fusion_model import FrozenVideoMAEBackbone
from videomae_finetune.src.logger import append_metrics_csv, setup_logger, write_json
from videomae_finetune.src.metrics import binary_classification_metrics
from videomae_finetune.src.train import sanitize_name


# Older checkpoints were created when the training package was imported as
# top-level ``src``. Keep this alias so torch.load can deserialize them.
sys.modules.setdefault("src", __import__("dinov2_finetune.src", fromlist=["*"]))
sys.modules.setdefault(
    "src.config",
    __import__("dinov2_finetune.src.config", fromlist=["*"]),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def _resolve_valid_media(dataset_dir: Path, value, extensions: set[str]):
    if _is_missing(value):
        return None
    resolved = resolve_media_path(dataset_dir, str(value))
    if resolved is None or resolved.suffix.lower() not in extensions:
        return None
    return resolved


def _output_dir(config: dict, image_columns: list[str], video_columns: list[str]) -> Path:
    name = sanitize_name("_".join(image_columns + video_columns) + "_frozen_fusion")
    return Path(config["output_dir"]) / name


def validate_configuration(config: dict):
    """Check required media availability and checkpoints before feature extraction."""
    if int(config["num_folds"]) != 5:
        raise RuntimeError(
            "This experiment is defined as a five-fold experiment; num_folds must be 5."
        )

    image_columns = list(config["image_columns"])
    video_columns = list(config["video_columns"])
    if not image_columns or not video_columns:
        raise RuntimeError("image_columns and video_columns must both be non-empty.")

    dataset_dir = find_dataset_dir(Path(config["data_root"]))
    missing = []
    incomplete_splits = []

    for fold in range(config["num_folds"]):
        for split in ("train", "val"):
            csv_path = Path(config["folds_dir"]) / f"fold_{fold}_{split}.csv"
            if not csv_path.exists():
                missing.append(f"missing fold CSV: {csv_path}")
                continue

            table = pd.read_csv(csv_path)
            requested_columns = image_columns + video_columns
            absent_columns = [
                column for column in requested_columns if column not in table.columns
            ]
            if absent_columns:
                missing.append(f"{csv_path}: missing columns {absent_columns}")
                continue

            availability = {}
            for media_type, columns, extensions in (
                ("image", image_columns, IMAGE_EXTENSIONS),
                ("video", video_columns, VIDEO_EXTENSIONS),
            ):
                for column in columns:
                    available = [
                        _resolve_valid_media(dataset_dir, value, extensions) is not None
                        for value in table[column]
                    ]
                    availability[column] = available
                    valid_count = sum(available)
                    if valid_count == 0:
                        missing.append(
                            f"{csv_path}: {media_type} column {column} has no valid paths"
                        )

            complete_count = sum(
                all(availability[column][index] for column in requested_columns)
                for index in range(len(table))
            )
            if complete_count == 0:
                missing.append(
                    f"{csv_path}: no samples contain all requested image/video views"
                )
            elif complete_count != len(table):
                incomplete_splits.append(
                    f"{csv_path}: excluding {len(table) - complete_count}/"
                    f"{len(table)} incomplete samples"
                )

    for media_type, columns, template_key in (
        ("image", image_columns, "image_checkpoint_template"),
        ("video", video_columns, "video_checkpoint_template"),
    ):
        for column in columns:
            for fold in range(config["num_folds"]):
                checkpoint = Path(
                    config[template_key].format(
                        image_column=column,
                        video_column=column,
                        fold=fold,
                    )
                )
                if not checkpoint.exists():
                    missing.append(f"missing {media_type} checkpoint: {checkpoint}")

    if missing and config.get("require_all_media", True):
        details = "\n".join(f"  - {item}" for item in missing[:40])
        extra = f"\n  ... and {len(missing) - 40} more" if len(missing) > 40 else ""
        raise RuntimeError(
            "The requested all-image/all-video experiment cannot start because "
            "some required media or pretrained checkpoints are unavailable.\n"
            f"{details}{extra}"
        )

    return image_columns, video_columns, dataset_dir, incomplete_splits


def load_complete_table(
    csv_path: Path,
    dataset_dir: Path,
    image_columns: list[str],
    video_columns: list[str],
) -> pd.DataFrame:
    """Build complete media cases and deliberately omit structured columns."""
    source = pd.read_csv(csv_path)
    rows = []

    for _, row in source.iterrows():
        media = {}
        valid = True
        for index, column in enumerate(image_columns, start=1):
            resolved = _resolve_valid_media(dataset_dir, row.get(column), IMAGE_EXTENSIONS)
            if resolved is None:
                valid = False
                break
            media[f"image_path_{index}"] = str(resolved)

        if valid:
            for index, column in enumerate(video_columns, start=1):
                resolved = _resolve_valid_media(dataset_dir, row.get(column), VIDEO_EXTENSIONS)
                if resolved is None:
                    valid = False
                    break
                media[f"video_path_{index}"] = str(resolved)

        if not valid:
            continue

        rows.append(
            {
                "patient_id": str(row["patient_id"]),
                "label": int(row["label"]),
                "source_excel": str(row.get("source_excel", "")),
                "source_row": int(row.get("source_row", -1)),
                "fold": int(row.get("fold", -1)),
                **media,
            }
        )

    if not rows:
        raise RuntimeError(f"No complete image/video samples found in {csv_path}")
    return pd.DataFrame(rows)


class MediaDataset(Dataset):
    def __init__(
        self,
        table: pd.DataFrame,
        media_type: str,
        index: int,
        image_size: int,
        num_frames: int,
    ):
        self.table = table.reset_index(drop=True)
        self.media_type = media_type
        self.index = index
        self.image_size = image_size
        self.num_frames = num_frames
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        if self.media_type == "image":
            image = Image.open(row[f"image_path_{self.index}"]).convert("RGB")
            pixel_values = self.transform(image)
        else:
            pixel_values = load_video_frames(
                row[f"video_path_{self.index}"],
                num_frames=self.num_frames,
                image_size=self.image_size,
            )
        return {"pixel_values": pixel_values}


class MediaFusionClassifier(torch.nn.Module):
    """Trainable head for concatenated frozen image and video embeddings."""

    def __init__(
        self,
        image_dim: int,
        video_dim: int,
        num_images: int,
        num_videos: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.image_dim = image_dim
        self.video_dim = video_dim
        self.num_images = num_images
        self.num_videos = num_videos

        image_total = image_dim * num_images
        video_total = video_dim * num_videos
        self.image_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(image_total),
            torch.nn.Linear(image_total, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.video_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(video_total),
            torch.nn.Linear(video_total, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        image_features: list[torch.Tensor],
        video_features: list[torch.Tensor],
    ):
        image_embedding = self.image_projection(torch.cat(image_features, dim=1))
        video_embedding = self.video_projection(torch.cat(video_features, dim=1))
        return self.classifier(
            torch.cat([image_embedding, video_embedding], dim=1)
        ).squeeze(-1)


def autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(
        device_type="cuda",
        enabled=enabled and device.type == "cuda",
    )


def extract_features(
    extractor,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
    desc: str,
) -> torch.Tensor:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    result = []

    # 当前已经训练好的图片/视频模型只用于提取特征；参数冻结且不参与反向传播。
    extractor.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, leave=False, desc=desc):
            values = batch["pixel_values"].to(device, non_blocking=True)
            with autocast_context(device, use_amp):
                result.append(extractor(values).float().cpu())

    if not result:
        raise RuntimeError(f"No features were extracted for {desc}")
    return torch.cat(result, dim=0)


def load_media_features(
    config: dict,
    train_table: pd.DataFrame,
    val_table: pd.DataFrame,
    fold: int,
    device: torch.device,
    logger,
    image_columns: list[str],
    video_columns: list[str],
):
    train_images, val_images = [], []
    train_videos, val_videos = [], []
    image_dim = None
    video_dim = None

    for index, column in enumerate(image_columns, start=1):
        checkpoint = Path(
            config["image_checkpoint_template"].format(
                image_column=column,
                fold=fold,
            )
        )
        logger.info(f"loading frozen image extractor: {column} -> {checkpoint}")
        # 加载现有单图片训练模型的 backbone，并冻结后提取本路图片特征。
        extractor = FrozenDinoV2Backbone(
            config["image_model_name"],
            checkpoint,
        ).to(device)
        image_dim = image_dim or extractor.hidden_size
        if image_dim != extractor.hidden_size:
            raise RuntimeError("Image frozen feature extractors have inconsistent dimensions")
        train_images.append(
            extract_features(
                extractor,
                MediaDataset(
                    train_table,
                    "image",
                    index,
                    config["image_size"],
                    config["num_frames"],
                ),
                config["feature_batch_size"],
                config["num_workers"],
                device,
                config["use_amp"],
                f"image {index} train",
            )
        )
        val_images.append(
            extract_features(
                extractor,
                MediaDataset(
                    val_table,
                    "image",
                    index,
                    config["image_size"],
                    config["num_frames"],
                ),
                config["feature_batch_size"],
                config["num_workers"],
                device,
                config["use_amp"],
                f"image {index} val",
            )
        )
        del extractor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for index, column in enumerate(video_columns, start=1):
        checkpoint = Path(
            config["video_checkpoint_template"].format(
                video_column=column,
                fold=fold,
            )
        )
        logger.info(f"loading frozen video extractor: {column} -> {checkpoint}")
        # 加载现有单视频训练模型的 VideoMAE backbone，并冻结后提取本路视频特征。
        extractor = FrozenVideoMAEBackbone(
            config["video_model_name"],
            checkpoint,
        ).to(device)
        video_dim = video_dim or extractor.hidden_size
        if video_dim != extractor.hidden_size:
            raise RuntimeError("Video frozen feature extractors have inconsistent dimensions")
        train_videos.append(
            extract_features(
                extractor,
                MediaDataset(
                    train_table,
                    "video",
                    index,
                    config["image_size"],
                    config["num_frames"],
                ),
                config["feature_batch_size"],
                config["num_workers"],
                device,
                config["use_amp"],
                f"video {index} train",
            )
        )
        val_videos.append(
            extract_features(
                extractor,
                MediaDataset(
                    val_table,
                    "video",
                    index,
                    config["image_size"],
                    config["num_frames"],
                ),
                config["feature_batch_size"],
                config["num_workers"],
                device,
                config["use_amp"],
                f"video {index} val",
            )
        )
        del extractor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return (
        torch.cat(train_images, dim=1),
        torch.cat(val_images, dim=1),
        torch.cat(train_videos, dim=1),
        torch.cat(val_videos, dim=1),
        image_dim,
        video_dim,
    )


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def run_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    scaler,
    config,
    train: bool,
):
    model.train(train)
    labels, probs, losses = [], [], []
    context = torch.enable_grad if train else torch.no_grad

    with context():
        for image_features, video_features, y in loader:
            image_features = image_features.to(device, non_blocking=True)
            video_features = video_features.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            image_views = list(image_features.split(model.image_dim, dim=1))
            video_views = list(video_features.split(model.video_dim, dim=1))

            with autocast_context(device, config["use_amp"]):
                logits = model(image_views, video_views)
                loss = criterion(logits, y)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            losses.append(float(loss.item()))
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            labels.extend(y.detach().cpu().numpy().tolist())

    metrics = binary_classification_metrics(
        labels,
        probs,
        config["decision_threshold"],
    )
    metrics["loss"] = float(np.mean(losses))
    return metrics, probs


def train_fold(
    config: dict,
    fold: int,
    image_columns: list[str],
    video_columns: list[str],
    dataset_dir: Path,
):
    set_seed(config["seed"] + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = _output_dir(config, image_columns, video_columns) / f"fold_{fold}"
    logger = setup_logger(output)

    train_table = load_complete_table(
        Path(config["folds_dir"]) / f"fold_{fold}_train.csv",
        dataset_dir,
        image_columns,
        video_columns,
    )
    val_table = load_complete_table(
        Path(config["folds_dir"]) / f"fold_{fold}_val.csv",
        dataset_dir,
        image_columns,
        video_columns,
    )
    logger.info(
        f"complete train={len(train_table)} val={len(val_table)} "
        f"images={image_columns} videos={video_columns} structured_data=False"
    )

    (
        train_images,
        val_images,
        train_videos,
        val_videos,
        image_dim,
        video_dim,
    ) = load_media_features(
        config,
        train_table,
        val_table,
        fold,
        device,
        logger,
        image_columns,
        video_columns,
    )
    train_labels = torch.tensor(train_table["label"].to_numpy(dtype=np.float32))
    val_labels = torch.tensor(val_table["label"].to_numpy(dtype=np.float32))

    train_loader = DataLoader(
        TensorDataset(train_images, train_videos, train_labels),
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(val_images, val_videos, val_labels),
        batch_size=config["valid_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # 这里只训练媒体融合头；所有图片和视频 backbone 的参数已经冻结。
    model = MediaFusionClassifier(
        image_dim,
        video_dim,
        len(image_columns),
        len(video_columns),
        config["hidden_dim"],
        config["dropout"],
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer,
        math.ceil(len(train_loader) * config["epochs"]),
        config["warmup_ratio"],
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config["use_amp"] and device.type == "cuda",
    )

    best_auc, best_epoch, best_metrics, patience = -1.0, -1, None, 0
    for epoch in range(1, config["epochs"] + 1):
        train_metrics, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            scaler,
            config,
            True,
        )
        val_metrics, val_probs = run_epoch(
            model,
            val_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            scaler,
            config,
            False,
        )
        append_metrics_csv(
            output / "metrics.csv",
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            },
        )
        logger.info(
            f"epoch={epoch} train_auc={train_metrics['auc']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        auc = -1.0 if np.isnan(val_metrics["auc"]) else val_metrics["auc"]
        if auc > best_auc:
            best_auc, best_epoch, best_metrics, patience = auc, epoch, val_metrics, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "image_columns": image_columns,
                    "video_columns": video_columns,
                    "image_feature_dim": image_dim,
                    "video_feature_dim": video_dim,
                    "uses_structured_data": False,
                    "config": config,
                },
                output / "best.pt",
            )
            prediction = val_table.copy().reset_index(drop=True)
            prediction["prob"] = val_probs
            prediction["prediction"] = (
                prediction["prob"] >= config["decision_threshold"]
            ).astype(int)
            prediction.to_csv(
                output / "val_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )
        else:
            patience += 1

        if patience >= config["early_stop_patience"]:
            logger.info(f"early stop at epoch={epoch}")
            break

    summary = {
        "fold": fold,
        "image_columns": image_columns,
        "video_columns": video_columns,
        "uses_structured_data": False,
        "frozen_image_and_video_backbones": True,
        "best_epoch": best_epoch,
        "best_auc": best_auc,
        "best_metrics": best_metrics,
        "train_samples": len(train_table),
        "val_samples": len(val_table),
        "num_images": len(image_columns),
        "num_videos": len(video_columns),
        "image_feature_dim": image_dim,
        "video_feature_dim": video_dim,
        "device": str(device),
    }
    write_json(output / "metrics.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    try:
        image_columns, video_columns, dataset_dir, incomplete_splits = validate_configuration(
            config
        )
    except Exception as exc:
        print(f"Preflight failed; no training was started:\n{exc}", file=sys.stderr)
        return 2

    for note in incomplete_splits:
        print(f"Preflight note: {note}")

    summaries = [
        train_fold(config, fold, image_columns, video_columns, dataset_dir)
        for fold in range(config["num_folds"])
    ]
    output = _output_dir(config, image_columns, video_columns)
    aucs = [item["best_auc"] for item in summaries if item["best_auc"] >= 0]
    summary = {
        "image_columns": image_columns,
        "video_columns": video_columns,
        "uses_structured_data": False,
        "frozen_image_and_video_backbones": True,
        "folds": summaries,
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "std_auc": float(np.std(aucs)) if aucs else None,
    }
    write_json(output / "cv_summary.json", summary)
    pd.DataFrame(
        [
            {
                "fold": item["fold"],
                "best_epoch": item["best_epoch"],
                "best_auc": item["best_auc"],
                "train_samples": item["train_samples"],
                "val_samples": item["val_samples"],
                "num_images": item["num_images"],
                "num_videos": item["num_videos"],
                "image_feature_dim": item["image_feature_dim"],
                "video_feature_dim": item["video_feature_dim"],
                "device": item["device"],
            }
            for item in summaries
        ]
    ).to_csv(output / "cv_summary.csv", index=False, encoding="utf-8-sig")
    print(
        "5-fold all-image/all-video frozen fusion finished: "
        f"mean_auc={summary['mean_auc']}, std_auc={summary['std_auc']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
