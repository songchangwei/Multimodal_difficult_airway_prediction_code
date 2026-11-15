import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import dump
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dinov2_finetune.src.data import IMAGE_EXTENSIONS, find_dataset_dir, resolve_media_path
from dinov2_finetune.src.fusion_model import FrozenDinoV2Backbone
from dinov2_finetune.src.multimodal_data import (
    build_structured_preprocessor,
    prepare_structured_features,
)
from videomae_finetune.src.data import VIDEO_EXTENSIONS, load_video_frames
from videomae_finetune.src.fusion_model import FrozenVideoMAEBackbone
from videomae_finetune.src.logger import append_metrics_csv, setup_logger, write_json
from videomae_finetune.src.metrics import binary_classification_metrics
from videomae_finetune.src.train import sanitize_name

# Older checkpoints were saved when the training package was imported as
# top-level ``src``. Register the current package under that name so torch.load
# can deserialize their optimizer/config metadata.
sys.modules.setdefault("src", __import__("dinov2_finetune.src", fromlist=["*"]))
sys.modules.setdefault(
    "src.config",
    __import__("dinov2_finetune.src.config", fromlist=["*"]),
)


DEFAULT_IMAGE_COLUMNS = [
    "外观检查_1",
    "外观检查_2",
    "颈围、甲颏距离和胸颏距离_1",
    "颈围、甲颏距离和胸颏距离_2",
    "张口度_1",
    "上唇咬合试验_1",
    "牙齿检查_1",
    "Mallampati分级_1",
]
DEFAULT_VIDEO_COLUMNS = [
    "前屈和后仰_1",
    "前屈和后仰_2",
    "左右旋转_1",
    "左右旋转_2",
    "侧屈_1",
    "侧屈_2",
]
STRUCTURED_CATEGORICAL = ["性别", "种族", "age_group", "bmi_group"]
STRUCTURED_NUMERICAL = [
    "age",
    "height_cm",
    "weight_kg",
    "BMI",
    "age_squared",
    "bmi_squared",
    "age_bmi",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_complete_table(csv_path, dataset_dir, image_columns, video_columns):
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        media = {}
        valid = True
        for column in image_columns:
            resolved = resolve_media_path(dataset_dir, str(row.get(column, "")))
            if resolved is None or resolved.suffix.lower() not in IMAGE_EXTENSIONS:
                valid = False
                break
            media[f"image_path_{len([k for k in media if k.startswith('image_path_')]) + 1}"] = str(resolved)
        if valid:
            for column in video_columns:
                resolved = resolve_media_path(dataset_dir, str(row.get(column, "")))
                if resolved is None or resolved.suffix.lower() not in VIDEO_EXTENSIONS:
                    valid = False
                    break
                media[f"video_path_{len([k for k in media if k.startswith('video_path_')]) + 1}"] = str(resolved)
        if not valid:
            continue
        sample = row.to_dict()
        sample["patient_id"] = str(row["patient_id"])
        sample["label"] = int(row["label"])
        sample.update(media)
        rows.append(sample)
    if not rows:
        raise RuntimeError(f"No complete image/video samples found in {csv_path}")
    return pd.DataFrame(rows)


def validate_media_configuration(config):
    """Fail before feature extraction when the requested six-video experiment is incomplete."""
    image_columns = config.get("image_columns", DEFAULT_IMAGE_COLUMNS)
    video_columns = config.get("video_columns", DEFAULT_VIDEO_COLUMNS)
    missing = []
    for fold in range(config["num_folds"]):
        train_csv = Path(config["folds_dir"]) / f"fold_{fold}_train.csv"
        val_csv = Path(config["folds_dir"]) / f"fold_{fold}_val.csv"
        for csv_path in (train_csv, val_csv):
            table = pd.read_csv(csv_path)
            for column in video_columns:
                if column not in table.columns:
                    missing.append(f"{csv_path}: missing column {column}")
                elif table[column].fillna("").astype(str).str.strip().eq("").all():
                    missing.append(f"{csv_path}: no paths in {column}")
        for column in video_columns:
            for fold_path in range(config["num_folds"]):
                checkpoint = Path(
                    config["video_checkpoint_template"].format(
                        video_column=column,
                        fold=fold_path,
                    )
                )
                if not checkpoint.exists():
                    missing.append(f"missing checkpoint: {checkpoint}")
    if missing and config.get("require_all_videos", True):
        details = "\n".join(f"  - {item}" for item in missing[:30])
        extra = f"\n  ... and {len(missing) - 30} more" if len(missing) > 30 else ""
        raise RuntimeError(
            "The requested all-six-video experiment cannot start because "
            "some video data/checkpoints are unavailable.\n"
            f"{details}{extra}"
        )
    return image_columns, video_columns


class MediaDataset(torch.utils.data.Dataset):
    def __init__(self, table, media_type, index, image_size, num_frames):
        self.table = table.reset_index(drop=True)
        self.media_type = media_type
        self.index = index
        self.image_size = image_size
        self.num_frames = num_frames

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        if self.media_type == "image":
            from PIL import Image
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])
            pixel_values = transform(Image.open(row[f"image_path_{self.index}"]).convert("RGB"))
        else:
            pixel_values = load_video_frames(
                row[f"video_path_{self.index}"],
                num_frames=self.num_frames,
                image_size=self.image_size,
            )
        return {"pixel_values": pixel_values}


class FusionClassifier(torch.nn.Module):
    def __init__(self, image_dim, video_dim, num_images, num_videos, structured_dim, hidden_dim=256, structured_hidden_dim=64, dropout=0.2):
        super().__init__()
        self.image_dim = image_dim
        self.video_dim = video_dim
        self.num_images = num_images
        self.num_videos = num_videos
        self.structured_dim = structured_dim
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
        self.structured_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(structured_dim),
            torch.nn.Linear(structured_dim, structured_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 + structured_hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, image_features, video_features, structured_features):
        image_embedding = self.image_projection(torch.cat(list(image_features), dim=1))
        video_embedding = self.video_projection(torch.cat(list(video_features), dim=1))
        structured_embedding = self.structured_projection(structured_features)
        return self.classifier(
            torch.cat([image_embedding, video_embedding, structured_embedding], dim=1)
        ).squeeze(-1)


def extract_features(extractor, dataset, batch_size, num_workers, device, use_amp, desc):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    result = []
    extractor.eval()
    # 已训练好的图像/视频模型只用于提取特征，参数冻结，不参与反向传播。
    with torch.inference_mode():
        for batch in tqdm(loader, leave=False, desc=desc):
            values = batch["pixel_values"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                result.append(extractor(values).float().cpu())
    return torch.cat(result, dim=0)


def load_media_features(config, train_table, val_table, fold, device, logger, image_columns, video_columns):
    train_parts, val_parts = [], []
    image_dim = video_dim = None
    for idx, column in enumerate(image_columns, 1):
        checkpoint = Path(config["image_checkpoint_template"].format(image_column=column, fold=fold))
        logger.info(f"loading frozen image extractor: {column} -> {checkpoint}")
        extractor = FrozenDinoV2Backbone(config["image_model_name"], checkpoint).to(device)
        image_dim = image_dim or extractor.hidden_size
        train_parts.append(extract_features(extractor, MediaDataset(train_table, "image", idx, config["image_size"], config["num_frames"]), config["feature_batch_size"], config["num_workers"], device, config["use_amp"], f"image {idx} train"))
        val_parts.append(extract_features(extractor, MediaDataset(val_table, "image", idx, config["image_size"], config["num_frames"]), config["feature_batch_size"], config["num_workers"], device, config["use_amp"], f"image {idx} val"))
        del extractor
        torch.cuda.empty_cache() if device.type == "cuda" else None
    for idx, column in enumerate(video_columns, 1):
        checkpoint = Path(config["video_checkpoint_template"].format(video_column=column, fold=fold))
        logger.info(f"loading frozen video extractor: {column} -> {checkpoint}")
        extractor = FrozenVideoMAEBackbone(config["video_model_name"], checkpoint).to(device)
        video_dim = video_dim or extractor.hidden_size
        train_parts.append(extract_features(extractor, MediaDataset(train_table, "video", idx, config["image_size"], config["num_frames"]), config["feature_batch_size"], config["num_workers"], device, config["use_amp"], f"video {idx} train"))
        val_parts.append(extract_features(extractor, MediaDataset(val_table, "video", idx, config["image_size"], config["num_frames"]), config["feature_batch_size"], config["num_workers"], device, config["use_amp"], f"video {idx} val"))
        del extractor
        torch.cuda.empty_cache() if device.type == "cuda" else None
    image_count = len(image_columns)
    return (
        torch.cat(train_parts[:image_count], dim=1),
        torch.cat(val_parts[:image_count], dim=1),
        torch.cat(train_parts[image_count:], dim=1),
        torch.cat(val_parts[image_count:], dim=1),
        image_dim,
        video_dim,
    )


def scheduler(optimizer, total_steps, warmup_ratio):
    warmup = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, optimizer, sched, criterion, device, scaler, config, train):
    model.train(train)
    labels, probs, losses = [], [], []
    context = torch.enable_grad if train else torch.no_grad
    with context():
        for image_features, video_features, structured_features, y in loader:
            image_features = image_features.to(device)
            video_features = video_features.to(device)
            structured_features = structured_features.to(device)
            y = y.to(device)
            image_views = image_features.split(model.image_dim, dim=1)
            video_views = video_features.split(model.video_dim, dim=1)
            with torch.amp.autocast("cuda", enabled=config["use_amp"] and device.type == "cuda"):
                logits = model(image_views, video_views, structured_features)
                loss = criterion(logits, y)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                sched.step()
            losses.append(float(loss.item()))
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            labels.extend(y.detach().cpu().numpy().tolist())
    metrics = binary_classification_metrics(labels, probs, config["decision_threshold"])
    metrics["loss"] = float(np.mean(losses))
    return metrics, probs


def train_fold(config, fold):
    set_seed(config["seed"] + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_columns = config.get("image_columns", DEFAULT_IMAGE_COLUMNS)
    video_columns = config.get("video_columns", DEFAULT_VIDEO_COLUMNS)
    out = Path(config["output_dir"]) / sanitize_name(
        "_".join(image_columns + video_columns) + "_structured_frozen_fusion"
    ) / f"fold_{fold}"
    logger = setup_logger(out)
    dataset_dir = find_dataset_dir(Path(config["data_root"]))
    train_table = load_complete_table(Path(config["folds_dir"]) / f"fold_{fold}_train.csv", dataset_dir, image_columns, video_columns)
    val_table = load_complete_table(Path(config["folds_dir"]) / f"fold_{fold}_val.csv", dataset_dir, image_columns, video_columns)
    logger.info(f"complete train={len(train_table)} val={len(val_table)} images={image_columns} videos={video_columns}")
    train_images, val_images, train_videos, val_videos, image_dim, video_dim = load_media_features(config, train_table, val_table, fold, device, logger, image_columns, video_columns)
    train_x, train_y, _ = prepare_structured_features(train_table, STRUCTURED_CATEGORICAL, STRUCTURED_NUMERICAL)
    val_x, val_y, val_raw = prepare_structured_features(val_table, STRUCTURED_CATEGORICAL, STRUCTURED_NUMERICAL)
    preprocessor = build_structured_preprocessor(STRUCTURED_CATEGORICAL, STRUCTURED_NUMERICAL)
    train_structured = torch.tensor(np.asarray(preprocessor.fit_transform(train_x)), dtype=torch.float32)
    val_structured = torch.tensor(np.asarray(preprocessor.transform(val_x)), dtype=torch.float32)
    dump(preprocessor, out / "structured_preprocessor.joblib")
    (out / "structured_feature_names.json").write_text(json.dumps(preprocessor.get_feature_names_out().tolist(), ensure_ascii=False, indent=2), encoding="utf-8")
    train_labels = torch.tensor(train_y.to_numpy(dtype=np.float32))
    val_labels = torch.tensor(val_y.to_numpy(dtype=np.float32))
    train_loader = DataLoader(TensorDataset(train_images, train_videos, train_structured, train_labels), batch_size=config["train_batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images, val_videos, val_structured, val_labels), batch_size=config["valid_batch_size"], shuffle=False)
    model = FusionClassifier(image_dim, video_dim, len(image_columns), len(video_columns), train_structured.shape[1], config["hidden_dim"], config["structured_hidden_dim"], config["dropout"]).to(device)
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    sched = scheduler(optimizer, math.ceil(len(train_loader) * config["epochs"]), config["warmup_ratio"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["use_amp"] and device.type == "cuda")
    criterion = torch.nn.BCEWithLogitsLoss()
    best_auc, best_epoch, best_metrics, patience = -1.0, -1, None, 0
    for epoch in range(1, config["epochs"] + 1):
        train_metrics, _ = run_epoch(model, train_loader, optimizer, sched, criterion, device, scaler, config, True)
        val_metrics, val_probs = run_epoch(model, val_loader, optimizer, sched, criterion, device, scaler, config, False)
        append_metrics_csv(out / "metrics.csv", {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        logger.info(f"epoch={epoch} train_auc={train_metrics['auc']:.4f} val_auc={val_metrics['auc']:.4f} val_f1={val_metrics['f1']:.4f}")
        auc = -1.0 if np.isnan(val_metrics["auc"]) else val_metrics["auc"]
        if auc > best_auc:
            best_auc, best_epoch, best_metrics, patience = auc, epoch, val_metrics, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "image_columns": image_columns, "video_columns": video_columns, "image_feature_dim": image_dim, "video_feature_dim": video_dim, "structured_dim": train_structured.shape[1], "config": config}, out / "best.pt")
            prediction = val_raw.copy().reset_index(drop=True)
            prediction["prob"] = val_probs
            prediction["prediction"] = (prediction["prob"] >= config["decision_threshold"]).astype(int)
            prediction.to_csv(out / "val_predictions.csv", index=False, encoding="utf-8-sig")
        else:
            patience += 1
        if patience >= config["early_stop_patience"]:
            logger.info(f"early stop at epoch={epoch}")
            break
    return {"fold": fold, "best_epoch": best_epoch, "best_auc": best_auc, "best_metrics": best_metrics, "train_samples": len(train_table), "val_samples": len(val_table), "num_images": len(image_columns), "num_videos": len(video_columns), "image_feature_dim": image_dim, "video_feature_dim": video_dim, "structured_dim": train_structured.shape[1], "device": str(device)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    image_columns, video_columns = validate_media_configuration(config)
    summaries = [train_fold(config, fold) for fold in range(config["num_folds"])]
    out = Path(config["output_dir"]) / sanitize_name("_".join(image_columns + video_columns) + "_structured_frozen_fusion")
    summary = {"image_columns": image_columns, "video_columns": video_columns, "folds": summaries, "mean_auc": float(np.mean([x["best_auc"] for x in summaries])), "std_auc": float(np.std([x["best_auc"] for x in summaries]))}
    write_json(out / "cv_summary.json", summary)
    pd.DataFrame([{"fold": x["fold"], "best_epoch": x["best_epoch"], "best_auc": x["best_auc"], "train_samples": x["train_samples"], "val_samples": x["val_samples"], "num_images": x["num_images"], "num_videos": x["num_videos"], "image_feature_dim": x["image_feature_dim"], "video_feature_dim": x["video_feature_dim"], "structured_dim": x["structured_dim"], "device": x["device"]} for x in summaries]).to_csv(out / "cv_summary.csv", index=False, encoding="utf-8-sig")
    print(f"5-fold all-media multimodal training finished: mean_auc={summary['mean_auc']}, std_auc={summary['std_auc']}")


if __name__ == "__main__":
    main()
