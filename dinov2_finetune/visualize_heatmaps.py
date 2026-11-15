import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data import AirwayImageDataset, build_single_column_table, find_dataset_dir
from src.model import DinoV2BinaryClassifier


class DinoV2GradCAM:
    """Grad-CAM for patch tokens entering the final DINOv2 self-attention block."""

    def __init__(self, model: DinoV2BinaryClassifier):
        layers = getattr(getattr(model.backbone, "encoder", None), "layer", None)
        if layers is None or len(layers) == 0:
            raise RuntimeError("Could not find DINOv2 encoder layers for Grad-CAM")
        self.activations = None
        self.gradients = None
        last_layer = layers[-1]
        target_layer = getattr(last_layer, "norm1", None)
        if target_layer is None:
            target_layer = getattr(last_layer, "layernorm_before", None)
        if target_layer is None:
            raise RuntimeError(
                "Could not find the pre-attention normalization in the final DINOv2 layer; "
                "hooking the post-block tokens would give zero patch-token gradients"
            )
        self.handle = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(self, _module, _inputs, output):
        hidden_states = output[0] if isinstance(output, (tuple, list)) else output
        self.activations = hidden_states
        self.gradients = None
        hidden_states.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient):
        self.gradients = gradient

    def heatmaps(self, patch_grid: tuple[int, int], special_tokens: int):
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM did not receive activations and gradients")
        grid_height, grid_width = patch_grid
        activations = self.activations[:, special_tokens:, :]
        gradients = self.gradients[:, special_tokens:, :]
        expected_patches = grid_height * grid_width
        if activations.shape[1] != expected_patches:
            raise RuntimeError(
                f"Got {activations.shape[1]} patch tokens, expected {expected_patches}; "
                "check image_size and DINOv2 patch size"
            )
        weights = gradients.mean(dim=1, keepdim=True)
        cams = torch.relu((activations * weights).sum(dim=-1))
        cams = cams.reshape(-1, grid_height, grid_width)
        minimum = cams.amin(dim=(1, 2), keepdim=True)
        maximum = cams.amax(dim=(1, 2), keepdim=True)
        cams = (cams - minimum) / (maximum - minimum).clamp_min(1e-8)
        return cams.detach().cpu().numpy()

    def close(self):
        self.handle.remove()

    def clear(self):
        self.activations = None
        self.gradients = None


def sanitize_name(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", str(value)).strip().strip(".") or "unnamed"


def get_patch_grid(model: DinoV2BinaryClassifier, image_size: int) -> tuple[tuple[int, int], int]:
    backbone_config = model.backbone.config
    patch_size = backbone_config.patch_size
    if isinstance(patch_size, (tuple, list)):
        patch_height, patch_width = patch_size
    else:
        patch_height = patch_width = patch_size
    if image_size % patch_height or image_size % patch_width:
        raise ValueError(f"image_size={image_size} must be divisible by patch size {patch_size}")
    grid = (image_size // patch_height, image_size // patch_width)
    special_tokens = 1 + int(getattr(backbone_config, "num_register_tokens", 0))
    return grid, special_tokens


def load_finetuned_model(config, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {checkpoint_path}")
    model = DinoV2BinaryClassifier(
        model_name=config.model.model_name,
        dropout=config.model.dropout,
        freeze_backbone=config.model.freeze_backbone,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def target_scores(logits: torch.Tensor, labels: torch.Tensor, target: str, threshold: float = 0.5):
    if target == "positive":
        return logits, np.ones(len(logits), dtype=np.int64)
    if target == "negative":
        return -logits, np.zeros(len(logits), dtype=np.int64)
    classes = labels.long() if target == "label" else (torch.sigmoid(logits) >= threshold).long()
    signs = classes.to(dtype=logits.dtype) * 2 - 1
    return logits * signs, classes.detach().cpu().numpy()


def save_heatmap(original_path: str, heatmap: np.ndarray, output_dir: Path, sample_index: int):
    with Image.open(original_path) as image_file:
        original = image_file.convert("RGB")
    width, height = original.size
    heatmap_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    resized = Image.fromarray(heatmap_u8).resize((width, height), resampling)
    heatmap = np.asarray(resized, dtype=np.float32) / 255.0
    red = np.clip(1.5 - np.abs(4.0 * heatmap - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * heatmap - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * heatmap - 1.0), 0.0, 1.0)
    heatmap_rgb = (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap_rgb)
    overlay = Image.blend(original, heatmap_image, alpha=0.4)
    stem = f"sample_{sample_index:06d}"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.jpg"
    heatmap_image.save(heatmap_path)
    overlay.save(overlay_path, quality=95)
    return str(heatmap_path), str(overlay_path)


def visualize_fold(args, config, fold_idx: int, device: torch.device):
    dataset_dir = find_dataset_dir(Path(args.data_root))
    train_csv = Path(config.folds_dir) / f"fold_{fold_idx}_train.csv"
    table = build_single_column_table(train_csv, dataset_dir, args.image_column)
    if args.limit is not None:
        table = table.iloc[:args.limit].reset_index(drop=True)

    checkpoint_path = (
        Path(config.output_dir) / sanitize_name(args.image_column) / f"fold_{fold_idx}" / "best.pt"
    )
    model = load_finetuned_model(config, checkpoint_path, device)
    gradcam = DinoV2GradCAM(model)
    patch_grid, special_tokens = get_patch_grid(model, config.data.image_size)
    dataset = AirwayImageDataset(table, image_size=config.data.image_size, is_train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config.data.valid_batch_size,
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else config.data.num_workers,
        pin_memory=device.type == "cuda",
    )

    output_dir = Path(args.output_dir or (Path(config.output_dir) / "heatmaps"))
    output_dir = output_dir / sanitize_name(args.image_column) / f"fold_{fold_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    offset = 0
    try:
        for batch in tqdm(loader, desc=f"Grad-CAM fold {fold_idx}"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            pixel_values.requires_grad_(True)
            labels = batch["label"].to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            logits = model(pixel_values)
            scores, classes = target_scores(
                logits, labels, args.target, config.train.decision_threshold
            )
            scores.sum().backward()
            maps = gradcam.heatmaps(patch_grid, special_tokens)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy().astype(int)
            gradcam.clear()
            del scores, logits, pixel_values, labels

            for batch_index, heatmap in enumerate(maps):
                row = table.iloc[offset + batch_index]
                heatmap_path, overlay_path = save_heatmap(
                    row["image_path"], heatmap, output_dir, offset + batch_index
                )
                records.append({
                    "sample_index": offset + batch_index,
                    "patient_id": row["patient_id"],
                    "image_path": row["image_path"],
                    "label": int(labels_np[batch_index]),
                    "probability": float(probs[batch_index]),
                    "predicted_label": int(probs[batch_index] >= config.train.decision_threshold),
                    "cam_target": int(classes[batch_index]),
                    "heatmap_path": heatmap_path,
                    "overlay_path": overlay_path,
                })
            offset += len(maps)
    finally:
        gradcam.close()
    pd.DataFrame(records).to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    print(f"Fold {fold_idx}: saved {len(records)} heatmaps to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps on training images with fine-tuned DINOv2 fold models."
    )
    parser.add_argument("--config", required=True, help="Training YAML config")
    parser.add_argument("--image-column", required=True, help="Image column name from the training config")
    parser.add_argument("--fold", default="all", help="Fold index or 'all' (default: all folds)")
    parser.add_argument("--target", choices=("predicted", "positive", "negative", "label"), default="predicted")
    parser.add_argument("--data-root", default="data", help="Root containing the image dataset directory")
    parser.add_argument("--output-dir", default=None, help="Output root (default: <output_dir>/heatmaps)")
    parser.add_argument("--batch-size", type=int, default=None, help="Grad-CAM batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--limit", type=int, default=None, help="Maximum training images per fold")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.fold.lower() == "all":
        folds = range(config.num_folds)
    else:
        fold_idx = int(args.fold)
        if not 0 <= fold_idx < config.num_folds:
            raise ValueError(f"fold must be between 0 and {config.num_folds - 1}, got {fold_idx}")
        folds = [fold_idx]
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be a positive integer")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    for fold_idx in folds:
        visualize_fold(args, config, fold_idx, device)


if __name__ == "__main__":
    main()

