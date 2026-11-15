from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class VideoStructuredModelConfig:
    model_name: str
    checkpoint_template: str
    hidden_dim: int
    structured_hidden_dim: int
    dropout: float


@dataclass
class VideoStructuredDataConfig:
    data_root: str
    num_frames: int
    image_size: int
    sampling_strategy: str
    num_workers: int
    feature_batch_size: int
    train_batch_size: int
    valid_batch_size: int
    video_columns: List[str]


@dataclass
class StructuredFeatureConfig:
    categorical_columns: List[str]
    numerical_columns: List[str]


@dataclass
class VideoStructuredOptimConfig:
    epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float


@dataclass
class VideoStructuredTrainConfig:
    use_amp: bool
    grad_accum_steps: int
    max_grad_norm: float
    log_interval: int
    early_stop_patience: int
    decision_threshold: float


@dataclass
class VideoStructuredAppConfig:
    seed: int
    num_folds: int
    folds_dir: str
    output_dir: str
    model: VideoStructuredModelConfig
    data: VideoStructuredDataConfig
    structured: StructuredFeatureConfig
    optim: VideoStructuredOptimConfig
    train: VideoStructuredTrainConfig


def load_video_structured_config(
    path: str | Path,
) -> VideoStructuredAppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return VideoStructuredAppConfig(
        seed=raw["seed"],
        num_folds=raw["num_folds"],
        folds_dir=raw["folds_dir"],
        output_dir=raw["output_dir"],
        model=VideoStructuredModelConfig(**raw["model"]),
        data=VideoStructuredDataConfig(**raw["data"]),
        structured=StructuredFeatureConfig(**raw["structured"]),
        optim=VideoStructuredOptimConfig(**raw["optim"]),
        train=VideoStructuredTrainConfig(**raw["train"]),
    )
