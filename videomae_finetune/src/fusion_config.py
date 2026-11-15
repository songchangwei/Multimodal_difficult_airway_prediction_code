from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class FeatureModelConfig:
    model_name: str
    checkpoint_template: str
    hidden_dim: int
    dropout: float


@dataclass
class FusionDataConfig:
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
class FusionOptimConfig:
    epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float


@dataclass
class FusionTrainConfig:
    use_amp: bool
    grad_accum_steps: int
    max_grad_norm: float
    log_interval: int
    early_stop_patience: int
    decision_threshold: float


@dataclass
class FusionAppConfig:
    seed: int
    num_folds: int
    folds_dir: str
    output_dir: str
    model: FeatureModelConfig
    data: FusionDataConfig
    optim: FusionOptimConfig
    train: FusionTrainConfig


def load_fusion_config(path: str | Path) -> FusionAppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return FusionAppConfig(
        seed=raw["seed"],
        num_folds=raw["num_folds"],
        folds_dir=raw["folds_dir"],
        output_dir=raw["output_dir"],
        model=FeatureModelConfig(**raw["model"]),
        data=FusionDataConfig(**raw["data"]),
        optim=FusionOptimConfig(**raw["optim"]),
        train=FusionTrainConfig(**raw["train"]),
    )
