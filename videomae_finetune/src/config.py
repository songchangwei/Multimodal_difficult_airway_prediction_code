from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class ModelConfig:
    model_name: str
    dropout: float
    freeze_backbone: bool


@dataclass
class DataConfig:
    num_frames: int
    image_size: int
    sampling_strategy: str
    num_workers: int
    batch_size: int
    valid_batch_size: int
    train_video_columns: List[str]


@dataclass
class OptimConfig:
    epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float


@dataclass
class TrainConfig:
    use_amp: bool
    grad_accum_steps: int
    max_grad_norm: float
    log_interval: int
    early_stop_patience: int
    decision_threshold: float


@dataclass
class AppConfig:
    seed: int
    num_folds: int
    folds_dir: str
    output_dir: str
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    train: TrainConfig


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        seed=raw["seed"],
        num_folds=raw["num_folds"],
        folds_dir=raw["folds_dir"],
        output_dir=raw["output_dir"],
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        optim=OptimConfig(**raw["optim"]),
        train=TrainConfig(**raw["train"]),
    )
