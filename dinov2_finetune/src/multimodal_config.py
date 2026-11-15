from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class MultiModalModelConfig:
    model_name: str
    checkpoint_template: str
    hidden_dim: int
    structured_hidden_dim: int
    dropout: float


@dataclass
class MultiModalDataConfig:
    data_root: str
    image_size: int
    num_workers: int
    feature_batch_size: int
    train_batch_size: int
    valid_batch_size: int
    image_columns: List[str]


@dataclass
class StructuredFeatureConfig:
    categorical_columns: List[str]
    numerical_columns: List[str]


@dataclass
class MultiModalOptimConfig:
    epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float


@dataclass
class MultiModalTrainConfig:
    use_amp: bool
    grad_accum_steps: int
    max_grad_norm: float
    log_interval: int
    early_stop_patience: int
    decision_threshold: float


@dataclass
class MultiModalAppConfig:
    seed: int
    num_folds: int
    folds_dir: str
    output_dir: str
    model: MultiModalModelConfig
    data: MultiModalDataConfig
    structured: StructuredFeatureConfig
    optim: MultiModalOptimConfig
    train: MultiModalTrainConfig


def load_multimodal_config(path: str | Path) -> MultiModalAppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return MultiModalAppConfig(
        seed=raw["seed"],
        num_folds=raw["num_folds"],
        folds_dir=raw["folds_dir"],
        output_dir=raw["output_dir"],
        model=MultiModalModelConfig(**raw["model"]),
        data=MultiModalDataConfig(**raw["data"]),
        structured=StructuredFeatureConfig(**raw["structured"]),
        optim=MultiModalOptimConfig(**raw["optim"]),
        train=MultiModalTrainConfig(**raw["train"]),
    )
