from pathlib import Path

import torch
import torch.nn as nn
from transformers import VideoMAEModel


class FrozenVideoMAEBackbone(nn.Module):
    """Load an existing single-video VideoMAE backbone and keep it frozen."""

    def __init__(self, model_name: str, checkpoint_path: str | Path):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained(model_name)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        backbone_state_dict = {
            key.removeprefix("backbone."): value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
        if not backbone_state_dict:
            raise RuntimeError(f"No backbone weights found in checkpoint: {checkpoint_path}")

        missing, unexpected = self.backbone.load_state_dict(
            backbone_state_dict,
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Could not load VideoMAE backbone checkpoint {checkpoint_path}; "
                f"missing={missing}, unexpected={unexpected}"
            )

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()
        self.hidden_size = self.backbone.config.hidden_size

    def train(self, mode: bool = True):
        # Frozen feature extractors must stay in eval mode while features are built.
        super().train(False)
        self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state.mean(dim=1)


class TwoVideoFusionClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        input_dim = feature_dim * 2
        self.norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features_1: torch.Tensor, features_2: torch.Tensor):
        features = torch.cat([features_1, features_2], dim=1)
        return self.classifier(self.norm(features)).squeeze(-1)
