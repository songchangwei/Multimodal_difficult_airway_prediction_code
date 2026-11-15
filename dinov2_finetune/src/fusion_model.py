from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel


class FrozenDinoV2Backbone(nn.Module):
    """Load the backbone from an existing single-image checkpoint and freeze it."""

    def __init__(self, model_name: str, checkpoint_path: str | Path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

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

        missing, unexpected = self.backbone.load_state_dict(backbone_state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Could not load backbone checkpoint {checkpoint_path}; "
                f"missing={missing}, unexpected={unexpected}"
            )

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()
        self.hidden_size = self.backbone.config.hidden_size

    def train(self, mode: bool = True):
        # The extractor is deliberately kept in eval mode so dropout/batch statistics
        # cannot change while frozen features are generated.
        super().train(False)
        self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]


class MultiImageFusionClassifier(nn.Module):
    """Fuse frozen features from any configured number of image views."""

    def __init__(
        self,
        feature_dim: int,
        num_views: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        if num_views < 2:
            raise ValueError(f"At least two image views are required, got {num_views}")
        self.feature_dim = feature_dim
        self.num_views = num_views
        input_dim = feature_dim * num_views
        self.norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, *features: torch.Tensor):
        if len(features) != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} feature tensors, got {len(features)}"
            )
        if any(feature.shape[1] != self.feature_dim for feature in features):
            raise ValueError(
                "All image feature tensors must have the configured dimension"
            )
        features = torch.cat(list(features), dim=1)
        return self.classifier(self.norm(features)).squeeze(-1)


class TwoImageFusionClassifier(MultiImageFusionClassifier):
    """Backward-compatible two-image specialization for existing experiments."""

    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__(
            feature_dim=feature_dim,
            num_views=2,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
