import torch
import torch.nn as nn


class ImageStructuredFusionClassifier(nn.Module):
    """Fuse frozen image features with trainable structured-data features."""

    def __init__(
        self,
        image_feature_dim: int,
        num_image_views: int,
        structured_dim: int,
        hidden_dim: int,
        structured_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        if num_image_views < 2:
            raise ValueError(
                f"At least two image views are required, got {num_image_views}"
            )
        if structured_dim < 1:
            raise ValueError("structured_dim must be positive")

        self.image_feature_dim = image_feature_dim
        self.num_image_views = num_image_views
        self.structured_dim = structured_dim
        self.image_input_dim = image_feature_dim * num_image_views

        self.image_projection = nn.Sequential(
            nn.LayerNorm(self.image_input_dim),
            nn.Linear(self.image_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.structured_projection = nn.Sequential(
            nn.LayerNorm(structured_dim),
            nn.Linear(structured_dim, structured_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + structured_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        image_features: tuple[torch.Tensor, ...],
        structured_features: torch.Tensor,
    ) -> torch.Tensor:
        if len(image_features) != self.num_image_views:
            raise ValueError(
                f"Expected {self.num_image_views} image feature tensors, "
                f"got {len(image_features)}"
            )
        if any(
            feature.shape[1] != self.image_feature_dim
            for feature in image_features
        ):
            raise ValueError(
                "All image feature tensors must have the configured dimension"
            )
        if structured_features.shape[1] != self.structured_dim:
            raise ValueError(
                "Structured features must have the configured dimension"
            )

        image_features = torch.cat(list(image_features), dim=1)
        image_embedding = self.image_projection(image_features)
        structured_embedding = self.structured_projection(structured_features)
        fused = torch.cat([image_embedding, structured_embedding], dim=1)
        return self.classifier(fused).squeeze(-1)
