import torch
import torch.nn as nn


class VideoStructuredFusionClassifier(nn.Module):
    """融合冻结的视频特征和可训练的结构化数据分支。"""

    def __init__(
        self,
        video_feature_dim: int,
        num_video_views: int,
        structured_dim: int,
        hidden_dim: int,
        structured_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        if num_video_views < 1:
            raise ValueError("num_video_views must be positive")
        if structured_dim < 1:
            raise ValueError("structured_dim must be positive")

        self.video_feature_dim = video_feature_dim
        self.num_video_views = num_video_views
        self.structured_dim = structured_dim
        video_input_dim = video_feature_dim * num_video_views

        self.video_projection = nn.Sequential(
            nn.LayerNorm(video_input_dim),
            nn.Linear(video_input_dim, hidden_dim),
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
        video_features: tuple[torch.Tensor, ...],
        structured_features: torch.Tensor,
    ) -> torch.Tensor:
        if len(video_features) != self.num_video_views:
            raise ValueError(
                f"Expected {self.num_video_views} video feature tensors, "
                f"got {len(video_features)}"
            )
        if any(
            feature.shape[1] != self.video_feature_dim
            for feature in video_features
        ):
            raise ValueError(
                "All video feature tensors must have the configured dimension"
            )
        if structured_features.shape[1] != self.structured_dim:
            raise ValueError(
                "Structured features must have the configured dimension"
            )

        video_embedding = self.video_projection(
            torch.cat(list(video_features), dim=1)
        )
        structured_embedding = self.structured_projection(structured_features)
        fused = torch.cat([video_embedding, structured_embedding], dim=1)
        return self.classifier(fused).squeeze(-1)
