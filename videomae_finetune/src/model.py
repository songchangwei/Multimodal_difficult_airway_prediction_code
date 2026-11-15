import torch.nn as nn
from transformers import VideoMAEModel


class VideoMAEBinaryClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(self.dropout(features)).squeeze(-1)
        return logits


def build_criterion():
    return nn.BCEWithLogitsLoss()
