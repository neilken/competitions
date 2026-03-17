from __future__ import annotations

from torch import nn

from src.models.cnn_alt import BirdCLEFAltCNN
from src.models.cnn_baseline import BirdCLEFBaselineCNN


def create_model(model_family: str, num_classes: int) -> nn.Module:
    key = model_family.strip().lower()
    if key in {"cnn_logmel_baseline", "baseline", "cnn_baseline"}:
        return BirdCLEFBaselineCNN(num_classes=num_classes)
    if key in {"cnn_alt_family", "alt", "depthwise_alt"}:
        return BirdCLEFAltCNN(num_classes=num_classes)
    raise ValueError(
        f"Unknown model_family '{model_family}'. Supported: "
        "cnn_logmel_baseline, cnn_alt_family"
    )
