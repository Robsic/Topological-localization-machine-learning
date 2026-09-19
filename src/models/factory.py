from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision import models


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def build_classifier(architecture: str, number_of_classes: int, *, pretrained: bool = False):
    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least 2")

    name = architecture.lower()
    weights = "DEFAULT" if pretrained else None
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, number_of_classes)
    elif name in {"mobilenet", "mobilenet_v3_small"}:
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, number_of_classes)
    elif name in {"shufflenet_v2", "shufflenet_v2_x1_0"}:
        model = models.shufflenet_v2_x1_0(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, number_of_classes)
    else:
        supported = "efficientnet_b0, mobilenet_v3_small, shufflenet_v2_x1_0"
        raise ValueError(f"unsupported architecture '{architecture}'. Choose one of: {supported}")
    return model


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint does not contain a state dictionary")
    return {str(key).removeprefix("module."): value for key, value in checkpoint.items()}


def load_classifier(
    checkpoint_path: str | Path,
    architecture: str,
    number_of_classes: int,
    *,
    device: str | torch.device | None = "auto",
):
    resolved_device = resolve_device(device)
    model = build_classifier(architecture, number_of_classes)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.to(resolved_device)
    model.eval()
    return model
