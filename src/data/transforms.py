from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class ImageTransformConfig:
    image_size: tuple[int, int] = (256, 256)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    horizontal_flip_probability: float = 0.5
    rotation_degrees: float = 5.0


def frame_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV grayscale, BGR or BGRA frame to an RGB PIL image."""
    if frame is None:
        raise ValueError("frame is None")
    if frame.ndim == 2:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"unexpected frame shape: {frame.shape}")
    return Image.fromarray(rgb)


def build_eval_transform(config: ImageTransformConfig | None = None):
    cfg = config or ImageTransformConfig()
    return transforms.Compose(
        [
            transforms.Resize(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )


def build_train_transform(config: ImageTransformConfig | None = None):
    cfg = config or ImageTransformConfig()
    return transforms.Compose(
        [
            transforms.Resize(cfg.image_size),
            transforms.RandomHorizontalFlip(p=cfg.horizontal_flip_probability),
            transforms.RandomRotation(degrees=cfg.rotation_degrees),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )
