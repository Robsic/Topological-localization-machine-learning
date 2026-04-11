from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2

@dataclass(frozen=True)
class Prediction:
    predicted_label: str          # "straight" or "intersection"
    predicted_node_index: str     # "NA" or "0..15"

def _get_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        return ckpt
    return ckpt

def _build_model(arch: str, num_classes: int):
    import torch
    from torchvision import models

    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return m

    if arch == "mobilenet":
        m = models.mobilenet_v3_small(weights=None)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return m

    raise ValueError("Invalid architecture.")

def _load_model(path: str, arch: str, num_classes: int, device):
    import torch

    model = _build_model(arch, num_classes)
    ckpt = torch.load(path, map_location=device)
    sd = _get_state_dict(ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model

class RoutePredictor:
    """
    Two-stage pipeline:
      Model 1 (binary): 0=straight, 1=intersection
      Model 2 (multiclass): node index for intersections only
    """

    def __init__(self, model_1_path: str, model_2_path: str, device: str | None = None):
        import torch
        from torchvision import transforms

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_1 = _load_model(model_1_path, "mobilenet", 2, self.device)
        self.model_2 = _load_model(model_2_path, "efficientnet_b0", 16, self.device)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def predict(self, frame) -> Prediction:
        import torch
        from PIL import Image
        import cv2

        # Input frame pode ser GRAY, BGR ou BGRA
        if frame is None:
            raise ValueError("Frame is None")

        if len(frame.shape) == 2:
            print(f"[DEBUG] Input frame: GRAY numpy shape={frame.shape}, dtype={frame.dtype}")
            gray = frame
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            print(f"[DEBUG] Input frame: BGR numpy shape={frame.shape}, dtype={frame.dtype}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(f"[DEBUG] After BGR->GRAY: shape={gray.shape}, dtype={gray.dtype}")
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            print(f"[DEBUG] Input frame: BGRA numpy shape={frame.shape}, dtype={frame.dtype}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            print(f"[DEBUG] After BGRA->GRAY: shape={gray.shape}, dtype={gray.dtype}")
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")

        # Converter GRAY para RGB (3 canais iguais)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        print(f"[DEBUG] After GRAY->RGB: shape={rgb.shape}, dtype={rgb.dtype}")

        pil_img = Image.fromarray(rgb)  # RGB
        print(f"[DEBUG] PIL: size={pil_img.size}, mode={pil_img.mode}")

        x = self.transform(pil_img)  # [3,255,255]
        print(f"[DEBUG] After transform (no batch): shape={x.shape}, dtype={x.dtype}, min={x.min():.3f}, max={x.max():.3f}")

        x = x.unsqueeze(0).to(self.device)  # [1,3,255,255]
        print(f"[DEBUG] Final input: shape={x.shape}, device={x.device}")

        with torch.no_grad():
            logits_1 = self.model_1(x)
            pred_1 = int(torch.argmax(logits_1, dim=1).item())

            if pred_1 == 0:
                return Prediction(predicted_label="straight", predicted_node_index="NA")

            logits_2 = self.model_2(x)
            pred_2 = int(torch.argmax(logits_2, dim=1).item())
            return Prediction(predicted_label="intersection", predicted_node_index=str(pred_2))


