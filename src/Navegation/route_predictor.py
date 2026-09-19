from __future__ import annotations

from dataclasses import dataclass
import torch

from src.data.transforms import build_eval_transform, frame_to_pil
from src.models.factory import load_classifier, resolve_device

@dataclass(frozen=True)
class Prediction:
    predicted_label: str          # "straight" or "intersection"
    predicted_node_index: str     # "NA" or "0..15"

class RoutePredictor:
    """
    Two-stage pipeline:
      Model 1 (binary): 0=straight, 1=intersection
      Model 2 (multiclass): node index for intersections only
    """

    def __init__(self, model_1_path: str, model_2_path: str, device: str | None = "auto"):
        self.device = resolve_device(device)
        self.model_1 = load_classifier(
            model_1_path, "mobilenet_v3_small", 2, device=self.device
        )
        self.model_2 = load_classifier(
            model_2_path, "efficientnet_b0", 16, device=self.device
        )
        self.transform = build_eval_transform()

    def predict(self, frame) -> Prediction:
        x = self.transform(frame_to_pil(frame)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits_1 = self.model_1(x)
            pred_1 = int(torch.argmax(logits_1, dim=1).item())

            if pred_1 == 0:
                return Prediction(predicted_label="straight", predicted_node_index="NA")

            logits_2 = self.model_2(x)
            pred_2 = int(torch.argmax(logits_2, dim=1).item())
            return Prediction(predicted_label="intersection", predicted_node_index=str(pred_2))

