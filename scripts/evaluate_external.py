#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.Navegation.route_predictor import RoutePredictor
from src.data.dataset import load_classification_manifest
from src.training.metrics import compute_pipeline_metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate current checkpoints on independent routes.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "validation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "baseline")
    parser.add_argument("--model-1", type=Path, default=ROOT / "src/Models/mobilenet_best.pth")
    parser.add_argument("--model-2", type=Path, default=ROOT / "src/Models/efficientnet_b0_best_16.pth")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_summary(frame: pd.DataFrame) -> dict[str, object]:
    expected = frame["expected_stage1"].astype(int).tolist()
    predicted = frame["predicted_stage1"].astype(int).tolist()
    expected_nodes = [None if pd.isna(value) else int(value) for value in frame["expected_node"]]
    predicted_nodes = [None if pd.isna(value) else int(value) for value in frame["predicted_node"]]
    metrics = compute_pipeline_metrics(expected, predicted, expected_nodes, predicted_nodes)
    metrics["stage1_pr_auc"] = float(
        average_precision_score(expected, frame["intersection_probability"].tolist())
    )
    intersections = frame[frame["expected_stage1"] == 1]
    if len(intersections):
        metrics["stage2_top2_accuracy"] = float(intersections["node_top2_correct"].mean())
    else:
        metrics["stage2_top2_accuracy"] = None
    return metrics


def main() -> None:
    args = parse_arguments()
    manifest = load_classification_manifest(args.data_dir, "stage1")
    predictor = RoutePredictor(str(args.model_1), str(args.model_2), device=args.device)
    rows: list[dict[str, object]] = []
    for position, record in enumerate(manifest.to_dict(orient="records"), start=1):
        image = cv2.imread(record["filepath"], cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"could not read image: {record['filepath']}")
        prediction = predictor.predict(image)
        expected_node = int(record["node_index"]) if record["label"] == "intersection" else None
        predicted_node = (
            int(prediction.predicted_node_index)
            if prediction.predicted_node_index != "NA"
            else None
        )
        rows.append(
            {
                "route_id": record["route_id"],
                "filename": record["filename"],
                "expected_label": record["label"],
                "predicted_label": prediction.predicted_label,
                "expected_stage1": int(record["target"]),
                "predicted_stage1": int(prediction.predicted_label == "intersection"),
                "intersection_probability": prediction.intersection_probability,
                "stage1_confidence": prediction.stage1_confidence,
                "expected_node": expected_node,
                "predicted_node": predicted_node,
                "stage2_confidence": prediction.stage2_confidence,
                "predicted_node_top2": ",".join(map(str, prediction.predicted_node_top2)),
                "node_top2_correct": bool(
                    expected_node is not None and expected_node in prediction.predicted_node_top2
                ),
            }
        )
        if position % 50 == 0 or position == len(manifest):
            print(f"Evaluated {position}/{len(manifest)} images", flush=True)

    predictions = pd.DataFrame(rows)
    summary = {
        "baseline": {
            "model_1": str(args.model_1.relative_to(ROOT)),
            "model_1_sha256": sha256(args.model_1),
            "model_2": str(args.model_2.relative_to(ROOT)),
            "model_2_sha256": sha256(args.model_2),
            "device": str(predictor.device),
            "external_data": str(args.data_dir.relative_to(ROOT)),
        },
        "overall": route_summary(predictions),
        "by_route": {
            route: route_summary(frame.reset_index(drop=True))
            for route, frame in predictions.groupby("route_id")
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
