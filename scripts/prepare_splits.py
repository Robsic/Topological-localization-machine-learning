#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import RouteSplit, create_route_splits


OUTPUT_DIR = ROOT / "data" / "splits"
ROUTES = RouteSplit(
    train_routes=("route_images_3",),
    validation_routes=("route_images_1",),
    test_routes=("route_images_2",),
)


def main() -> None:
    splits = create_route_splits(ROOT / "data", ROUTES, repository_root=ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "strategy": "complete routes; no frame-level random split",
        "external_evaluation_directory": "validation",
        "splits": {},
    }
    for name, frame in splits.items():
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        intersections = frame[frame["stage1_target"] == 1]
        summary["splits"][name] = {
            "routes": sorted(frame["route_id"].unique().tolist()),
            "samples_stage1": len(frame),
            "straight": int((frame["stage1_target"] == 0).sum()),
            "intersections": len(intersections),
            "nodes_present": sorted(intersections["stage2_target"].dropna().astype(int).unique().tolist()),
        }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
