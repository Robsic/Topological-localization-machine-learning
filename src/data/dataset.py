from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

TaskName = Literal["stage1", "stage2"]


@dataclass(frozen=True)
class DatasetIntegrityReport:
    total_rows: int
    existing_images: int
    missing_images: tuple[str, ...]
    duplicate_paths: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing_images and not self.duplicate_paths


def discover_route_directories(base_dir: str | Path) -> list[Path]:
    """Return route directories that contain a gps_log.csv file."""
    base = Path(base_dir)
    return sorted(
        (path.parent for path in base.glob("route_images_*/gps_log.csv")),
        key=lambda path: path.name,
    )


def _parse_node_index(value: object, number_of_nodes: int) -> int | None:
    try:
        node_index = int(float(str(value)))
    except (TypeError, ValueError):
        return None
    return node_index if 0 <= node_index < number_of_nodes else None


def load_classification_manifest(
    base_dir: str | Path,
    task: TaskName,
    *,
    number_of_nodes: int = 16,
) -> pd.DataFrame:
    """Build one normalized manifest from all route CSV files.

    ``stage1`` maps straight to 0 and intersection to 1. ``stage2`` keeps
    intersection samples only and uses the node index as the target.
    """
    if task not in {"stage1", "stage2"}:
        raise ValueError("task must be 'stage1' or 'stage2'")

    rows: list[dict[str, object]] = []
    for route_dir in discover_route_directories(base_dir):
        csv_path = route_dir / "gps_log.csv"
        frame = pd.read_csv(csv_path)
        required = {"filename", "label", "node_index"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        for record in frame.to_dict(orient="records"):
            label = str(record.get("label", "")).strip().lower()
            if label not in {"straight", "intersection"}:
                continue

            node_index = _parse_node_index(record.get("node_index"), number_of_nodes)
            if task == "stage2" and (label != "intersection" or node_index is None):
                continue

            filename = str(record["filename"]).strip()
            normalized = dict(record)
            normalized.update(
                route_id=route_dir.name,
                filepath=str((route_dir / "images" / filename).resolve()),
                label=label,
                target=(int(label == "intersection") if task == "stage1" else node_index),
            )
            rows.append(normalized)

    return pd.DataFrame(rows)


def validate_manifest_images(manifest: pd.DataFrame) -> DatasetIntegrityReport:
    if "filepath" not in manifest.columns:
        raise ValueError("manifest must contain a filepath column")
    paths = manifest["filepath"].astype(str)
    missing = tuple(sorted(path for path in paths if not Path(path).is_file()))
    duplicates = tuple(sorted(paths[paths.duplicated(keep=False)].unique()))
    return DatasetIntegrityReport(
        total_rows=len(paths),
        existing_images=len(paths) - len(missing),
        missing_images=missing,
        duplicate_paths=duplicates,
    )


class ImagePathDataset(Dataset):
    """PyTorch dataset backed by a normalized classification manifest."""

    def __init__(self, manifest: pd.DataFrame, transform: Callable | None = None):
        required = {"filepath", "target"}
        missing = required.difference(manifest.columns)
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        self.manifest = manifest.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        with Image.open(row["filepath"]) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["target"])
