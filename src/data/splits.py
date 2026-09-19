from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .dataset import load_classification_manifest


@dataclass(frozen=True)
class RouteSplit:
    train_routes: tuple[str, ...]
    validation_routes: tuple[str, ...]
    test_routes: tuple[str, ...]

    def validate(self) -> None:
        groups = [set(self.train_routes), set(self.validation_routes), set(self.test_routes)]
        if any(not group for group in groups):
            raise ValueError("train, validation and test must each contain at least one route")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("a route cannot occur in more than one split")


def create_route_splits(
    data_dir: str | Path,
    split: RouteSplit,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Create leakage-free dataframes using complete routes as groups."""
    split.validate()
    manifest = load_classification_manifest(data_dir, "stage1")
    manifest = manifest.rename(columns={"target": "stage1_target"})
    manifest["stage2_target"] = pd.to_numeric(manifest["node_index"], errors="coerce")
    manifest.loc[manifest["label"] != "intersection", "stage2_target"] = pd.NA

    assignments = {
        "train": set(split.train_routes),
        "validation": set(split.validation_routes),
        "test": set(split.test_routes),
    }
    expected_routes = set().union(*assignments.values())
    available_routes = set(manifest["route_id"].unique())
    if expected_routes != available_routes:
        missing = sorted(available_routes - expected_routes)
        unknown = sorted(expected_routes - available_routes)
        raise ValueError(f"route assignment mismatch; unassigned={missing}, unknown={unknown}")

    if repository_root is not None:
        root = Path(repository_root).resolve()
        manifest["filepath"] = manifest["filepath"].map(
            lambda value: str(Path(value).resolve().relative_to(root))
        )

    result = {
        name: manifest[manifest["route_id"].isin(routes)].reset_index(drop=True)
        for name, routes in assignments.items()
    }
    route_sets = [set(frame["route_id"]) for frame in result.values()]
    if route_sets[0] & route_sets[1] or route_sets[0] & route_sets[2] or route_sets[1] & route_sets[2]:
        raise AssertionError("route leakage detected")
    return result
