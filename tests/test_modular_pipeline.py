import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.dataset import ImagePathDataset, load_classification_manifest, validate_manifest_images
from src.data.transforms import build_eval_transform, frame_to_pil
from src.models.factory import build_classifier, resolve_device
from src.training.metrics import compute_classification_metrics, compute_pipeline_metrics


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        route = Path(self.temporary_directory.name) / "route_images_1"
        images = route / "images"
        images.mkdir(parents=True)
        for filename in ("straight.png", "intersection.png"):
            Image.new("RGB", (20, 10), color="white").save(images / filename)
        with (route / "gps_log.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["filename", "label", "node_index"])
            writer.writeheader()
            writer.writerow({"filename": "straight.png", "label": "straight", "node_index": 4})
            writer.writerow({"filename": "intersection.png", "label": "intersection", "node_index": 7})

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_manifests_separate_the_two_tasks(self):
        stage1 = load_classification_manifest(self.temporary_directory.name, "stage1")
        stage2 = load_classification_manifest(self.temporary_directory.name, "stage2")
        self.assertEqual(stage1["target"].tolist(), [0, 1])
        self.assertEqual(stage2["target"].tolist(), [7])
        self.assertTrue(validate_manifest_images(stage1).is_valid)

    def test_dataset_returns_tensor_and_target(self):
        manifest = load_classification_manifest(self.temporary_directory.name, "stage1")
        image, target = ImagePathDataset(manifest, build_eval_transform())[0]
        self.assertEqual(tuple(image.shape), (3, 256, 256))
        self.assertEqual(target, 0)


class TransformTests(unittest.TestCase):
    def test_opencv_frame_formats_are_supported(self):
        for frame in (
            np.zeros((10, 20), dtype=np.uint8),
            np.zeros((10, 20, 3), dtype=np.uint8),
            np.zeros((10, 20, 4), dtype=np.uint8),
        ):
            self.assertEqual(frame_to_pil(frame).mode, "RGB")


class ModelTests(unittest.TestCase):
    def test_factory_builds_all_supported_architectures(self):
        for architecture in ("mobilenet", "efficientnet_b0", "shufflenet_v2"):
            model = build_classifier(architecture, 3)
            self.assertIsNotNone(model)
        self.assertEqual(resolve_device("cpu").type, "cpu")


class MetricsTests(unittest.TestCase):
    def test_classification_and_pipeline_metrics(self):
        metrics = compute_classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1])
        self.assertEqual(metrics["accuracy"], 0.75)
        pipeline = compute_pipeline_metrics(
            [0, 1, 1], [0, 1, 1], [None, 3, 4], [None, 3, 2]
        )
        self.assertAlmostEqual(pipeline["end_to_end_accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
