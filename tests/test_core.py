"""Regression tests for the incremental mosaicking pipeline.

Run with ``python -m unittest discover -s tests`` from the repository root.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mosaic.config import MosaicConfig, MosaicStoreConfig, OutputConfig
from mosaic.dataset import ImageSequence
from mosaic.features import FeatureSet
from mosaic.pipeline import StitchingPipeline
from mosaic.registration import homogeneous_divide
from mosaic.store import MosaicFeatureStore


def _textured(height: int = 240, width: int = 320) -> np.ndarray:
    rng = np.random.default_rng(7)
    image = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


class DatasetTests(unittest.TestCase):
    def test_pattern_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            cv2.imwrite(str(Path(tmp) / "frame_0001.jpg"), _textured())
            # default pattern is *.JPG but the file is *.jpg
            sequence = ImageSequence(tmp)
            self.assertEqual(len(sequence), 1)


class StoreTests(unittest.TestCase):
    def test_frame_ids_are_sequence_indices(self):
        store = MosaicFeatureStore(MosaicStoreConfig(min_distance=1.0))
        descriptors = np.ones((1, 128), np.float32)
        for index in (0, 2, 5):
            store.update(
                FeatureSet(
                    points=np.array([[index * 200.0, 0.0]], np.float32),
                    descriptors=descriptors,
                ),
                np.eye(3),
                frame_id=index,
            )
        self.assertEqual(sorted(int(v) for v in np.unique(store._frames)), [0, 2, 5])


class GeometryTests(unittest.TestCase):
    def test_homogeneous_divide_handles_negative_w(self):
        points = np.array([[2.0, 4.0, -2.0], [1.0, 1.0, 0.0]])
        result = homogeneous_divide(points)
        np.testing.assert_allclose(result[0], [-1.0, -2.0])
        self.assertTrue(np.isnan(result[1]).all())


class PipelineTests(unittest.TestCase):
    def test_featureless_frame_is_rejected_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cv2.imwrite(str(tmp / "image_0000.jpg"), _textured())
            cv2.imwrite(str(tmp / "image_0001.jpg"), np.zeros((240, 320, 3), np.uint8))

            config = MosaicConfig(
                dataset_dir=str(tmp),
                image_pattern="*.jpg",
                output=OutputConfig(root=str(tmp / "out"), save_preview=False),
            )
            pipeline = StitchingPipeline(config, log=lambda *_: None, run_name="test")
            report = pipeline.run()
            self.assertEqual(len(report.steps), 2)
            self.assertEqual(report.rejected, 1)
            self.assertEqual(report.accepted, 0)


if __name__ == "__main__":
    unittest.main()
