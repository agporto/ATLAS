import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[2]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from Resources.Python.PoseEMTemplate import (
    LEGACY_BACKEND,
    POSE_EM_BACKEND,
    PoseEMSettings,
    optimization_backend,
    pose_em_enabled,
    similarity_matrix,
    transform_points,
)


class PoseEMInitializationTest(unittest.TestCase):
    def test_backend_defaults_to_legacy(self):
        self.assertEqual(optimization_backend({}), LEGACY_BACKEND)
        self.assertEqual(optimization_backend({"optimizationBackend": "pose_em"}), POSE_EM_BACKEND)
        self.assertTrue(pose_em_enabled({"optimizationBackend": "pose_em"}))

    def test_default_settings_validate(self):
        PoseEMSettings().validate()

    def test_similarity_transform(self):
        rotation = np.eye(3)
        matrix = similarity_matrix(rotation, 2.0, np.array([1.0, 2.0, 3.0]))
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        transformed = transform_points(points, matrix)
        expected = np.array([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]])
        np.testing.assert_allclose(transformed, expected)


if __name__ == "__main__":
    unittest.main()
