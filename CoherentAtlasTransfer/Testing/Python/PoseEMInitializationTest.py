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
    def test_backend_defaults_to