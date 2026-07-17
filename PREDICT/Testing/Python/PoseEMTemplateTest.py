import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

PREDICT_DIR = Path(__file__).resolve().parents[2]
if str(PREDICT_DIR) not in sys.path:
    sys.path.insert(0, str(PREDICT_DIR))

from PoseEMTemplate import (
    LEGACY_BACKEND,
    POSE_EM_BACKEND,
    PoseEMSettings,
    coverage_prescale,
    optimization_backend,
    pose_em_enabled,
    run_pose_em_registration,
    similarity_matrix,
    ssm_sample,
    transform_points,
)


class _FakeRegistration:
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.initial_state = None
        _FakeRegistration.last = self

    def set_initial_state(self, coefficients, rotation, scale, translation, world_units=True):
        self.initial_state = (
            np.asarray(coefficients).copy(),
            np.asarray(rotation).copy(),
            float(scale),
            np.asarray(translation).copy(),
            bool(world_units),
        )

    def register(self):
        coefficients, rotation, scale, translation, _ = self.initial_state
        shaped = ssm_sample(self.kwargs["Y"], self.kwargs["U"], coefficients)
        points = scale * (shaped @ rotation.T) + translation
        return points, {
            "b": coefficients.reshape(-1, 1),
            "R_world": rotation,
            "s_world": scale,
            "t_world": translation,
        }


def _initializer(coefficients, rotation, scale, translation):
    def initialize(*args, **kwargs):
        return SimpleNamespace(
            coefficients=np.asarray(coefficients),
            rotation=np.asarray(rotation),
            scale=float(scale),
            translation=np.asarray(translation).reshape(1, 3),
            score=12.5,
            score_margin=3.25,
            posterior_entropy=0.4,
            effective_hypotheses=1.49,
            hypotheses_evaluated=61,
            hypotheses_refined=2,
        )
    return initialize


class PoseEMTemplateUnitTest(unittest.TestCase):
    def test_slicer_module_wires_backend_selector_to_standalone_and_batch_paths(self):
        source = (PREDICT_DIR / "PREDICT.py").read_text(encoding="utf-8")
        compile(source, str(PREDICT_DIR / "PREDICT.py"), "exec")
        self.assertIn("Pose-marginalized EM (experimental)", source)
        self.assertIn("backend = optimization_backend(parameters)", source)
        self.assertIn("pose_em = pose_em_enabled(parameters, skip_optimization=skipOpt)", source)
        self.assertIn("logic.initialize_template_pose_em", source)
        self.assertIn("self.runPoseEMDeformable", source)
        self.assertIn('"pose_em_diagnostics.json"', source)

    def test_backend_is_explicit_and_legacy_by_default(self):
        self.assertEqual(optimization_backend({}), LEGACY_BACKEND)
        self.assertEqual(optimization_backend({"optimizationBackend": "pose_em"}), POSE_EM_BACKEND)
        self.assertTrue(pose_em_enabled({"optimizationBackend": "pose_em"}))
        self.assertFalse(pose_em_enabled({"optimizationBackend": "pose_em"}, skip_optimization=True))
        self.assertFalse(pose_em_enabled({"optimizationBackend": "ransac"}))
        with self.assertRaises(ValueError):
            optimization_backend({"optimizationBackend": "unknown"})

    def test_settings_map_ui_values_to_biocpd_arguments(self):
        settings = PoseEMSettings.from_mapping({
            "poseRotationCount": 24,
            "poseCoarseSourceCount": 80,
            "poseCoarseTargetCount": 90,
            "poseCoarseRank": 3,
            "poseCoarseIterations": 5,
            "poseRefineCount": 1,
            "poseRefineTargetCount": 120,
            "poseRefineIterations": 9,
            "lambda_reg": 0.03,
            "w": 0.15,
            "poseIdentityPrior": 0.25,
            "poseSeed": 7,
        })
        self.assertEqual(settings.rotation_count, 24)
        self.assertEqual(settings.refine_count, 1)
        self.assertEqual(settings.seed, 7)
        self.assertEqual(settings.initializer_kwargs()["outlier_weight"], 0.15)

    def test_ssm_sample_uses_native_coefficients_without_sqrt_eigen_scaling(self):
        mean = np.zeros((2, 3))
        modes = np.zeros((2, 3, 2))
        modes[0, 0, 0] = 1.0
        modes[1, 1, 1] = 2.0
        sample = ssm_sample(mean, modes, np.array([3.0, -0.5]))
        np.testing.assert_allclose(sample, [[3.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

    def test_similarity_matrix_matches_biocpd_row_point_convention(self):
        points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
        rotation = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
        translation = np.array([[4.0, -2.0, 0.5]])
        matrix = similarity_matrix(rotation, 1.5, translation)
        expected = 1.5 * (points @ rotation.T) + translation
        np.testing.assert_allclose(transform_points(points, matrix), expected, atol=1e-12)

    def test_coverage_prescale_uses_expected_full_target_size(self):
        source = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        target = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        self.assertAlmostEqual(coverage_prescale(source, target, 0.25), 1.0)

    def test_centered_warm_state_is_composed_back_to_world_coordinates(self):
        mean = np.array([[-1.0, -0.5, 0.25], [1.0, 0.5, -0.25]])
        modes = np.zeros((2, 3, 1)); modes[:, 2, 0] = [0.5, -0.5]
        eigenvalues = np.array([0.4])
        coefficients = np.array([0.3])
        rotation = Rotation.from_euler("xyz", [10.0, -15.0, 25.0], degrees=True).as_matrix()
        translation = np.array([[2.0, -1.0, 0.4]])
        target = 1.2 * (ssm_sample(mean, modes, coefficients) @ rotation.T) + translation
        result = run_pose_em_registration(
            mean, target, modes, eigenvalues, PoseEMSettings(rotation_count=12),
            max_iterations=40, tolerance=1e-6, with_scale=True,
            initializer=_initializer(coefficients, rotation, 1.2, np.zeros((1, 3))),
            registration_class=_FakeRegistration,
        )
        state = _FakeRegistration.last.initial_state
        self.assertFalse(_FakeRegistration.last.kwargs["use_kdtree"])
        np.testing.assert_allclose(state[0], coefficients)
        np.testing.assert_allclose(state[1], rotation)
        self.assertAlmostEqual(state[2], 1.2)
        np.testing.assert_allclose(state[3], np.zeros((1, 3)), atol=1e-12)
        self.assertTrue(state[4])
        np.testing.assert_allclose(result.points, target, atol=1e-12)
        np.testing.assert_allclose(result.translation, translation, atol=1e-12)
        np.testing.assert_allclose(
            transform_points(ssm_sample(mean, modes, coefficients), result.similarity_matrix()),
            target,
            atol=1e-12,
        )
        self.assertEqual(result.hypotheses_evaluated, 61)

    def test_fixed_scale_handoff_recenters_pose_and_disables_scale_optimization(self):
        mean = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        modes = np.zeros((2, 3, 1))
        eigenvalues = np.array([1.0])
        rotation = Rotation.from_euler("z", 35.0, degrees=True).as_matrix()
        target = mean @ rotation.T + np.array([3.0, -2.0, 0.5])
        result = run_pose_em_registration(
            mean, target, modes, eigenvalues, PoseEMSettings(rotation_count=12),
            max_iterations=10, tolerance=1e-6, with_scale=False,
            initializer=_initializer([0.0], rotation, 0.4, [[9.0, 9.0, 9.0]]),
            registration_class=_FakeRegistration,
        )
        self.assertFalse(_FakeRegistration.last.kwargs["with_scale"])
        self.assertAlmostEqual(result.scale, 1.0)
        np.testing.assert_allclose(result.points.mean(0), target.mean(0), atol=1e-12)

    def test_partial_target_prescale_is_carried_into_fixed_scale_registration(self):
        mean = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        modes = np.zeros((2, 3, 1))
        target = np.array([[4.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
        source_scale = coverage_prescale(mean, target, coverage=0.5)
        result = run_pose_em_registration(
            mean, target, modes, np.array([1.0]), PoseEMSettings(rotation_count=12),
            max_iterations=10, tolerance=1e-6, with_scale=False, source_scale=source_scale,
            initializer=_initializer([0.0], np.eye(3), 0.5, [[0.0, 0.0, 0.0]]),
            registration_class=_FakeRegistration,
        )
        self.assertAlmostEqual(source_scale, 0.5)
        self.assertAlmostEqual(result.scale, source_scale)
        np.testing.assert_allclose(result.points.mean(0), target.mean(0), atol=1e-12)


class PoseEMTemplateIntegrationTest(unittest.TestCase):
    def test_small_offset_target_retains_scale(self):
        try:
            from biocpd import pose_marginalized_initialization
        except (ImportError, AttributeError):
            self.skipTest("biocpd pose-EM API is unavailable")
        rng = np.random.default_rng(17)
        mean = rng.normal(size=(80, 3)) * np.array([1.0, 0.55, 0.25])
        mean[:, 0] += 0.15 * mean[:, 1] ** 2
        modes = np.zeros((80, 3, 2))
        modes[:, 0, 0] = 0.03 * mean[:, 0]
        modes[:, 1, 1] = 0.02 * mean[:, 1]
        eigenvalues = np.array([0.4, 0.2])
        rotation = Rotation.from_euler("xyz", [40.0, -30.0, 28.0], degrees=True).as_matrix()
        target = 0.08 * (mean @ rotation.T) + np.array([25.0, -12.0, 4.0])
        settings = PoseEMSettings(
            rotation_count=24,
            coarse_source_count=60,
            coarse_target_count=60,
            coarse_rank=2,
            coarse_iterations=5,
            refine_count=2,
            refine_target_count=80,
            refine_iterations=10,
            seed=5,
        )
        result = run_pose_em_registration(
            mean, target, modes, eigenvalues, settings,
            max_iterations=100, tolerance=1e-5, with_scale=True,
        )
        registered_diagonal = np.linalg.norm(np.ptp(result.points, axis=0))
        target_diagonal = np.linalg.norm(np.ptp(target, axis=0))
        self.assertGreater(registered_diagonal / target_diagonal, 0.9)
        self.assertLess(registered_diagonal / target_diagonal, 1.1)
        self.assertGreater(result.scale, 0.05)

    def test_large_rotation_recovery_through_atlas_adapter(self):
        try:
            from biocpd import pose_marginalized_initialization
        except (ImportError, AttributeError):
            self.skipTest("biocpd pose-EM API is unavailable")
        rng = np.random.default_rng(19)
        mean = rng.normal(size=(60, 3)) * np.array([2.0, 1.1, 0.6])
        mean[:, 0] += 0.2 * mean[:, 1] ** 2
        modes = np.zeros((60, 3, 2))
        modes[:, 0, 0] = 0.03 * mean[:, 0]
        modes[:, 1, 1] = 0.02 * mean[:, 1]
        eigenvalues = np.array([0.4, 0.2])
        coefficients = np.array([0.35, -0.25])
        deformed = ssm_sample(mean, modes, coefficients)
        rotation = Rotation.from_euler("xyz", [36.0, -24.0, 31.0], degrees=True).as_matrix()
        target = 1.08 * (deformed @ rotation.T) + np.array([0.8, -0.4, 0.2])
        settings = PoseEMSettings(
            rotation_count=24,
            coarse_source_count=50,
            coarse_target_count=50,
            coarse_rank=2,
            coarse_iterations=4,
            refine_count=2,
            refine_target_count=60,
            refine_iterations=8,
            seed=7,
        )
        result = run_pose_em_registration(
            mean, target, modes, eigenvalues, settings,
            max_iterations=120, tolerance=1e-5, with_scale=True, use_kdtree=False,
        )
        error = np.median(np.linalg.norm(result.points - target, axis=1))
        diagonal = np.linalg.norm(np.ptp(target, axis=0))
        self.assertLess(error / diagonal, 0.05)
        self.assertLess(result.final_parameters.get("b").size, 3)


if __name__ == "__main__":
    unittest.main()
