from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np


LEGACY_BACKEND = "ransac"
POSE_EM_BACKEND = "pose_em"


def optimization_backend(parameters: Mapping[str, Any]) -> str:
    backend = str(parameters.get("optimizationBackend", LEGACY_BACKEND)).strip().lower()
    if backend not in (LEGACY_BACKEND, POSE_EM_BACKEND):
        raise ValueError(f"Unknown template optimization backend: {backend}")
    return backend


def pose_em_enabled(parameters: Mapping[str, Any], skip_optimization: bool = False) -> bool:
    return bool(not skip_optimization and optimization_backend(parameters) == POSE_EM_BACKEND)


@dataclass(frozen=True)
class PoseEMSettings:
    rotation_count: int = 96
    coarse_source_count: int = 400
    coarse_target_count: int = 400
    coarse_rank: int = 12
    coarse_iterations: int = 8
    refine_count: int = 12
    refine_target_count: int = 1600
    refine_iterations: int = 30
    lambda_reg: float = 0.01
    outlier_weight: float = 0.1
    identity_prior_probability: float = 0.2
    seed: int = 0

    @classmethod
    def from_mapping(cls, parameters: Mapping[str, Any]) -> "PoseEMSettings":
        settings = cls(
            rotation_count=int(parameters.get("poseRotationCount", 96)),
            coarse_source_count=int(parameters.get("poseCoarseSourceCount", 400)),
            coarse_target_count=int(parameters.get("poseCoarseTargetCount", 400)),
            coarse_rank=int(parameters.get("poseCoarseRank", 12)),
            coarse_iterations=int(parameters.get("poseCoarseIterations", 8)),
            refine_count=int(parameters.get("poseRefineCount", 12)),
            refine_target_count=int(parameters.get("poseRefineTargetCount", 1600)),
            refine_iterations=int(parameters.get("poseRefineIterations", 30)),
            lambda_reg=float(parameters.get("lambda_reg", 0.01)),
            outlier_weight=float(parameters.get("w", 0.1)),
            identity_prior_probability=float(parameters.get("poseIdentityPrior", 0.2)),
            seed=int(parameters.get("poseSeed", 0)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        positive = {
            "rotation_count": self.rotation_count,
            "coarse_source_count": self.coarse_source_count,
            "coarse_target_count": self.coarse_target_count,
            "coarse_rank": self.coarse_rank,
            "coarse_iterations": self.coarse_iterations,
            "refine_count": self.refine_count,
            "refine_target_count": self.refine_target_count,
            "refine_iterations": self.refine_iterations,
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise ValueError(f"Pose EM settings must be positive: {', '.join(invalid)}")
        if self.lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative")
        if not 0 <= self.outlier_weight < 1:
            raise ValueError("outlier_weight must be in [0, 1)")
        if not 0 < self.identity_prior_probability < 1:
            raise ValueError("identity_prior_probability must be in (0, 1)")

    def initializer_kwargs(self) -> dict[str, Any]:
        return {
            "rotation_count": self.rotation_count,
            "coarse_source_count": self.coarse_source_count,
            "coarse_target_count": self.coarse_target_count,
            "coarse_rank": self.coarse_rank,
            "coarse_iterations": self.coarse_iterations,
            "refine_count": self.refine_count,
            "refine_target_count": self.refine_target_count,
            "refine_iterations": self.refine_iterations,
            "lambda_reg": self.lambda_reg,
            "outlier_weight": self.outlier_weight,
            "identity_prior_probability": self.identity_prior_probability,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class PoseEMRegistrationResult:
    points: np.ndarray
    coefficients: np.ndarray
    rotation: np.ndarray
    scale: float
    translation: np.ndarray
    score: float
    score_margin: float
    posterior_entropy: float
    effective_hypotheses: float
    hypotheses_evaluated: int
    hypotheses_refined: int
    final_parameters: Mapping[str, Any]

    def similarity_matrix(self) -> np.ndarray:
        return similarity_matrix(self.rotation, self.scale, self.translation)


def validate_ssm(mean: np.ndarray, modes: np.ndarray, eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=np.float64)
    modes = np.asarray(modes, dtype=np.float64)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if mean.ndim != 2 or mean.shape[1] != 3 or len(mean) == 0:
        raise ValueError("SSM mean must have non-empty shape (M, 3)")
    if modes.ndim == 2:
        if modes.shape[0] != mean.size:
            raise ValueError("Flattened SSM modes do not match the mean")
        modes = modes.reshape(len(mean), 3, modes.shape[1])
    if modes.ndim != 3 or modes.shape[:2] != mean.shape:
        raise ValueError("SSM modes must have shape (M, 3, K) or (M*3, K)")
    if eigenvalues.ndim != 1 or modes.shape[2] != len(eigenvalues) or len(eigenvalues) == 0:
        raise ValueError("SSM eigenvalues do not match the modes")
    if not np.isfinite(mean).all() or not np.isfinite(modes).all() or not np.isfinite(eigenvalues).all():
        raise ValueError("SSM arrays must be finite")
    if np.any(eigenvalues <= 0):
        raise ValueError("SSM eigenvalues must be positive")
    return mean, modes, eigenvalues


def ssm_sample(mean: np.ndarray, modes: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    mean, modes, eigenvalues = validate_ssm(
        mean,
        modes,
        np.ones(np.asarray(modes).shape[-1], dtype=np.float64),
    )
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if len(coefficients) != len(eigenvalues):
        raise ValueError("SSM coefficients do not match the modes")
    return mean + (modes.reshape(mean.size, -1) @ coefficients).reshape(mean.shape)


def similarity_matrix(rotation: np.ndarray, scale: float, translation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Similarity rotation and translation must be 3D")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Similarity scale must be finite and positive")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = float(scale) * rotation
    matrix[:3, 3] = translation
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or matrix.shape != (4, 4):
        raise ValueError("Expected points (N, 3) and a 4x4 transform")
    return np.c_[points, np.ones(len(points), dtype=np.float64)] @ matrix.T[:, :3]


def coverage_prescale(mean: np.ndarray, target: np.ndarray, coverage: float) -> float:
    mean = np.asarray(mean, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    coverage = float(np.clip(coverage, 1e-3, 1.0))
    source_diag = float(np.linalg.norm(np.ptp(mean, axis=0)))
    target_diag = float(np.linalg.norm(np.ptp(target, axis=0)))
    if source_diag <= np.finfo(float).eps:
        return 1.0
    return (target_diag / coverage) / source_diag


def fixed_scale_translation(source: np.ndarray, target: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    return (target.mean(axis=0) - source.mean(axis=0) @ rotation.T).reshape(1, 3)


def _biocpd_api() -> tuple[Callable[..., Any], type]:
    try:
        from biocpd import AtlasRegistration, pose_marginalized_initialization
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Pose EM template optimization requires biocpd 1.3.0 or newer. "
            "Upgrade biocpd in the Slicer Python environment."
        ) from exc
    return pose_marginalized_initialization, AtlasRegistration


def run_pose_em_registration(
    mean: np.ndarray,
    target: np.ndarray,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    settings: PoseEMSettings,
    *,
    max_iterations: int,
    tolerance: float,
    with_scale: bool,
    use_kdtree: bool = True,
    k_neighbors: int = 10,
    source_scale: float = 1.0,
    initializer: Optional[Callable[..., Any]] = None,
    registration_class: Optional[type] = None,
) -> PoseEMRegistrationResult:
    mean, modes, eigenvalues = validate_ssm(mean, modes, eigenvalues)
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3 or len(target) == 0 or not np.isfinite(target).all():
        raise ValueError("Target must have non-empty finite shape (N, 3)")
    settings.validate()
    if source_scale <= 0 or not np.isfinite(source_scale):
        raise ValueError("source_scale must be finite and positive")

    working_mean = mean * float(source_scale)
    working_modes = modes * float(source_scale)
    if initializer is None or registration_class is None:
        default_initializer, default_registration = _biocpd_api()
        initializer = default_initializer if initializer is None else initializer
        registration_class = default_registration if registration_class is None else registration_class

    initial = initializer(
        working_mean,
        target,
        working_modes,
        eigenvalues,
        **settings.initializer_kwargs(),
    )
    coefficients = np.asarray(initial.coefficients, dtype=np.float64).reshape(-1)
    rotation = np.asarray(initial.rotation, dtype=np.float64)
    initial_shape = ssm_sample(working_mean, working_modes, coefficients)
    if with_scale:
        scale = float(initial.scale)
        translation = np.asarray(initial.translation, dtype=np.float64).reshape(1, 3)
    else:
        scale = 1.0
        translation = fixed_scale_translation(initial_shape, target, rotation)

    registration = registration_class(
        X=target,
        Y=working_mean,
        U=working_modes,
        eigenvalues=eigenvalues,
        lambda_reg=settings.lambda_reg,
        normalize=True,
        optimize_similarity=True,
        with_scale=bool(with_scale),
        use_kdtree=bool(use_kdtree),
        k=max(1, int(k_neighbors)),
        w=settings.outlier_weight,
        max_iterations=max(1, int(max_iterations)),
        tolerance=max(0.0, float(tolerance)),
        dtype=np.float32,
    )
    registration.set_initial_state(
        coefficients,
        rotation,
        scale,
        translation,
        world_units=True,
    )
    points, final_parameters = registration.register()
    final_coefficients = np.asarray(final_parameters.get("b", coefficients), dtype=np.float64).reshape(-1)
    final_rotation = np.asarray(final_parameters.get("R_world", rotation), dtype=np.float64)
    final_scale = float(final_parameters.get("s_world", scale))
    final_translation = np.asarray(final_parameters.get("t_world", translation), dtype=np.float64).reshape(1, 3)
    return PoseEMRegistrationResult(
        points=np.asarray(points, dtype=np.float64),
        coefficients=final_coefficients,
        rotation=final_rotation,
        scale=final_scale,
        translation=final_translation,
        score=float(initial.score),
        score_margin=float(initial.score_margin),
        posterior_entropy=float(initial.posterior_entropy),
        effective_hypotheses=float(initial.effective_hypotheses),
        hypotheses_evaluated=int(initial.hypotheses_evaluated),
        hypotheses_refined=int(initial.hypotheses_refined),
        final_parameters=final_parameters,
    )
