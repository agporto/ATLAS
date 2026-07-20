from pathlib import Path
import importlib
import inspect
import sys
import types

import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest

_RESOURCE_DIR = Path(__file__).resolve().parent / "Resources" / "Python"
if str(_RESOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_RESOURCE_DIR))

# Preserve the private helper import contract used by the transfer implementation.
_pose_em = importlib.import_module("PoseEMTemplate")
_resources_package = sys.modules.setdefault("Resources", types.ModuleType("Resources"))
_resources_package.__path__ = []
_python_package = sys.modules.setdefault("Resources.Python", types.ModuleType("Resources.Python"))
_python_package.__path__ = []
sys.modules["Resources.Python.PoseEMTemplate"] = _pose_em

from TransferImplementation import (
    PREDICT as _TransferModuleBase,
    PREDICTLogic as _TransferLogicBase,
    PREDICTWidget as _TransferWidgetBase,
)


class CoherentAtlasTransfer(_TransferModuleBase):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Landmark Transfer"
        self.parent.categories = ["CoherentAtlas"]
        self.parent.helpText = (
            "Transfer landmarks to complete or partial target surfaces using rigid "
            "alignment, statistical shape model optimization, coherent deformable "
            "registration, and optional surface projection."
        ) + self.getDefaultModuleDocumentationLink()


class CoherentAtlasTransferWidget(_TransferWidgetBase):
    def _ensure_deps_async(self):
        """Install Python dependencies through Slicer's supported installer."""
        required = [("tiny3d", "tiny3d"), ("biocpd", "biocpd>=1.3")]

        def has_pose_api():
            try:
                module = importlib.import_module("biocpd")
                initializer = getattr(module, "pose_marginalized_initialization")
                parameters = inspect.signature(initializer).parameters
                return all(
                    name in parameters
                    for name in (
                        "coarse_screen_iterations",
                        "coarse_survivor_count",
                        "coarse_score_mode",
                        "refine_source_count",
                        "n_jobs",
                    )
                )
            except Exception:
                return False

        missing = []
        for module_name, _ in required:
            try:
                available = importlib.util.find_spec(module_name) is not None
            except Exception:
                available = False
            if not available or (module_name == "biocpd" and not has_pose_api()):
                missing.append(module_name)

        if missing:
            specifications = [spec for name, spec in required if name in missing]
            message = "Landmark Transfer requires: " + ", ".join(specifications) + ".\nInstall now?"
            if not slicer.util.confirmOkCancelDisplay(message):
                self._deps_ready = False
                slicer.util.errorDisplay("Dependencies were not installed; transfer actions may fail.")
                return
            try:
                slicer.util.pip_install(" ".join(specifications))
                for loaded_name in list(sys.modules):
                    if any(
                        loaded_name == package_name or loaded_name.startswith(package_name + ".")
                        for package_name in missing
                    ):
                        sys.modules.pop(loaded_name, None)
                importlib.invalidate_caches()
            except Exception as exc:
                self._deps_ready = False
                self._deps_error = exc
                slicer.util.errorDisplay(f"Dependency installation failed:\n{exc}")
                return

        try:
            for module_name in ("tiny3d", "biocpd", "scipy.spatial", "scipy.optimize"):
                importlib.import_module(module_name)
            if not has_pose_api():
                raise RuntimeError("The installed biocpd package does not provide the required Pose-EM API.")
        except Exception as exc:
            self._deps_ready = False
            self._deps_error = exc
            slicer.util.errorDisplay(f"Dependency validation failed:\n{exc}")
            return

        self._deps_error = None
        self._deps_ready = True
        slicer.util.showStatusMessage("CoherentAtlas Landmark Transfer dependencies ready", 3000)


class CoherentAtlasTransferLogic(_TransferLogicBase):
    pass


class CoherentAtlasTransferTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.assertIsNotNone(CoherentAtlasTransferLogic())
        self.delayDisplay("CoherentAtlas Transfer test passed")
