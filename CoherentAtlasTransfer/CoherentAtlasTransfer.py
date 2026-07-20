from pathlib import Path
import importlib
import sys
import types

from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest

_RESOURCE_DIR = Path(__file__).resolve().parent / "Resources" / "Python"
if str(_RESOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_RESOURCE_DIR))

# Preserve the legacy helper import contract while the implementation is migrated.
_pose_em = importlib.import_module("PoseEMTemplate")
_resources_package = sys.modules.setdefault("Resources", types.ModuleType("Resources"))
_resources_package.__path__ = []
_python_package = sys.modules.setdefault("Resources.Python", types.ModuleType("Resources.Python"))
_python_package.__path__ = []
sys.modules["Resources.Python.PoseEMTemplate"] = _pose_em

from LegacyPredict import PREDICT, PREDICTLogic, PREDICTWidget


class CoherentAtlasTransfer(PREDICT):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Landmark Transfer"
        self.parent.categories = ["CoherentAtlas"]
        self.parent.helpText = (
            "Transfer landmarks to complete or partial target surfaces using rigid "
            "alignment, statistical shape model optimization, coherent deformable "
            "registration, and optional surface projection."
        ) + self.getDefaultModuleDocumentationLink()


class CoherentAtlasTransferWidget(PREDICTWidget):
    pass


class CoherentAtlasTransferLogic(PREDICTLogic):
    pass


class CoherentAtlasTransferTest(ScriptedLoadableModuleTest):
    def setUp(self):
        import slicer
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.assertIsNotNone(CoherentAtlasTransferLogic())
        self.delayDisplay("CoherentAtlas Transfer test passed")
