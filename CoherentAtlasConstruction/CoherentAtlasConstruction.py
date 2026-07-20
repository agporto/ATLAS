from pathlib import Path
import sys

from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest

_RESOURCE_DIR = Path(__file__).resolve().parent / "Resources" / "Python"
if str(_RESOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_RESOURCE_DIR))

from ConstructionImplementation import (
    BUILDER as _ConstructionModuleBase,
    BUILDERLogic as _ConstructionLogicBase,
    BUILDERWidget as _ConstructionWidgetBase,
)


class CoherentAtlasConstruction(_ConstructionModuleBase):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Atlas Construction"
        self.parent.categories = ["CoherentAtlas"]
        self.parent.helpText = (
            "Construct a population atlas, align meshes and landmarks, and export "
            "index-consistent dense correspondences."
        ) + self.getDefaultModuleDocumentationLink()


class CoherentAtlasConstructionWidget(_ConstructionWidgetBase):
    pass


class CoherentAtlasConstructionLogic(_ConstructionLogicBase):
    pass


class CoherentAtlasConstructionTest(ScriptedLoadableModuleTest):
    def setUp(self):
        import slicer
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.assertIsNotNone(CoherentAtlasConstructionLogic())
        self.delayDisplay("CoherentAtlas Construction test passed")
