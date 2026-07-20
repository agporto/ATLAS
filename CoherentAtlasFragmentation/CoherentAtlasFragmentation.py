from pathlib import Path
import sys

from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest

_RESOURCE_DIR = Path(__file__).resolve().parent / "Resources" / "Python"
if str(_RESOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_RESOURCE_DIR))

from FragmentationImplementation import (
    SEGMENTATION as _FragmentationModuleBase,
    SEGMENTATIONLogic as _FragmentationLogicBase,
    SEGMENTATIONWidget as _FragmentationWidgetBase,
)


class CoherentAtlasFragmentation(_FragmentationModuleBase):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Surface Fragmentation"
        self.parent.categories = ["CoherentAtlas"]
        self.parent.helpText = (
            "Generate population-consistent surface fragments from dense "
            "correspondences and geometric features."
        ) + self.getDefaultModuleDocumentationLink()


class CoherentAtlasFragmentationWidget(_FragmentationWidgetBase):
    pass


class CoherentAtlasFragmentationLogic(_FragmentationLogicBase):
    pass


class CoherentAtlasFragmentationTest(ScriptedLoadableModuleTest):
    def setUp(self):
        import slicer
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.assertIsNotNone(CoherentAtlasFragmentationLogic())
        self.delayDisplay("CoherentAtlas Fragmentation test passed")
