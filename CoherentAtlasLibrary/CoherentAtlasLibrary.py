from pathlib import Path
import sys

from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest

_RESOURCE_DIR = Path(__file__).resolve().parent / "Resources" / "Python"
if str(_RESOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_RESOURCE_DIR))

from LibraryImplementation import (
    DATABASE as _LibraryModuleBase,
    DATABASELogic as _LibraryLogicBase,
    DATABASEWidget as _LibraryWidgetBase,
)


class CoherentAtlasLibrary(_LibraryModuleBase):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Atlas Library"
        self.parent.categories = ["CoherentAtlas"]
        self.parent.helpText = (
            "Create, store, load, and explore reusable statistical shape model "
            "libraries for CoherentAtlas workflows."
        ) + self.getDefaultModuleDocumentationLink()


class CoherentAtlasLibraryWidget(_LibraryWidgetBase):
    pass


class CoherentAtlasLibraryLogic(_LibraryLogicBase):
    pass


class CoherentAtlasLibraryTest(ScriptedLoadableModuleTest):
    def setUp(self):
        import slicer
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.assertIsNotNone(CoherentAtlasLibraryLogic())
        self.delayDisplay("CoherentAtlas Library test passed")
