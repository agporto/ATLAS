import ast
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[2]
SOURCE_PATH = MODULE_DIR / "MorphoWeaveAtlasBuilder.py"


def _safe_name_function():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    logic_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MorphoWeaveAtlasBuilderLogic"
    )
    function = next(
        node for node in logic_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "isSafeLibraryName"
    )
    function.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace["isSafeLibraryName"]


class AtlasBuilderWorkflowUnitTest(unittest.TestCase):
    def test_library_names_are_portable_and_path_safe(self):
        is_safe = _safe_name_function()
        self.assertTrue(is_safe("mouse_atlas"))
        self.assertTrue(is_safe("Mouse Atlas 2026"))
        for invalid in ("", ".", "..", "../atlas", "atlas/model", "atlas\\model", "CON", "name."):
            self.assertFalse(is_safe(invalid), invalid)

    def test_overwrite_gate_precedes_output_creation(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        on_run = source.split("  def _onRun(self):", 1)[1].split(
            "  def _saveSsmToLibrary", 1
        )[0]
        self.assertLess(
            on_run.index("confirmOkCancelDisplay"),
            on_run.index("F = self._outFolders"),
        )
        self.assertIn("if dense_ok and saveToLibrary:", on_run)

    def test_configured_library_path_supports_current_default_and_legacy_keys(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertIn('"MorphoWeaveModelLibrary/databasePath"', source)
        self.assertIn('"DATABASE/databasePath"', source)
        self.assertIn('"MorphoWeaveModels"', source)


if __name__ == "__main__":
    unittest.main()
