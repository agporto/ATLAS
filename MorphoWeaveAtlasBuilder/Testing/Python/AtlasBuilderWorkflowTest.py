import ast
import os
import tempfile
import unittest
from datetime import datetime
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


def _widget_method(name):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    widget_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MorphoWeaveAtlasBuilderWidget"
    )
    function = next(
        node for node in widget_class.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"os": os, "datetime": datetime}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace[name]


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

    def test_aligned_outputs_can_be_routed_outside_final_output(self):
        out_folders = _widget_method("_outFolders")
        with tempfile.TemporaryDirectory() as output_root, tempfile.TemporaryDirectory() as aligned_root:
            folders = out_folders(object(), output_root, aligned_root)
            self.assertEqual(Path(folders["alignedModels"]).parent, Path(aligned_root))
            self.assertEqual(Path(folders["alignedLMs"]).parent, Path(aligned_root))
            self.assertEqual(Path(folders["atlas"]).parent, Path(folders["output"]))
            self.assertFalse((Path(folders["output"]) / "alignedModels").exists())

    def test_aligned_output_retention_is_default_and_temp_workspace_is_cleaned(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertIn('self.keepAlignedOutputs.setChecked(True)', source)
        self.assertIn('tempfile.TemporaryDirectory(prefix="MorphoWeave-aligned-")', source)
        self.assertIn("if alignedWorkspace is not None:", source)
        self.assertIn("alignedWorkspace.cleanup()", source)
        self.assertIn("if keepAlignedOutputs:", source)


if __name__ == "__main__":
    unittest.main()
