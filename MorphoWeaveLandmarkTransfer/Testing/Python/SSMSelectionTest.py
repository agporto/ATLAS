import ast
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[2]
SOURCE_PATH = MODULE_DIR / "MorphoWeaveLandmarkTransfer.py"


def _load_selection_helpers():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    helper_names = {
        "canonical_ssm_node_names",
        "fill_empty_selectors",
        "latest_complete_ssm_set",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in (
        "canonical_ssm_node_names",
        "fill_empty_selectors",
        "latest_complete_ssm_set",
    ))


canonical_ssm_node_names, fill_empty_selectors, latest_complete_ssm_set = _load_selection_helpers()


class _Node:
    def __init__(self, name, npoints=None, **attributes):
        self.name = name
        self.npoints = npoints
        self.attributes = attributes

    def GetName(self):
        return self.name

    def GetAttribute(self, name):
        return self.attributes.get(name)

    def GetNumberOfControlPoints(self):
        return self.npoints


class _Selector:
    def __init__(self, node=None):
        self.node = node

    def currentNode(self):
        return self.node

    def setCurrentNode(self, node):
        self.node = node


class SSMSelectionUnitTest(unittest.TestCase):
    def test_latest_complete_point_validated_quartet_wins(self):
        old = canonical_ssm_node_names("old")
        new = canonical_ssm_node_names("new")
        tables = [
            _Node(old["table"], ssm_npoints="10"),
            _Node(new["table"], ssm_npoints="12"),
        ]
        models = [_Node(old["model"]), _Node(new["model"])]
        landmarks = [
            _Node(old["dense"], 10),
            _Node(old["sparse"], 4),
            _Node(new["dense"], 11),
            _Node(new["sparse"], 5),
        ]
        self.assertIs(latest_complete_ssm_set(tables, models, landmarks)["table"], tables[0])

        landmarks[2].npoints = 12
        resolved = latest_complete_ssm_set(tables, models, landmarks)
        self.assertIs(resolved["table"], tables[1])
        self.assertIs(resolved["dense"], landmarks[2])

    def test_incomplete_quartet_is_ignored(self):
        names = canonical_ssm_node_names("mouse")
        table = _Node(names["table"], ssm_npoints="10")
        model = _Node(names["model"])
        dense = _Node(names["dense"], 10)
        self.assertIsNone(latest_complete_ssm_set([table], [model], [dense]))

    def test_only_empty_selectors_are_filled(self):
        manual = _Node("manual")
        canonical = _Node("canonical")
        empty = _Selector()
        selected = _Selector(manual)
        self.assertTrue(fill_empty_selectors(((empty, canonical), (selected, canonical))))
        self.assertIs(empty.currentNode(), canonical)
        self.assertIs(selected.currentNode(), manual)

    def test_settings_boxes_are_explicitly_collapsed(self):
        source = (MODULE_DIR / "MorphoWeaveLandmarkTransfer.py").read_text(encoding="utf-8")
        self.assertIn("self.legacyOptimizationBox.collapsed=True", source)
        self.assertIn("self.poseOptimizationBox.collapsed=True", source)
        self.assertIn('if name in ("Rigid registration", "Deformation backend"):', source)
        backend_handler = source.split("  def _onOptimizationBackendChanged", 1)[1].split(
            "  # ----- UI setup -----", 1
        )[0]
        self.assertNotIn(".collapsed", backend_handler)

    def test_canonical_resolver_does_not_assign_target_selectors(self):
        source = (MODULE_DIR / "MorphoWeaveLandmarkTransfer.py").read_text(encoding="utf-8")
        resolver = source.split("  def _autoSelectCanonicalSsmSet(self):", 1)[1].split(
            "  def enter(self):", 1
        )[0]
        self.assertNotIn("targetModelSelector", resolver)
        self.assertNotIn("d2targetModelSelector", resolver)


if __name__ == "__main__":
    unittest.main()
