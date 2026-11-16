import json
import re
import unittest
from pathlib import Path


class NotebookStructureTest(unittest.TestCase):
    def test_notebook_is_valid_and_code_cells_compile(self):
        notebook_path = Path(__file__).parents[1] / "notebooks" / "template2.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if source.lstrip().startswith("%"):
                continue
            compile(source, f"notebook-cell-{index}", "exec")

    def test_test_set_is_evaluated_only_after_model_selection(self):
        notebook_path = Path(__file__).parents[1] / "notebooks" / "template2.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = {
            cell.get("id"): "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        }
        code_cell_ids = [
            cell.get("id") for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]

        comparison = code_cells["fit-dual-target-models"]
        tuning = code_cells["tune-dual-target-models"]
        final_evaluation = code_cells["dual-target-test-predictions"]
        importance = code_cells["permutation-importance"]

        self.assertNotIn("predict(X_test)", comparison)
        self.assertNotIn("X_test", tuning)
        self.assertNotIn("y_test", tuning)
        self.assertNotIn("X_test", importance)
        self.assertNotIn("y_test", importance)
        self.assertIn("predict(X_test)", final_evaluation)
        self.assertEqual(final_evaluation.count("predict(X_test)"), 1)
        self.assertLess(
            code_cell_ids.index("tune-dual-target-models"),
            code_cell_ids.index("dual-target-test-predictions"),
        )

        scoring_cells = [
            source for source in code_cells.values()
            if re.search(r"(?:r2_score|mean_absolute_error|mean_squared_error)\(y_test", source)
        ]
        self.assertEqual(len(scoring_cells), 1)


if __name__ == "__main__":
    unittest.main()