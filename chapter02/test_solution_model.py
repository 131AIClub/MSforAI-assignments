import csv
import importlib
import os
import sys
import tempfile
import unittest

import numpy as np


def load_solution_module():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    solution_dir = os.path.join(base_dir, "solution")
    if solution_dir not in sys.path:
        sys.path.insert(0, solution_dir)
    if "model" in sys.modules:
        del sys.modules["model"]
    import model

    return importlib.reload(model)


class TestSolutionExercise2(unittest.TestCase):
    def setUp(self):
        self.model = load_solution_module()
        self.processor = self.model.Exercise2Processor()
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_rows(self, file_name, rows):
        path = os.path.join(self.tmpdir.name, file_name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "X", "Y"])
            writer.writerows(rows)
        return path

    def test_feature_matches_formula(self):
        x = np.array([2.0, -1.0, 0.0], dtype=np.float64)
        y = np.array([1.0, 3.0, 0.0], dtype=np.float64)
        z = self.processor.compute_feature(x, y)
        expected = np.log(np.square(x - y) + 1e-12)
        np.testing.assert_allclose(z, expected, rtol=1e-9, atol=1e-12)

    def test_clean_and_sort_behavior(self):
        input_path = self._write_rows(
            "clean_sort.csv",
            [
                [3, 9.0, 1.0],
                [1, 2.0, 5.0],
                [3, 1.0, 1.0],
                [2, "bad", 1.0],
            ],
        )
        indices, x_vals, y_vals, total_rows, dropped_rows = self.processor.load_and_clean(input_path)
        self.assertEqual(total_rows, 4)
        self.assertEqual(dropped_rows, 1)
        np.testing.assert_array_equal(indices, np.array([1, 3], dtype=np.int64))
        np.testing.assert_allclose(x_vals, np.array([2.0, 1.0]))
        np.testing.assert_allclose(y_vals, np.array([5.0, 1.0]))

    def test_run_outputs_are_created(self):
        input_path = self._write_rows(
            "run.csv",
            [
                [1, 1.0, 3.0],
                [2, -2.0, -3.0],
                [3, 4.0, 4.5],
            ],
        )
        out_csv = os.path.join(self.tmpdir.name, "result.csv")
        out_png = os.path.join(self.tmpdir.name, "result.png")
        report = self.processor.run(input_path, out_csv, out_png)
        self.assertTrue(os.path.exists(out_csv))
        self.assertTrue(os.path.exists(out_png))
        self.assertEqual(report.total_rows, 3)
        self.assertEqual(report.valid_rows, 3)
        for key in ("min", "max", "mean", "std"):
            self.assertTrue(np.isfinite(report.stats[key]))


if __name__ == "__main__":
    unittest.main()
