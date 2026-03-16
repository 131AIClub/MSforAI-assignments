import csv
import importlib
import os
import sys

import numpy as np
import pytest


def load_model_module():
    if "model" in sys.modules:
        del sys.modules["model"]
    import model

    return importlib.reload(model)


def write_csv(path: str, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "X", "Y"])
        writer.writerows(rows)


class TestExercise2Processor:
    def test_solution_pipeline(self, mode, tmp_path):
        if mode != "solution":
            pytest.skip("solution only")
        model = load_model_module()
        processor = model.Exercise2Processor()
        input_path = os.path.join(tmp_path, "input.csv")
        output_csv = os.path.join(tmp_path, "output.csv")
        output_plot = os.path.join(tmp_path, "plot.png")
        write_csv(
            input_path,
            [
                [1, 1.0, 3.0],
                [2, 5.0, 2.0],
                [3, -1.0, -4.0],
            ],
        )
        report = processor.run(input_path, output_csv, output_plot)
        assert report.total_rows == 3
        assert report.valid_rows == 3
        assert report.dropped_rows == 0
        assert os.path.exists(output_csv)
        assert os.path.exists(output_plot)

    def test_solution_duplicate_index_keep_last(self, mode, tmp_path):
        if mode != "solution":
            pytest.skip("solution only")
        model = load_model_module()
        processor = model.Exercise2Processor()
        input_path = os.path.join(tmp_path, "input_dup.csv")
        output_csv = os.path.join(tmp_path, "output_dup.csv")
        output_plot = os.path.join(tmp_path, "plot_dup.png")
        write_csv(
            input_path,
            [
                [1, 1.0, 3.0],
                [1, 10.0, 8.0],
                [2, 5.0, 2.0],
            ],
        )
        report = processor.run(input_path, output_csv, output_plot)
        assert report.total_rows == 3
        assert report.valid_rows == 2
        with open(output_csv, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        assert len(lines) == 3
        assert lines[1].startswith("1,")

    def test_solution_invalid_rows_drop(self, mode, tmp_path):
        if mode != "solution":
            pytest.skip("solution only")
        model = load_model_module()
        processor = model.Exercise2Processor()
        input_path = os.path.join(tmp_path, "input_dirty.csv")
        output_csv = os.path.join(tmp_path, "output_dirty.csv")
        output_plot = os.path.join(tmp_path, "plot_dirty.png")
        write_csv(
            input_path,
            [
                [1, 1.0, 3.0],
                [2, "abc", 2.0],
                [3, "", 2.0],
                [4, 0.0, 0.0],
            ],
        )
        report = processor.run(input_path, output_csv, output_plot)
        assert report.total_rows == 4
        assert report.valid_rows == 2
        assert report.dropped_rows == 2
        assert np.isfinite(report.stats["mean"])

    def test_solution_missing_file(self, mode, tmp_path):
        if mode != "solution":
            pytest.skip("solution only")
        model = load_model_module()
        processor = model.Exercise2Processor()
        with pytest.raises(FileNotFoundError):
            processor.run(
                os.path.join(tmp_path, "not_found.csv"),
                os.path.join(tmp_path, "o.csv"),
                os.path.join(tmp_path, "o.png"),
            )

    def test_solution_stability_extreme_values(self, mode, tmp_path):
        if mode != "solution":
            pytest.skip("solution only")
        model = load_model_module()
        processor = model.Exercise2Processor()
        input_path = os.path.join(tmp_path, "extreme.csv")
        output_csv = os.path.join(tmp_path, "extreme_out.csv")
        output_plot = os.path.join(tmp_path, "extreme_out.png")
        write_csv(
            input_path,
            [
                [1, 1e154, -1e154],
                [2, -1e154, 1e154],
                [3, 0.0, 0.0],
            ],
        )
        report = processor.run(input_path, output_csv, output_plot)
        assert report.valid_rows == 3
        assert np.isfinite(report.stats["min"])
        assert np.isfinite(report.stats["max"])

    def test_startup_template_not_implemented(self, mode):
        if mode != "startup":
            pytest.skip("startup only")
        model = load_model_module()
        processor = model.Exercise2Processor()
        with pytest.raises(NotImplementedError):
            processor.run("a.csv", "b.csv", "c.png")
