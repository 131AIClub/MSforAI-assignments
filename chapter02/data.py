import csv
import os
from typing import Tuple

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def ensure_data_dir() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def create_sample_csv(
    output_path: str,
    num_rows: int = 32,
    seed: int = 42,
    with_dirty_rows: bool = True,
) -> str:
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "X", "Y"])
        for i in range(1, num_rows + 1):
            x = float(rng.normal(0.0, 3.0))
            y = float(rng.normal(0.0, 3.0))
            writer.writerow([i, x, y])
        if with_dirty_rows:
            writer.writerow([num_rows + 1, "", 1.0])
            writer.writerow([num_rows + 2, "abc", 2.0])
            writer.writerow([1, 100.0, 99.0])
    return output_path


def default_demo_paths() -> Tuple[str, str, str]:
    base = ensure_data_dir()
    input_path = os.path.join(base, "exercise2_input.csv")
    output_csv_path = os.path.join(base, "exercise2_output.csv")
    output_plot_path = os.path.join(base, "exercise2_plot.png")
    return input_path, output_csv_path, output_plot_path
