import csv
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ProcessReport:
    total_rows: int
    valid_rows: int
    dropped_rows: int
    stats: Dict[str, float]


class Exercise2Processor:
    def __init__(self, epsilon: float = 1e-12, logger: logging.Logger | None = None) -> None:
        self.epsilon = epsilon
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def _validate_headers(self, headers: List[str] | None) -> None:
        if headers is None:
            raise ValueError("CSV 文件缺少表头")
        required = {"index", "X", "Y"}
        if not required.issubset(set(headers)):
            raise ValueError("CSV 必须包含 index, X, Y 三列")

    def load_and_clean(self, input_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        rows_by_index: Dict[int, Tuple[float, float]] = {}
        total_rows = 0
        dropped_rows = 0

        with open(input_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._validate_headers(reader.fieldnames)
            for row in reader:
                total_rows += 1
                try:
                    idx = int(str(row["index"]).strip())
                    x = float(str(row["X"]).strip())
                    y = float(str(row["Y"]).strip())
                except (TypeError, ValueError):
                    dropped_rows += 1
                    self.logger.warning("丢弃非法行: %s", row)
                    continue
                rows_by_index[idx] = (x, y)

        if not rows_by_index:
            raise ValueError("清洗后无有效数据")

        sorted_idx = np.array(sorted(rows_by_index.keys()), dtype=np.int64)
        x_vals = np.array([rows_by_index[i][0] for i in sorted_idx], dtype=np.float64)
        y_vals = np.array([rows_by_index[i][1] for i in sorted_idx], dtype=np.float64)
        return sorted_idx, x_vals, y_vals, total_rows, dropped_rows

    def compute_feature(self, x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
        diff = x_vals - y_vals
        abs_diff = np.abs(diff)
        limit = np.sqrt(np.finfo(np.float64).max)
        z_vals = np.empty_like(abs_diff, dtype=np.float64)
        large_mask = abs_diff > limit
        small_mask = ~large_mask
        z_vals[small_mask] = np.log(np.square(abs_diff[small_mask]) + self.epsilon)
        z_vals[large_mask] = 2.0 * np.log(abs_diff[large_mask])
        return z_vals

    def summarize(self, z_vals: np.ndarray) -> Dict[str, float]:
        return {
            "count": float(z_vals.shape[0]),
            "min": float(np.min(z_vals)),
            "max": float(np.max(z_vals)),
            "mean": float(np.mean(z_vals)),
            "std": float(np.std(z_vals)),
        }

    def save_csv(self, indices: np.ndarray, z_vals: np.ndarray, output_csv_path: str) -> None:
        output_dir = os.path.dirname(os.path.abspath(output_csv_path))
        os.makedirs(output_dir, exist_ok=True)
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "Z"])
            for i, z in zip(indices.tolist(), z_vals.tolist()):
                writer.writerow([i, z])

    def save_plot(self, indices: np.ndarray, z_vals: np.ndarray, output_plot_path: str) -> None:
        output_dir = os.path.dirname(os.path.abspath(output_plot_path))
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(indices, z_vals, marker="o")
        plt.xlabel("index")
        plt.ylabel("Z")
        plt.title("Z = log((X-Y)^2 + eps)")
        plt.tight_layout()
        plt.savefig(output_plot_path)
        plt.close()

    def run(self, input_path: str, output_csv_path: str, output_plot_path: str) -> ProcessReport:
        indices, x_vals, y_vals, total_rows, dropped_rows = self.load_and_clean(input_path)
        z_vals = self.compute_feature(x_vals, y_vals)
        stats = self.summarize(z_vals)
        self.save_csv(indices, z_vals, output_csv_path)
        self.save_plot(indices, z_vals, output_plot_path)
        valid_rows = int(stats["count"])
        return ProcessReport(
            total_rows=total_rows,
            valid_rows=valid_rows,
            dropped_rows=dropped_rows + max(0, total_rows - valid_rows - dropped_rows),
            stats=stats,
        )
