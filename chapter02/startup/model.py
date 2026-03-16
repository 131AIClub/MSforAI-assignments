import logging
from dataclasses import dataclass
from typing import Dict

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

    def load_and_clean(self, input_path: str):
        raise NotImplementedError("请在 startup/model.py 中实现 load_and_clean")

    def compute_feature(self, x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
        raise NotImplementedError("请在 startup/model.py 中实现 compute_feature")

    def summarize(self, z_vals: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError("请在 startup/model.py 中实现 summarize")

    def save_csv(self, indices: np.ndarray, z_vals: np.ndarray, output_csv_path: str) -> None:
        raise NotImplementedError("请在 startup/model.py 中实现 save_csv")

    def save_plot(self, indices: np.ndarray, z_vals: np.ndarray, output_plot_path: str) -> None:
        raise NotImplementedError("请在 startup/model.py 中实现 save_plot")

    def run(self, input_path: str, output_csv_path: str, output_plot_path: str) -> ProcessReport:
        raise NotImplementedError("请在 startup/model.py 中实现 run")
