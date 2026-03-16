from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.table import Table

import data

Processor = None


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_modules(mode: str) -> None:
    global Processor
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, mode)
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)
    if "model" in sys.modules:
        del sys.modules["model"]
    import model

    importlib.reload(model)
    Processor = model.Exercise2Processor


def run_pytest(args_list: list[str], mode: str) -> int:
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_bench.py")
    pytest_args = ["-v", test_file, "--mode", mode] + args_list
    return pytest.main(pytest_args)


def run_exercise(mode: str) -> None:
    load_modules(mode)
    logger = logging.getLogger("Exercise2Runner")
    input_path, output_csv_path, output_plot_path = data.default_demo_paths()
    if not os.path.exists(input_path):
        data.create_sample_csv(input_path)
    processor = Processor(logger=logger)
    report = processor.run(input_path, output_csv_path, output_plot_path)
    console = Console()
    table = Table(title=f"Exercise 2 Result [{mode.upper()}]")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("total_rows", str(report.total_rows))
    table.add_row("valid_rows", str(report.valid_rows))
    table.add_row("dropped_rows", str(report.dropped_rows))
    for key, value in report.stats.items():
        table.add_row(key, f"{value:.6f}")
    console.print(table)
    console.print(f"CSV 输出: {output_csv_path}")
    console.print(f"图像输出: {output_plot_path}")


def get_args():
    parser = argparse.ArgumentParser(description="Chapter02 Exercise2 Runner")
    parser.add_argument("--mode", type=str, choices=["startup", "solution"])
    return parser.parse_args()


def select_mode() -> str:
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", width=4)
    table.add_column("Mode", min_width=16)
    table.add_column("Description")
    table.add_row("1", "Startup", "Run student template")
    table.add_row("2", "Solution", "Run reference implementation")
    console.print(Panel(table, title="Mode Selection", border_style="blue"))
    choice = IntPrompt.ask("Select mode", choices=["1", "2"], default=2)
    return "startup" if choice == 1 else "solution"


def bench(mode: str) -> None:
    console = Console()
    while True:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", width=4)
        table.add_column("Action", min_width=20)
        table.add_column("Description")
        table.add_row("1", "Run All Tests", f"Run test_bench in {mode}")
        table.add_row("2", "Run Exercise", "Generate outputs and summary")
        table.add_row("0", "Exit", "Quit")
        console.print(Panel(table, title=f"Chapter02 Exercise2 [{mode.upper()}]", border_style="cyan"))
        choice = IntPrompt.ask("Select", choices=["0", "1", "2"], default="1")
        if choice == 0:
            return
        if choice == 1:
            result = run_pytest([], mode)
            style = "green" if result == 0 else "red"
            console.print(Panel(f"pytest exit code = {result}", style=style))
        if choice == 2:
            try:
                run_exercise(mode)
            except Exception as exc:
                console.print(Panel(f"运行失败: {exc}", style="red"))


def main() -> None:
    configure_logging()
    args = get_args()
    mode = args.mode or select_mode()
    bench(mode)


if __name__ == "__main__":
    main()
