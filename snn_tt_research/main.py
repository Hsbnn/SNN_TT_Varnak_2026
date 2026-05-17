"""Unified command-line entry point for the three research blocks.

Examples:
    python -m snn_tt_research.main block1
    python -m snn_tt_research.main block2 --data-root data
    python -m snn_tt_research.main block3
    python -m snn_tt_research.main all
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict

from .block1_iris.run import run_block1
from .block2_mnist_stdp.run import run_block2
from .block3_fashion_surgrad.run import run_block3


def _build_dispatch(data_root: str, plots_dir: str) -> Dict[str, Callable]:
    """Map block names to zero-arg callables that execute the chosen block.

    The ``data_root`` argument is captured so the MNIST and Fashion-MNIST
    blocks know where to download / read the torchvision datasets.  The
    ``plots_dir`` argument is the parent directory that holds per-block
    output sub-folders ``block2`` and ``block3``.
    """
    return {
        "block1": lambda: run_block1(),
        "block2": lambda: run_block2(data_root=data_root, plots_dir=f"{plots_dir}/block2"),
        "block3": lambda: run_block3(data_root=data_root, plots_dir=f"{plots_dir}/block3"),
    }


def main() -> None:
    """Parse CLI arguments and execute one of the three research blocks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "block",
        choices=("block1", "block2", "block3", "all"),
        help="Research block to execute. Use 'all' to run them sequentially.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Filesystem path that holds (or will hold) MNIST / Fashion-MNIST.",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Parent directory for figures produced by the MNIST and Fashion blocks.",
    )
    args = parser.parse_args()

    dispatch = _build_dispatch(args.data_root, args.plots_dir)
    if args.block == "all":
        for name in ("block1", "block2", "block3"):
            print(f"\n##### {name} #####")
            dispatch[name]()
        return
    dispatch[args.block]()


if __name__ == "__main__":
    main()
