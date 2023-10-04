"""
First run `perplexity.py` to generate one or more `csv` files.
This script can plot those csv files.

Usage:
python benchmark/plot_perplexity.py
python benchmark/plot_perplexity.py --features perplexity latency --title "Log perplexity & latency of Llama 2 7B as a function of input lengths"
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

FEATURE_DF_MAP = {
    "perplexity": "overall_ppl",
    "vram": "cuda_vram_allocated",
    "latency": "latency",
}
FEATURE_STYLE_MAP = {
    "perplexity": "-",
    "vram": "--",
    "latency": ":",
}
FEATURE_LABEL_MAP = {
    "perplexity": "Perplexity (log), lower is better",
    "vram": "CUDA VRAM Usage (GB), lower is better",
    "latency": "Time per token (sec), lower is better",
}


def plot(
    features: List[str],
    output_dir: str = "outputs",
    title: Optional[str] = None,
    perplexity_limit: Optional[float] = None,
    skip_first: int = 100,
):
    output_dir = Path(output_dir)

    fig, ax = plt.subplots()
    ax.set_xlabel("Input Sequence Length")

    for feature_i, feature in enumerate(features):
        # If we already plotted on this ax, make a new one
        if feature_i:
            ax = ax.twinx()

        for file in output_dir.glob("*.csv"):
            experiment = file.stem
            df = pd.read_csv(file)
            X = df["input_length"][skip_first:]
            Y = df[FEATURE_DF_MAP[feature]][skip_first:]
            if feature == "perplexity":
                Y = np.log(Y)
            ax.plot(X, Y, FEATURE_STYLE_MAP[feature], label=f"{experiment} {feature}")

        ax.set_ylabel(FEATURE_LABEL_MAP[feature])
        if perplexity_limit and feature == "perplexity":
            ax.set_ylim(top=min(ax.get_ylim()[1], perplexity_limit))

        ax.legend(loc=[1, 2, 7][feature_i])  # upper right, upper left, center right

    ax.set_title(title or "Log perplexity as a function of input lengths")
    fig.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser()
    # Where csv files have been logged
    parser.add_argument("--output_dir", type=str, default="benchmark/outputs")
    parser.add_argument(
        "--features", choices=["perplexity", "vram", "latency"], nargs="+", default=["perplexity", "vram"]
    )
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--log_perplexity_limit", type=float, default=5.0)
    # Perplexity starts a bit unstable, so we skip the start
    parser.add_argument("--skip_first", type=int, default=100)

    args = parser.parse_args()

    figure = plot(
        args.features,
        output_dir=args.output_dir,
        title=args.title,
        perplexity_limit=args.log_perplexity_limit,
        skip_first=args.skip_first,
    )

    # Add your own code here if you'd like to change the figure

    plt.show()


if __name__ == "__main__":
    main()
