"""Small shared plotting helpers used by the notebook examples."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_curves(curves: dict[str, list[float]], title: str, ylabel: str, xlabel: str = "epoch"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in curves.items():
        ax.plot(values, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig
