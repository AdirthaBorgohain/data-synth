"""Plot helpers — save PNGs to the run's artifact store, return file paths.

Headless by design: the Agg backend is forced before pyplot is imported so plots
render without a display. Every helper closes its figure to bound memory across a
run that may emit many plots.

Each helper accepts optional `caption` (the finding's plain-English observation,
wrapped below the axes), so a PNG opened in isolation explains itself.
"""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from datasynth.artifacts.store import ArtifactStore, slugify  # noqa: E402

sns.set_theme(style="whitegrid")

_EDA_SUBDIR = "eda"
_FIG_DPI = 110
_CAPTION_WRAP = 78


def _annotate(fig, caption: str | None) -> None:
    """Add finding caption below the figure, left-aligned, with a divider."""
    if not caption:
        return
    wrapped = textwrap.fill(caption.strip(), _CAPTION_WRAP)
    # Thin divider rule between chart and caption — drawn in figure coords.
    fig.add_artist(plt.Line2D(
        [0.04, 0.96], [-0.015, -0.015],
        transform=fig.transFigure, color="#cccccc", linewidth=0.8,
    ))
    # wrap=False on purpose — textwrap above already inserted newlines at word
    # boundaries; letting matplotlib re-wrap to its clip width would chop
    # sentences mid-phrase ("...active and only\n13% have churned...").
    fig.text(
        0.04, -0.04, wrapped,
        ha="left", va="top",
        fontsize=10, color="#333333", family="DejaVu Sans",
        linespacing=1.35, wrap=False,
    )


def _save(fig, artifacts: ArtifactStore, slug: str,
          caption: str | None = None) -> str:
    _annotate(fig, caption)
    path = artifacts.path(_EDA_SUBDIR, f"{slugify(slug)}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def histogram(values: list, title: str, artifacts: ArtifactStore, slug: str,
              *, caption: str | None = None) -> str:
    arr = np.asarray([v for v in values if v is not None], dtype="float64")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(arr, bins=min(40, max(10, int(np.sqrt(arr.size)) if arr.size else 10)),
            color="#4C78A8", edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("count")
    return _save(fig, artifacts, slug, caption)


def bar_counts(labels: list, counts: list[int], title: str,
               artifacts: ArtifactStore, slug: str,
               *, caption: str | None = None) -> str:
    labels = [str(x) for x in labels]
    fig, ax = plt.subplots(figsize=(6, max(3.0, 0.4 * len(labels) + 1)))
    sns.barplot(x=counts, y=labels, ax=ax, color="#4C78A8", orient="h")
    ax.set_title(title)
    ax.set_xlabel("count")
    return _save(fig, artifacts, slug, caption)


def correlation_heatmap(matrix: list[list[float]], labels: list[str],
                        artifacts: ArtifactStore, slug: str,
                        *, caption: str | None = None) -> str:
    arr = np.asarray(matrix, dtype="float64")
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(4.0, 0.6 * n + 2), max(3.5, 0.6 * n + 1.5)))
    sns.heatmap(
        arr, xticklabels=labels, yticklabels=labels, annot=n <= 12,
        fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0,
        square=True, cbar_kws={"shrink": 0.7}, ax=ax,
    )
    ax.set_title("Correlation matrix")
    return _save(fig, artifacts, slug, caption)


def missingness_bar(columns: list[str], null_rates: list[float],
                    artifacts: ArtifactStore, slug: str = "missingness",
                    *, caption: str | None = None) -> str:
    fig, ax = plt.subplots(figsize=(6, max(3.0, 0.4 * len(columns) + 1)))
    colors = ["#E45756" if r >= 0.4 else "#4C78A8" for r in null_rates]
    sns.barplot(x=[r * 100 for r in null_rates], y=columns, ax=ax,
                palette=colors, orient="h", hue=columns, legend=False)
    ax.set_title("Missingness by column")
    ax.set_xlabel("null %")
    ax.set_xlim(0, 100)
    return _save(fig, artifacts, slug, caption)


def boxplot(values: list, title: str, artifacts: ArtifactStore, slug: str,
            *, caption: str | None = None) -> str:
    arr = np.asarray([v for v in values if v is not None], dtype="float64")
    fig, ax = plt.subplots(figsize=(5, 3.0))
    sns.boxplot(x=arr, ax=ax, color="#4C78A8")
    ax.set_title(title)
    return _save(fig, artifacts, slug, caption)


def grouped_box(groups: dict[str, list], title: str, xlabel: str,
                artifacts: ArtifactStore, slug: str,
                *, caption: str | None = None) -> str:
    """Feature distribution split by target class (classification target-vs-feature)."""
    data, labels = [], []
    for label, vals in groups.items():
        arr = np.asarray([v for v in vals if v is not None], dtype="float64")
        if arr.size:
            data.append(arr)
            labels.append(str(label))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.boxplot(data, tick_labels=labels, patch_artist=True)
    ax.set_title(title)
    ax.set_ylabel(xlabel)
    return _save(fig, artifacts, slug, caption)


def line_counts(x: list, y: list[int], title: str, xlabel: str,
                artifacts: ArtifactStore, slug: str,
                *, caption: str | None = None) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([str(v) for v in x], y, marker="o", color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("row count")
    fig.autofmt_xdate(rotation=45)
    return _save(fig, artifacts, slug, caption)
