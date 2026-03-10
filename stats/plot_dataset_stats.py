"""Minimal plotting helpers for ligand-kinase search-space analysis.

Expected inputs:
- molecules_per_kinase: pd.Series indexed by kinase_id, values = n_molecules
- kinases_per_molecule: pd.Series indexed by ligand_id, values = n_kinases
- pairs: pd.DataFrame with columns ['ligand_id', 'kinase_id']
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    """Save figure if path is provided; otherwise show it."""
    if save_path is None:
        plt.show()
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(
    series: pd.Series,
    title: str,
    xlabel: str,
    bins: int = 60,
    log_x: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Histogram for a degree distribution."""
    values = series.values.astype(float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color="#4C78A8", alpha=0.85, edgecolor="white")
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    _save_or_show(fig, save_path)


def plot_rank_frequency(
    series: pd.Series,
    title: str,
    ylabel: str,
    save_path: str | Path | None = None,
) -> None:
    """Rank-frequency plot on log-log axes."""
    values = np.sort(series.values)[::-1]
    ranks = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranks, values, color="#54A24B", linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Rank")
    ax.set_ylabel(ylabel)
    _save_or_show(fig, save_path)


def plot_coverage_curve(
    series: pd.Series,
    title: str = "Coverage curve",
    ylabel: str = "Cumulative fraction",
    save_path: str | Path | None = None,
) -> None:
    """Cumulative coverage over sorted counts."""
    values = np.sort(series.values)[::-1]
    cumulative = np.cumsum(values) / values.sum()
    x = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, cumulative, color="#E45756", linewidth=2)
    ax.set_ylim(0, 1.01)
    ax.set_title(title)
    ax.set_xlabel("Top-N entities")
    ax.set_ylabel(ylabel)
    _save_or_show(fig, save_path)


def plot_shared_kinase_heatmap(
    pairs: pd.DataFrame,
    top_k: int = 20,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Heatmap of shared ligands between top-k kinases by ligand count."""
    top_kinases = pairs["kinase_id"].value_counts().head(top_k).index
    sub = pairs[pairs["kinase_id"].isin(top_kinases)]

    matrix = pd.crosstab(sub["ligand_id"], sub["kinase_id"]).astype(bool).astype(int)
    shared = matrix.T.dot(matrix)

    values = shared.values
    positive = values[values > 0]
    vmin = int(positive.min()) if positive.size else 1
    vmax = int(values.max()) if values.size else 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(values, cmap="viridis", norm=LogNorm(vmin=max(1, vmin), vmax=max(1, vmax)))
    ax.set_xticks(np.arange(len(shared.columns)))
    ax.set_yticks(np.arange(len(shared.index)))
    ax.set_xticklabels(shared.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(shared.index, fontsize=8)
    ax.set_title(f"Shared ligands between top {top_k} kinases")
    fig.colorbar(im, ax=ax, label="Shared ligand count (log)")
    _save_or_show(fig, save_path)
    return shared


def plot_pair_degree_hexbin(
    pairs: pd.DataFrame,
    gridsize: int = 45,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Hexbin of pair-level degree context."""
    mol_deg = pairs.groupby("ligand_id")["kinase_id"].nunique().rename("n_kinases_for_ligand")
    kin_deg = pairs.groupby("kinase_id")["ligand_id"].nunique().rename("n_ligands_for_kinase")
    edge_view = pairs.join(mol_deg, on="ligand_id").join(kin_deg, on="kinase_id")

    fig, ax = plt.subplots(figsize=(8, 5))
    hb = ax.hexbin(
        edge_view["n_kinases_for_ligand"],
        edge_view["n_ligands_for_kinase"],
        gridsize=gridsize,
        bins="log",
        cmap="magma",
        mincnt=1,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("# kinases for molecule")
    ax.set_ylabel("# molecules for kinase")
    ax.set_title("Edge-level degree landscape")
    fig.colorbar(hb, ax=ax, label="log10(edge count per hexbin)")
    _save_or_show(fig, save_path)
    return edge_view
