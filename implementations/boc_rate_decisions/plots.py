"""Plotting helpers for the BoC rate-decision experiment.

Centralises the matplotlib boilerplate so the notebooks stay narrative.
All plots use matplotlib directly (no seaborn / plotly) to minimise
dependencies. Each helper returns the ``(fig, ax)`` pair it created so the
caller can further customise or save the figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .analysis import calibration_table


DEFAULT_PREDICTOR_PALETTE: list[str] = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]
"""Default colour palette for up to six predictors."""


def _resolve_colors(predictors: list[str], colors: dict[str, str] | None) -> dict[str, str]:
    """Return a ``predictor_id -> colour`` map covering every predictor."""
    resolved: dict[str, str] = dict(colors or {})
    next_idx = 0
    for pid in predictors:
        if pid in resolved:
            continue
        resolved[pid] = DEFAULT_PREDICTOR_PALETTE[next_idx % len(DEFAULT_PREDICTOR_PALETTE)]
        next_idx += 1
    return resolved


def _resolve_labels(predictors: list[str], labels: dict[str, str] | None) -> dict[str, str]:
    """Return a ``predictor_id -> display label`` map for legends."""
    return {pid: (labels or {}).get(pid, pid) for pid in predictors}


# ---------------------------------------------------------------------------
# Exploration: policy rate path with decision markers
# ---------------------------------------------------------------------------


def plot_policy_rate_with_decisions(
    rate_df: pd.DataFrame,
    event_df: pd.DataFrame,
    *,
    start: pd.Timestamp | None = None,
) -> tuple[Figure, Axes]:
    """Plot the daily target rate with each announcement marked by its outcome.

    Cuts are red down-triangles, holds/hikes are light grey dots — the visual
    motivation for the whole experiment: cuts are rare and strongly clustered
    into easing cycles.

    Parameters
    ----------
    rate_df : pd.DataFrame
        Daily target-rate series (``timestamp`` / ``value`` columns).
    event_df : pd.DataFrame
        Per-meeting 0/1 event series.
    start : pd.Timestamp or None
        Optional left cutoff for the x-axis.

    Returns
    -------
    (Figure, Axes)
    """
    rate = rate_df.copy()
    rate["timestamp"] = pd.to_datetime(rate["timestamp"])
    events = event_df.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    if start is not None:
        rate = rate[rate["timestamp"] >= start]
        events = events[events["timestamp"] >= start]

    rate_by_date = rate.set_index("timestamp")["value"]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(rate["timestamp"], rate["value"], color="k", linewidth=1.4, label="Target rate", zorder=3)

    for outcome, marker, color, size, label in [
        (0, "o", "#bbbbbb", 18, "Hold / hike"),
        (1, "v", "#d62728", 55, "Cut"),
    ]:
        sub = events[events["value"] == outcome]
        # Rate level at (or just before) each meeting, for marker placement.
        levels = [
            float(rate_by_date[rate_by_date.index <= ts].iloc[-1]) if (rate_by_date.index <= ts).any() else None
            for ts in sub["timestamp"]
        ]
        ax.scatter(sub["timestamp"], levels, marker=marker, s=size, color=color, label=label, zorder=4)

    ax.set_ylabel("Target for the overnight rate (%)")
    ax.set_title("Bank of Canada target rate with fixed announcement dates by outcome")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Reliability (calibration) curve
# ---------------------------------------------------------------------------


def plot_reliability_curve(
    predictions_df: pd.DataFrame,
    *,
    n_bins: int = 5,
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
) -> tuple[Figure, Axes]:
    """Draw one reliability curve per predictor against the diagonal.

    Points on the diagonal are perfectly calibrated; above it the predictor
    under-predicts the event, below it it over-predicts. Marker size scales
    with bin population, since with ~120 meetings most bins are thin.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Tidy frame from :func:`~boc_rate_decisions.analysis.predictions_to_frame`.
    n_bins : int
        Number of probability bins (keep small: the sample is ~120 meetings).
    colors, labels : dict[str, str] or None
        Optional predictor_id -> colour / display-label maps.

    Returns
    -------
    (Figure, Axes)
    """
    predictor_ids = sorted(predictions_df["predictor_id"].unique())
    color_map = _resolve_colors(predictor_ids, colors)
    label_map = _resolve_labels(predictor_ids, labels)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="#999", linewidth=1.0, linestyle="--", zorder=1)

    for pid in predictor_ids:
        table = calibration_table(predictions_df, predictor_id=pid, n_bins=n_bins)
        if table.empty:
            continue
        ax.plot(
            table["mean_predicted"],
            table["observed_frequency"],
            color=color_map[pid],
            linewidth=1.2,
            alpha=0.8,
            zorder=2,
        )
        ax.scatter(
            table["mean_predicted"],
            table["observed_frequency"],
            s=table["n"] * 4,
            color=color_map[pid],
            label=label_map[pid],
            alpha=0.85,
            zorder=3,
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted P(cut)")
    ax.set_ylabel("Observed cut frequency")
    ax.set_title(f"Reliability curve ({n_bins} bins; marker size = bin count)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Decision timeline: predicted probabilities vs realised decisions
# ---------------------------------------------------------------------------


def plot_decision_timeline(
    predictions_df: pd.DataFrame,
    *,
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
) -> tuple[Figure, Axes]:
    """Plot predicted P(cut) per meeting over time, with realised cuts shaded.

    Vertical red bands mark meetings where the Bank actually cut. A good
    predictor's probability rises into easing cycles and stays low through
    long holds; an uninformative one is a flat line at the base rate.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Tidy frame from :func:`~boc_rate_decisions.analysis.predictions_to_frame`.
    colors, labels : dict[str, str] or None
        Optional predictor_id -> colour / display-label maps.

    Returns
    -------
    (Figure, Axes)
    """
    predictor_ids = sorted(predictions_df["predictor_id"].unique())
    color_map = _resolve_colors(predictor_ids, colors)
    label_map = _resolve_labels(predictor_ids, labels)

    fig, ax = plt.subplots(figsize=(13, 4.5))

    cut_meetings = sorted(predictions_df.loc[predictions_df["outcome"] == 1, "meeting_date"].unique())
    for md in cut_meetings:
        ts = pd.Timestamp(md)
        ax.axvspan(ts - pd.Timedelta(days=10), ts + pd.Timedelta(days=10), color="#d62728", alpha=0.15, zorder=1)

    for pid in predictor_ids:
        sub = predictions_df[predictions_df["predictor_id"] == pid].sort_values("meeting_date")
        ax.plot(
            sub["meeting_date"],
            sub["probability"],
            color=color_map[pid],
            linewidth=1.3,
            marker="o",
            markersize=3.5,
            label=label_map[pid],
            zorder=3,
        )

    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Predicted P(cut)")
    ax.set_title("Predicted cut probability by meeting (red bands = realised cuts)")
    handles, handle_labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#d62728", alpha=0.3, linewidth=8))
    handle_labels.append("Realised cut")
    ax.legend(handles, handle_labels, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


__all__ = [
    "DEFAULT_PREDICTOR_PALETTE",
    "plot_decision_timeline",
    "plot_policy_rate_with_decisions",
    "plot_reliability_curve",
]
