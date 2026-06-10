"""Analysis helpers for the BoC rate-decision experiment.

Pure functions that turn :class:`BacktestResult` / :class:`EvalResult`
objects into tidy DataFrames for the binary-event evaluation: per-meeting
prediction tables, the Brier leaderboard (with skill scores against the
base-rate predictor), and reliability/calibration bins.

Kept separate from the notebooks so they can be unit-tested and reused.
All functions are pure: they take results plus an observed event series and
return DataFrames. They never fetch data or mutate global state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.eval import EvalResult
from aieng.forecasting.evaluation.prediction import BinaryForecast


def predictions_to_frame(
    results: dict[str, BacktestResult | EvalResult],
    event_df: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten binary predictions + Brier scores into a tidy per-meeting DataFrame.

    Parameters
    ----------
    results : dict[str, BacktestResult | EvalResult]
        Mapping ``predictor_id -> result``. Scores must be Brier scores
        (``result.metric == "brier"``).
    event_df : pd.DataFrame
        Observed 0/1 event series (``timestamp`` / ``value`` columns, as
        returned by :meth:`DataService.get_series`), used to attach the
        realised outcome at each prediction's ``forecast_date``.

    Returns
    -------
    pd.DataFrame
        Columns: ``predictor_id``, ``origin``, ``meeting_date``,
        ``probability``, ``outcome``, ``brier``.
    """
    outcome_by_date = {
        pd.Timestamp(ts).normalize(): int(v) for ts, v in zip(event_df["timestamp"], event_df["value"])
    }

    rows: list[dict[str, object]] = []
    for predictor_id, result in results.items():
        for pred, score in zip(result.predictions, result.scores):
            if not isinstance(pred.payload, BinaryForecast):
                continue
            meeting_date = pd.Timestamp(pred.forecast_date).normalize()
            rows.append(
                {
                    "predictor_id": predictor_id,
                    "origin": pd.Timestamp(pred.as_of),
                    "meeting_date": meeting_date,
                    "probability": float(pred.payload.probability),
                    "outcome": outcome_by_date.get(meeting_date),
                    "brier": float(score),
                }
            )
    return pd.DataFrame(rows)


def brier_leaderboard(
    results: dict[str, BacktestResult | EvalResult],
    *,
    reference_id: str | None = None,
) -> pd.DataFrame:
    """Build a mean-Brier leaderboard, optionally with skill scores.

    The Brier skill score against a reference predictor is
    ``1 - brier / brier_ref``: positive means the predictor beats the
    reference, 0 means it matches it, negative means it loses. The natural
    reference for this experiment is ``HistoricalFrequencyPredictor`` — a
    model that cannot beat the base rate has learned nothing.

    Parameters
    ----------
    results : dict[str, BacktestResult | EvalResult]
        Mapping ``predictor_id -> result`` with Brier scores.
    reference_id : str or None
        Predictor id to use as the skill-score reference. When ``None`` (or
        not present in ``results``) the ``skill_vs_base_rate`` column is
        omitted.

    Returns
    -------
    pd.DataFrame
        One row per predictor, sorted by ``mean_brier`` ascending. Columns:
        ``predictor_id``, ``mean_brier``, ``n_predictions``,
        ``n_skipped_origins`` and optionally ``skill_vs_base_rate``.
    """
    rows: list[dict[str, object]] = []
    for predictor_id, result in results.items():
        rows.append(
            {
                "predictor_id": predictor_id,
                "mean_brier": result.mean_score,
                "n_predictions": len(result.predictions),
                "n_skipped_origins": result.skipped_origins,
            }
        )
    board = pd.DataFrame(rows).sort_values("mean_brier").reset_index(drop=True)

    if reference_id is not None and reference_id in results:
        ref_brier = results[reference_id].mean_score
        if ref_brier > 0:
            board["skill_vs_base_rate"] = (1.0 - board["mean_brier"] / ref_brier).round(4)
    return board


def calibration_table(
    predictions_df: pd.DataFrame,
    *,
    predictor_id: str | None = None,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Bin predicted probabilities and compare against observed event frequency.

    This is the tabular form of the reliability curve: a perfectly calibrated
    predictor has ``observed_frequency ~= mean_predicted`` in every bin.
    With only ~120 meetings (and ~12% cuts), bins are necessarily coarse —
    five equal-width bins is about as fine as the sample supports.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Tidy frame from :func:`predictions_to_frame`. Rows with missing
        ``outcome`` (unresolved meetings) are dropped.
    predictor_id : str or None
        Restrict to one predictor; ``None`` uses all rows (caller's
        responsibility to pass a single-predictor frame in that case).
    n_bins : int
        Number of equal-width probability bins over [0, 1].

    Returns
    -------
    pd.DataFrame
        One row per non-empty bin: ``bin_left``, ``bin_right``,
        ``mean_predicted``, ``observed_frequency``, ``n``.
    """
    df = predictions_df.dropna(subset=["outcome"])
    if predictor_id is not None:
        df = df[df["predictor_id"] == predictor_id]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        # Right-inclusive last bin so p=1.0 is counted.
        upper_ok = df["probability"] <= right if right >= 1.0 else df["probability"] < right
        in_bin = df[(df["probability"] >= left) & upper_ok]
        if in_bin.empty:
            continue
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "mean_predicted": float(in_bin["probability"].mean()),
                "observed_frequency": float(in_bin["outcome"].mean()),
                "n": int(len(in_bin)),
            }
        )
    return pd.DataFrame(rows)


def yearly_outcome_table(event_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise meetings and cuts per calendar year (class-imbalance view).

    Parameters
    ----------
    event_df : pd.DataFrame
        Observed 0/1 event series (``timestamp`` / ``value`` columns).

    Returns
    -------
    pd.DataFrame
        Indexed by year with ``n_meetings``, ``n_cuts``, ``cut_rate`` columns.
    """
    df = event_df.copy()
    df["year"] = pd.to_datetime(df["timestamp"]).dt.year
    grouped = df.groupby("year")["value"].agg(n_meetings="count", n_cuts="sum")
    grouped["n_cuts"] = grouped["n_cuts"].astype(int)
    grouped["cut_rate"] = (grouped["n_cuts"] / grouped["n_meetings"]).round(3)
    return grouped


def rationales_table(result: BacktestResult | EvalResult) -> pd.DataFrame:
    """Extract per-prediction metadata (reasoning traces etc.) into a DataFrame.

    For the agent predictor, ``metadata`` carries ``reasoning`` and
    ``key_signals`` — the inputs for the planned reasoning-alignment
    evaluation against the Bank's own published rationale.

    Parameters
    ----------
    result : BacktestResult | EvalResult
        Result to introspect.

    Returns
    -------
    pd.DataFrame
        Columns: ``origin``, ``meeting_date``, ``probability``, plus one
        ``meta_*`` column per distinct metadata key (missing values filled
        with ``None``).
    """
    base_rows: list[dict[str, object]] = []
    all_keys: set[str] = set()
    for pred in result.predictions:
        row: dict[str, object] = {
            "origin": pd.Timestamp(pred.as_of),
            "meeting_date": pd.Timestamp(pred.forecast_date),
            "probability": float(pred.payload.probability) if isinstance(pred.payload, BinaryForecast) else None,
        }
        for k, v in pred.metadata.items():
            row[f"meta_{k}"] = v
            all_keys.add(f"meta_{k}")
        base_rows.append(row)

    for row in base_rows:
        for k in all_keys:
            row.setdefault(k, None)
    return pd.DataFrame(base_rows)


__all__ = [
    "brier_leaderboard",
    "calibration_table",
    "predictions_to_frame",
    "rationales_table",
    "yearly_outcome_table",
]
