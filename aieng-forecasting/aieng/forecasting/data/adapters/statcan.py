"""Statistics Canada adapter using the stats-can library."""

from pathlib import Path

import pandas as pd

from aieng.forecasting.data.adapters.base import BaseAdapter


# Canonical column names in StatCan CSV exports (stable across tables).
_STATCAN_DATE_COL = "REF_DATE"
_STATCAN_VALUE_COL = "VALUE"


class StatCanAdapter(BaseAdapter):
    """Adapter for a single series from a Statistics Canada table.

    Uses the ``stats-can`` library to download and cache tables locally.
    After the initial download, all data is served from the local cache —
    no further network calls are made unless the cache is cleared.

    Each instance represents **one series**, identified by a set of filter
    criteria (e.g. geography + product group). For tables that contain many
    series, instantiate one ``StatCanAdapter`` per series and register each
    with ``DataService`` under a distinct ``series_id``.

    Parameters
    ----------
    table_id : str
        Statistics Canada table identifier (e.g. ``"18-10-0004-13"``).
    member_filter : dict[str, str]
        Column-value pairs used to select a single series from the table.
        For example: ``{"GEO": "Canada", "Products and product groups": "All-items"}``.
        All specified columns must be present in the downloaded table.
    cache_dir : str or Path
        Directory where the ``stats-can`` library stores its local table cache.
        Defaults to ``"data/statcan"`` relative to the current working directory.

    Notes
    -----
    **Information cutoff**: StatCan publishes CPI data roughly 3 weeks after
    the reference month. For example, January CPI is released in mid-February.
    This adapter currently sets ``released_at = None``, which causes
    ``CutoffEnforcer`` to fall back to ``timestamp`` (the reference month).
    This is a slight optimistic bias in backtests. A future improvement would
    populate ``released_at`` from StatCan's release schedule API.

    Examples
    --------
    >>> adapter = StatCanAdapter(
    ...     table_id="18-10-0004-13",
    ...     member_filter={
    ...         "GEO": "Canada",
    ...         "Products and product groups": "All-items",
    ...     },
    ... )
    >>> df = adapter.fetch()
    >>> df.columns.tolist()
    ['timestamp', 'value']
    """

    def __init__(
        self,
        table_id: str,
        member_filter: dict[str, str],
        cache_dir: str | Path = "data/statcan",
    ) -> None:
        self._table_id = table_id
        self._member_filter = member_filter
        self._cache_dir = Path(cache_dir)

    @property
    def table_id(self) -> str:
        """Return the StatCan table identifier."""
        return self._table_id

    @property
    def member_filter(self) -> dict[str, str]:
        """Return the filter criteria that identify this series."""
        return dict(self._member_filter)

    def fetch(self) -> pd.DataFrame:
        """Download (or load from cache) and return the series in canonical format.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``timestamp`` (datetime64[ns]) and ``value``
            (float64), sorted ascending by ``timestamp``. Rows with missing
            values are dropped.

        Raises
        ------
        RuntimeError
            If the table cannot be downloaded or the filter criteria do not
            match any rows.
        ValueError
            If a column named in ``member_filter`` is not present in the table.
        """
        import stats_can  # local import — optional dependency

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            sc = stats_can.StatsCan(data_folder=str(self._cache_dir))
            raw: pd.DataFrame = sc.table_to_df(self._table_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch StatCan table {self._table_id!r}: {exc}"
            ) from exc

        # Validate that all filter columns exist before filtering.
        missing_cols = [col for col in self._member_filter if col not in raw.columns]
        if missing_cols:
            raise ValueError(
                f"Filter column(s) {missing_cols} not found in table {self._table_id!r}. "
                f"Available columns: {raw.columns.tolist()}"
            )

        # Apply member filter to isolate the target series.
        mask = pd.Series(True, index=raw.index)
        for col, val in self._member_filter.items():
            mask &= raw[col] == val

        filtered = raw.loc[mask].copy()

        if filtered.empty:
            raise RuntimeError(
                f"No rows matched filter {self._member_filter} in table {self._table_id!r}."
            )

        if _STATCAN_VALUE_COL not in filtered.columns:
            raise ValueError(
                f"Expected value column {_STATCAN_VALUE_COL!r} not found in table. "
                f"Available columns: {filtered.columns.tolist()}"
            )

        if _STATCAN_DATE_COL not in filtered.columns:
            raise ValueError(
                f"Expected date column {_STATCAN_DATE_COL!r} not found in table. "
                f"Available columns: {filtered.columns.tolist()}"
            )

        # Build canonical output: (timestamp, value).
        result = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(filtered[_STATCAN_DATE_COL]),
                "value": pd.to_numeric(filtered[_STATCAN_VALUE_COL], errors="coerce"),
            }
        )

        # Drop rows with missing values (StatCan uses blank VALUE for suppressed data).
        result = result.dropna(subset=["value"])
        result = result.sort_values("timestamp").reset_index(drop=True)

        return result
