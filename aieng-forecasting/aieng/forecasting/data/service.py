"""DataService: the primary interface for predictors and notebooks."""

from datetime import datetime

import pandas as pd

from aieng.forecasting.data.adapters.base import BaseAdapter
from aieng.forecasting.data.cutoff import CutoffEnforcer
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.store import SeriesStore


class DataService:
    """Unified interface for registering and querying time series data.

    ``DataService`` is the single point of contact between the data layer and
    anything that consumes data (predictors, notebooks, evaluation harnesses).
    It owns the ``SeriesStore`` and ``CutoffEnforcer``, and exposes two
    operations:

    1. ``register`` — fetch data via an adapter and store it in memory.
    2. ``get_series`` — retrieve a series as of a given date, with cutoff
       filtering applied automatically.

    **Intended usage pattern**

    A data-loading script (e.g. ``scripts/fetch_cpi.py``) calls ``register``
    once at startup, downloading and caching data. After that, all calls go
    through ``get_series`` — no further network access occurs.

    **Predictor usage**

    Predictors decide which series to request; the ``DataService`` does not
    impose which covariates are used. A predictor may call ``get_series`` for
    any registered ``series_id`` and receive the observations available up to
    its forecast ``as_of`` date.

    Examples
    --------
    >>> from aieng.forecasting.data import DataService, SeriesMetadata
    >>> from aieng.forecasting.data.adapters import StatCanAdapter
    >>> svc = DataService()
    >>> adapter = StatCanAdapter(
    ...     table_id="18-10-0004-13",
    ...     member_filter={"GEO": "Canada", "Products and product groups": "All-items"},
    ... )
    >>> meta = SeriesMetadata(
    ...     series_id="cpi_all_items_canada",
    ...     description="CPI All-items, Canada (2002=100)",
    ...     source="StatCan",
    ...     units="Index 2002=100",
    ...     frequency="MS",
    ...     table_id="18-10-0004-13",
    ... )
    >>> svc.register("cpi_all_items_canada", adapter, meta)
    >>> df = svc.get_series("cpi_all_items_canada", as_of=datetime(2023, 1, 1))
    """

    def __init__(self) -> None:
        self._store = SeriesStore()
        self._cutoff = CutoffEnforcer()

    def register(
        self,
        series_id: str,
        adapter: BaseAdapter,
        metadata: SeriesMetadata,
    ) -> None:
        """Fetch data via an adapter and register the series in the store.

        Parameters
        ----------
        series_id : str
            Unique identifier for the series. Used as the lookup key in
            subsequent ``get_series`` calls.
        adapter : BaseAdapter
            Adapter responsible for fetching the data. ``adapter.fetch()`` is
            called exactly once; the result is stored in memory.
        metadata : SeriesMetadata
            Descriptive metadata (units, source, frequency, etc.).

        Raises
        ------
        RuntimeError
            If the adapter fails to fetch data.
        ValueError
            If the fetched DataFrame is missing required columns.
        """
        df = adapter.fetch()
        self._store.put(series_id, df, metadata)

    def get_series(self, series_id: str, as_of: datetime) -> pd.DataFrame:
        """Return a series filtered to observations available as of ``as_of``.

        The ``CutoffEnforcer`` ensures that only data published on or before
        ``as_of`` is returned. This guarantees that backtests and live
        forecasts share the same information discipline.

        Parameters
        ----------
        series_id : str
            The series to retrieve.
        as_of : datetime
            Information cutoff point. Observations released after this date
            are excluded.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``timestamp`` and ``value`` (and optionally
            ``released_at``), containing only rows available as of ``as_of``,
            sorted ascending by ``timestamp``.

        Raises
        ------
        KeyError
            If ``series_id`` is not registered.
        """
        raw = self._store.get(series_id)
        return self._cutoff.filter(raw, as_of)

    def get_metadata(self, series_id: str) -> SeriesMetadata:
        """Return metadata for a registered series.

        Parameters
        ----------
        series_id : str
            The series identifier.

        Returns
        -------
        SeriesMetadata
            Metadata for the series.

        Raises
        ------
        KeyError
            If ``series_id`` is not registered.
        """
        return self._store.get_metadata(series_id)

    @property
    def series_ids(self) -> list[str]:
        """Return a sorted list of registered series identifiers."""
        return self._store.series_ids

    def summary(self) -> pd.DataFrame:
        """Return a summary table of all registered series.

        Returns
        -------
        pd.DataFrame
            One row per series with columns: ``series_id``, ``description``,
            ``source``, ``units``, ``frequency``, ``n_obs``, ``start``, ``end``.
        """
        rows = []
        for sid in self._store.series_ids:
            df = self._store.get(sid)
            meta = self._store.get_metadata(sid)
            rows.append(
                {
                    "series_id": sid,
                    "description": meta.description,
                    "source": meta.source,
                    "units": meta.units,
                    "frequency": meta.frequency,
                    "n_obs": len(df),
                    "start": df["timestamp"].min() if len(df) > 0 else None,
                    "end": df["timestamp"].max() if len(df) > 0 else None,
                }
            )
        return pd.DataFrame(rows)
