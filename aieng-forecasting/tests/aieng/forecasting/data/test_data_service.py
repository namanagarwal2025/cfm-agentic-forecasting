"""Tests for SeriesStore and DataService."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.service import DataService
from aieng.forecasting.data.store import SeriesStore


def _make_meta(series_id: str = "test_series") -> SeriesMetadata:
    """Return a minimal SeriesMetadata for testing."""
    return SeriesMetadata(
        series_id=series_id,
        description="Test series",
        source="test",
        units="Index",
        frequency="MS",
    )


def _make_df(
    timestamps: list[str],
    values: list[float],
) -> pd.DataFrame:
    """Return a canonical-format DataFrame."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "value": values,
        }
    )


def _make_adapter(df: pd.DataFrame) -> MagicMock:
    """Return a mock adapter whose fetch() returns the given DataFrame."""
    adapter = MagicMock()
    adapter.fetch.return_value = df
    return adapter


class TestSeriesStore:
    """Tests for SeriesStore."""

    def setup_method(self) -> None:
        """Create a fresh store for each test."""
        self.store = SeriesStore()

    def test_put_and_get_roundtrip(self) -> None:
        """put() followed by get() returns equivalent data."""
        df = _make_df(["2022-01-01", "2022-02-01"], [100.0, 101.0])
        meta = _make_meta("s1")
        self.store.put("s1", df, meta)
        result = self.store.get("s1")
        pd.testing.assert_frame_equal(result, df)

    def test_get_returns_copy(self) -> None:
        """get() returns a copy; mutations do not affect the store."""
        df = _make_df(["2022-01-01"], [100.0])
        self.store.put("s1", df, _make_meta("s1"))
        copy = self.store.get("s1")
        copy["value"] = 999.0
        assert self.store.get("s1")["value"].iloc[0] == 100.0

    def test_get_unknown_series_raises(self) -> None:
        """get() raises KeyError for unregistered series_id."""
        with pytest.raises(KeyError, match="not_a_series"):
            self.store.get("not_a_series")

    def test_series_ids_sorted(self) -> None:
        """series_ids returns sorted list."""
        for sid in ["beta", "alpha", "gamma"]:
            self.store.put(sid, _make_df(["2022-01-01"], [1.0]), _make_meta(sid))
        assert self.store.series_ids == ["alpha", "beta", "gamma"]

    def test_contains(self) -> None:
        """__contains__ returns True for registered series."""
        self.store.put("s1", _make_df(["2022-01-01"], [1.0]), _make_meta("s1"))
        assert "s1" in self.store
        assert "s2" not in self.store

    def test_put_validates_required_columns(self) -> None:
        """put() raises ValueError when timestamp or value column is absent."""
        bad_df = pd.DataFrame({"timestamp": pd.to_datetime(["2022-01-01"])})
        with pytest.raises(ValueError, match="value"):
            self.store.put("s1", bad_df, _make_meta("s1"))


class TestDataService:
    """Tests for DataService."""

    def setup_method(self) -> None:
        """Create a fresh DataService for each test."""
        self.svc = DataService()

    def test_register_and_get_series(self) -> None:
        """register() then get_series() returns cutoff-filtered data."""
        df = _make_df(["2022-01-01", "2022-02-01", "2022-03-01"], [100.0, 101.0, 102.0])
        adapter = _make_adapter(df)
        self.svc.register("s1", adapter, _make_meta("s1"))

        result = self.svc.get_series("s1", as_of=datetime(2022, 2, 1))
        assert len(result) == 2
        assert list(result["value"]) == [100.0, 101.0]

    def test_get_series_unknown_raises(self) -> None:
        """get_series() raises KeyError for unregistered series_id."""
        with pytest.raises(KeyError):
            self.svc.get_series("not_registered", as_of=datetime(2022, 1, 1))

    def test_register_calls_adapter_fetch_once(self) -> None:
        """register() calls adapter.fetch() exactly once."""
        df = _make_df(["2022-01-01"], [100.0])
        adapter = _make_adapter(df)
        self.svc.register("s1", adapter, _make_meta("s1"))
        adapter.fetch.assert_called_once()

    def test_series_ids_empty_initially(self) -> None:
        """series_ids is empty before any series are registered."""
        assert self.svc.series_ids == []

    def test_series_ids_after_registration(self) -> None:
        """series_ids returns all registered series ids sorted."""
        for sid in ["beta", "alpha"]:
            df = _make_df(["2022-01-01"], [1.0])
            self.svc.register(sid, _make_adapter(df), _make_meta(sid))
        assert self.svc.series_ids == ["alpha", "beta"]

    def test_get_metadata(self) -> None:
        """get_metadata() returns the metadata supplied at registration."""
        df = _make_df(["2022-01-01"], [1.0])
        meta = _make_meta("s1")
        self.svc.register("s1", _make_adapter(df), meta)
        result = self.svc.get_metadata("s1")
        assert result.series_id == "s1"
        assert result.source == "test"

    def test_summary_structure(self) -> None:
        """summary() returns a DataFrame with expected columns."""
        df = _make_df(["2022-01-01", "2022-02-01"], [100.0, 101.0])
        self.svc.register("s1", _make_adapter(df), _make_meta("s1"))
        summary = self.svc.summary()
        expected_cols = {"series_id", "description", "source", "units", "frequency", "n_obs", "start", "end"}
        assert expected_cols.issubset(set(summary.columns))
        assert summary.loc[0, "n_obs"] == 2
