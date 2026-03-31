"""Tests for StatCanAdapter (no live network calls)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aieng.forecasting.data.adapters.statcan import StatCanAdapter


def _make_raw_statcan_table() -> pd.DataFrame:
    """Return a minimal mock of a stats-can table DataFrame.

    Mimics the structure returned by ``StatsCan.table_to_df()`` for a
    CPI-like table with two geographies and two product groups.
    """
    return pd.DataFrame(
        {
            "REF_DATE": [
                "2022-01",
                "2022-02",
                "2022-01",
                "2022-02",
                "2022-01",
                "2022-02",
            ],
            "GEO": [
                "Canada",
                "Canada",
                "Canada",
                "Canada",
                "Ontario",
                "Ontario",
            ],
            "Products and product groups": [
                "All-items",
                "All-items",
                "Food",
                "Food",
                "All-items",
                "All-items",
            ],
            "VALUE": [151.2, 152.4, 165.3, 166.1, 148.0, 149.5],
            "STATUS": ["", "", "", "", "", ""],
        }
    )


@pytest.fixture()
def adapter() -> StatCanAdapter:
    """Return a StatCanAdapter configured for All-items Canada."""
    return StatCanAdapter(
        table_id="18-10-0004-13",
        member_filter={"GEO": "Canada", "Products and product groups": "All-items"},
        cache_dir="data/statcan_test",
    )


class TestStatCanAdapterProperties:
    """Tests for StatCanAdapter property accessors."""

    def test_table_id(self, adapter: StatCanAdapter) -> None:
        """table_id property returns the configured table identifier."""
        assert adapter.table_id == "18-10-0004-13"

    def test_member_filter_is_copy(self, adapter: StatCanAdapter) -> None:
        """member_filter returns a copy, not the internal dict."""
        f1 = adapter.member_filter
        f1["GEO"] = "Ontario"
        assert adapter.member_filter["GEO"] == "Canada"


class TestStatCanAdapterFetch:
    """Tests for StatCanAdapter.fetch() with mocked stats-can.

    Notes
    -----
    ``stats_can`` is imported lazily inside ``fetch()`` to keep it optional.
    All tests mock it via ``patch.dict("sys.modules", ...)`` so that the
    ``import stats_can`` statement inside ``fetch()`` resolves to the mock.
    """

    def _mock_stats_can(self, raw_df: pd.DataFrame) -> MagicMock:
        """Return a mock stats_can module with a pre-configured StatsCan class."""
        mock_sc_instance = MagicMock()
        mock_sc_instance.table_to_df.return_value = raw_df

        mock_module = MagicMock()
        mock_module.StatsCan.return_value = mock_sc_instance
        return mock_module

    def test_fetch_returns_correct_shape(self, adapter: StatCanAdapter) -> None:
        """fetch() returns a DataFrame with timestamp and value columns."""
        raw = _make_raw_statcan_table()
        mock_module = self._mock_stats_can(raw)

        with patch.dict("sys.modules", {"stats_can": mock_module}):
            result = adapter.fetch()

        assert set(result.columns) == {"timestamp", "value"}
        assert len(result) == 2  # 2 Canada All-items rows

    def test_fetch_filters_correctly(self, adapter: StatCanAdapter) -> None:
        """fetch() returns only rows matching member_filter."""
        raw = _make_raw_statcan_table()
        mock_module = self._mock_stats_can(raw)

        with patch.dict("sys.modules", {"stats_can": mock_module}):
            result = adapter.fetch()

        # All rows should be Canada / All-items only.
        assert (result["value"] == pd.Series([151.2, 152.4])).all()

    def test_fetch_sorted_by_timestamp(self, adapter: StatCanAdapter) -> None:
        """fetch() returns rows sorted ascending by timestamp."""
        raw = _make_raw_statcan_table().iloc[::-1].reset_index(drop=True)
        mock_module = self._mock_stats_can(raw)

        with patch.dict("sys.modules", {"stats_can": mock_module}):
            result = adapter.fetch()

        assert result["timestamp"].is_monotonic_increasing

    def test_fetch_drops_nan_values(self, adapter: StatCanAdapter) -> None:
        """fetch() drops rows where VALUE is NaN."""
        raw = _make_raw_statcan_table()
        raw.loc[raw["REF_DATE"] == "2022-02", "VALUE"] = float("nan")
        mock_module = self._mock_stats_can(raw)

        with patch.dict("sys.modules", {"stats_can": mock_module}):
            result = adapter.fetch()

        # Only 2022-01 Canada All-items should survive.
        assert len(result) == 1
        assert result["value"].iloc[0] == 151.2

    def test_fetch_raises_on_missing_filter_column(self) -> None:
        """fetch() raises ValueError when a filter column is absent from the table."""
        raw = pd.DataFrame({"REF_DATE": ["2022-01"], "VALUE": [100.0]})  # no GEO column
        mock_module = self._mock_stats_can(raw)

        bad_adapter = StatCanAdapter(
            table_id="18-10-0004-13",
            member_filter={"GEO": "Canada"},
        )
        with patch.dict("sys.modules", {"stats_can": mock_module}):
            with pytest.raises(ValueError, match="GEO"):
                bad_adapter.fetch()

    def test_fetch_raises_when_no_rows_match(self) -> None:
        """fetch() raises RuntimeError when filter matches zero rows."""
        raw = _make_raw_statcan_table()
        mock_module = self._mock_stats_can(raw)

        bad_adapter = StatCanAdapter(
            table_id="18-10-0004-13",
            member_filter={"GEO": "Narnia"},
        )
        with patch.dict("sys.modules", {"stats_can": mock_module}):
            with pytest.raises(RuntimeError, match="No rows matched"):
                bad_adapter.fetch()

    def test_fetch_raises_on_stats_can_error(self, adapter: StatCanAdapter) -> None:
        """fetch() wraps stats-can errors in RuntimeError."""
        mock_module = MagicMock()
        mock_module.StatsCan.side_effect = ConnectionError("network down")

        with patch.dict("sys.modules", {"stats_can": mock_module}):
            with pytest.raises(RuntimeError, match="Failed to fetch"):
                adapter.fetch()
