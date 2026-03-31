"""Tests for data layer Pydantic models."""

from datetime import datetime, timezone

import pytest

from aieng.forecasting.data.models import SeriesMetadata, SeriesRecord


class TestSeriesRecord:
    """Tests for SeriesRecord."""

    def test_basic_construction(self) -> None:
        """SeriesRecord can be constructed with timestamp and value."""
        rec = SeriesRecord(timestamp=datetime(2023, 1, 1), value=150.5)
        assert rec.timestamp == datetime(2023, 1, 1)
        assert rec.value == 150.5
        assert rec.released_at is None

    def test_with_released_at(self) -> None:
        """SeriesRecord accepts a released_at later than timestamp."""
        rec = SeriesRecord(
            timestamp=datetime(2023, 1, 1),
            value=150.5,
            released_at=datetime(2023, 1, 20),
        )
        assert rec.released_at == datetime(2023, 1, 20)

    def test_released_at_before_timestamp_raises(self) -> None:
        """released_at before timestamp should raise ValueError."""
        with pytest.raises(ValueError, match="released_at"):
            SeriesRecord(
                timestamp=datetime(2023, 2, 1),
                value=150.5,
                released_at=datetime(2023, 1, 1),
            )

    def test_released_at_equal_to_timestamp_is_valid(self) -> None:
        """released_at equal to timestamp is allowed."""
        ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
        rec = SeriesRecord(timestamp=ts, value=1.0, released_at=ts)
        assert rec.released_at == rec.timestamp


class TestSeriesMetadata:
    """Tests for SeriesMetadata."""

    def test_basic_construction(self) -> None:
        """SeriesMetadata can be constructed with required fields."""
        meta = SeriesMetadata(
            series_id="cpi_all_items_canada",
            description="CPI All-items, Canada",
            source="StatCan",
            units="Index 2002=100",
            frequency="MS",
        )
        assert meta.series_id == "cpi_all_items_canada"
        assert meta.table_id is None

    def test_with_table_id(self) -> None:
        """SeriesMetadata accepts an optional table_id."""
        meta = SeriesMetadata(
            series_id="test",
            description="test",
            source="StatCan",
            units="Index",
            frequency="MS",
            table_id="18-10-0004-13",
        )
        assert meta.table_id == "18-10-0004-13"
