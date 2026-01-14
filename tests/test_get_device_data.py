import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from hive import get_device_data, CACHED_DEVICE_DATA_FILE, Device
from tests.conftest import HEATER_ID, TRV_ID

UTC_TZ = ZoneInfo("UTC")


class TestGetDeviceDataCached:
    """Tests for get_device_data when using cached data (no API calls)."""

    @pytest.mark.skipif(
        not CACHED_DEVICE_DATA_FILE.exists(),
        reason="No cached data file exists - run with refresh=True first"
    )
    @pytest.mark.asyncio
    async def test_returns_cached_dataframe(self):
        """Test that get_device_data returns cached DataFrame when refresh=False."""
        result = await get_device_data(refresh=False)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    @pytest.mark.skipif(
        not CACHED_DEVICE_DATA_FILE.exists(),
        reason="No cached data file exists - run with refresh=True first"
    )
    @pytest.mark.asyncio
    async def test_cached_data_has_expected_columns(self):
        """Test that cached DataFrame has expected columns."""
        result = await get_device_data(refresh=False)

        expected_columns = [
            "date", "device_id", "device_name",
            "temperature", "heat_target", "heating_relay",
            "heating_minutes", "is_heater"
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    @pytest.mark.skipif(
        not CACHED_DEVICE_DATA_FILE.exists(),
        reason="No cached data file exists - run with refresh=True first"
    )
    @pytest.mark.asyncio
    async def test_cached_data_has_utc_dates(self):
        """Test that date column is UTC timezone-aware."""
        result = await get_device_data(refresh=False)

        assert result.date.dt.tz is not None
        assert str(result.date.dt.tz) == "UTC"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_cache_and_no_refresh(self, tmp_path, monkeypatch):
        """Test that get_device_data returns None when no cache exists and refresh=False."""
        # Point to non-existent cache file
        fake_cache = tmp_path / "nonexistent.pickle"
        monkeypatch.setattr("hive.CACHED_DEVICE_DATA_FILE", fake_cache)

        result = await get_device_data(refresh=False)

        assert result is None


class TestGetDeviceDataMocked:
    """Tests for get_device_data with mocked I/O operations."""

    @pytest.fixture
    def mock_devices(self):
        """Minimal device fixtures."""
        return {
            HEATER_ID: Device(id=HEATER_ID, name="Zone 1", type="boilermodule"),
            TRV_ID: Device(id=TRV_ID, name="Diana's room", type="trv"),
        }

    @pytest.fixture
    def mock_raw_api_data(self):
        """Simulates API response for a short time period."""
        base_ts = int(datetime(2024, 1, 7, 10, 0, 0, tzinfo=UTC_TZ).timestamp())
        return {
            HEATER_ID: {
                "heating_relay": {
                    str(base_ts + 0): 0,
                    str(base_ts + 60): 1,
                    str(base_ts + 120): 1,
                    str(base_ts + 180): 0,
                },
                "temperature": {
                    str(base_ts + 0): 18.5,
                    str(base_ts + 60): 18.6,
                    str(base_ts + 120): 18.7,
                    str(base_ts + 180): 18.8,
                },
                "heat_target": {
                    str(base_ts + 0): 21,
                    str(base_ts + 60): 21,
                    str(base_ts + 120): 21,
                    str(base_ts + 180): 21,
                },
            },
            TRV_ID: {
                "heating_demand": {
                    str(base_ts + 0): 0,
                    str(base_ts + 60): 1,
                    str(base_ts + 120): 1,
                    str(base_ts + 180): 0,
                },
                "heating_demand_percentage": {
                    str(base_ts + 0): 0,
                    str(base_ts + 60): 50,
                    str(base_ts + 120): 60,
                    str(base_ts + 180): 0,
                },
                "temperature": {
                    str(base_ts + 0): 17.5,
                    str(base_ts + 60): 17.6,
                    str(base_ts + 120): 17.7,
                    str(base_ts + 180): 17.8,
                },
                "heat_target": {
                    str(base_ts + 0): 20,
                    str(base_ts + 60): 20,
                    str(base_ts + 120): 20,
                    str(base_ts + 180): 20,
                },
            },
        }

    @pytest.mark.asyncio
    async def test_refresh_fetches_from_api(self, mock_devices, mock_raw_api_data):
        """Test that get_device_data fetches from API when refresh=True."""
        with patch("hive._load_cached_dataframe") as mock_load, \
             patch("hive._save_cached_dataframe") as mock_save, \
             patch("hive._save_raw_data") as mock_save_raw, \
             patch("hive.parse_raw_device_data") as mock_parse_raw, \
             patch("hive.get_devices", new_callable=AsyncMock) as mock_get_devices, \
             patch("hive._fetch_all_device_data", new_callable=AsyncMock) as mock_fetch:

            # Setup mocks
            mock_load.return_value = pd.DataFrame()  # No cached data
            mock_parse_raw.return_value = pd.DataFrame()  # No raw files either
            mock_get_devices.return_value = mock_devices
            mock_fetch.return_value = mock_raw_api_data

            result = await get_device_data(refresh=True)

            # Verify API was called
            mock_get_devices.assert_called_once()
            mock_fetch.assert_called_once()

            # Verify result
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert not result.empty

            # Verify data was saved
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_creates_dataframe_with_correct_structure(self, mock_devices, mock_raw_api_data):
        """Test that refreshed data has correct DataFrame structure."""
        with patch("hive._load_cached_dataframe") as mock_load, \
             patch("hive._save_cached_dataframe") as mock_save, \
             patch("hive._save_raw_data") as mock_save_raw, \
             patch("hive.parse_raw_device_data") as mock_parse_raw, \
             patch("hive.get_devices", new_callable=AsyncMock) as mock_get_devices, \
             patch("hive._fetch_all_device_data", new_callable=AsyncMock) as mock_fetch:

            mock_load.return_value = pd.DataFrame()
            mock_parse_raw.return_value = pd.DataFrame()  # No raw files
            mock_get_devices.return_value = mock_devices
            mock_fetch.return_value = mock_raw_api_data

            result = await get_device_data(refresh=True)

            # Check expected columns
            expected_columns = ["date", "device_id", "device_name", "temperature", "heating_relay"]
            for col in expected_columns:
                assert col in result.columns, f"Missing column: {col}"

            # Check device IDs are correct
            assert set(result.device_id.unique()) == {HEATER_ID, TRV_ID}

    @pytest.mark.asyncio
    async def test_no_refresh_returns_cached_without_api_call(self, mock_devices):
        """Test that refresh=False returns cached data without API calls."""
        # Create a minimal cached DataFrame
        cached_df = pd.DataFrame({
            "date": pd.to_datetime([datetime(2024, 1, 7, 10, 0, 0, tzinfo=UTC_TZ)]),
            "device_id": [HEATER_ID],
            "device_name": ["Zone 1"],
            "temperature": [20.0],
        })

        with patch("hive._load_cached_dataframe") as mock_load, \
             patch("hive.get_devices", new_callable=AsyncMock) as mock_get_devices:

            mock_load.return_value = cached_df

            result = await get_device_data(refresh=False)

            # Verify cached data returned
            assert result is not None
            assert len(result) == 1

            # Verify API was NOT called
            mock_get_devices.assert_not_called()
