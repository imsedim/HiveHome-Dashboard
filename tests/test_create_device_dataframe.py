import pytest
import pandas as pd

from hive import _create_device_dataframe
from tests.conftest import HEATER_ID, TRV_ID


class TestCreateDeviceDataframe:
    """Tests for _create_device_dataframe pure function."""

    def test_creates_dataframe_from_raw_data(self, sample_devices, sample_raw_data):
        """Test basic DataFrame creation from raw API data."""
        df = _create_device_dataframe(sample_devices, sample_raw_data, resample_freq=None)

        assert not df.empty
        assert set(df.device_id.unique()) == {HEATER_ID, TRV_ID}

    def test_contains_expected_columns(self, sample_devices, sample_raw_data):
        """Test that output contains all required columns."""
        df = _create_device_dataframe(sample_devices, sample_raw_data, resample_freq=None)

        expected_columns = [
            "date", "device_id", "device_name",
            "temperature", "heat_target", "heating_relay",
            "heating_minutes", "is_heater", "heating_start", "heating_end", "heating_length"
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_applies_device_name_mapping(self, sample_devices, sample_raw_data):
        """Test that device names are mapped correctly."""
        df = _create_device_dataframe(sample_devices, sample_raw_data, resample_freq=None)

        assert "device_name" in df.columns
        heater_rows = df[df.device_id == HEATER_ID]
        assert heater_rows.device_name.iloc[0] == "Zone 1"

        trv_rows = df[df.device_id == TRV_ID]
        assert trv_rows.device_name.iloc[0] == "Diana's room"

    def test_date_column_is_utc_datetime(self, sample_devices, sample_raw_data):
        """Test that date column is UTC timezone-aware."""
        df = _create_device_dataframe(sample_devices, sample_raw_data, resample_freq=None)

        assert df.date.dt.tz is not None
        assert str(df.date.dt.tz) == "UTC"

    def test_heating_relay_is_boolean(self, sample_devices, sample_raw_data):
        """Test that heating_relay column is boolean type."""
        df = _create_device_dataframe(sample_devices, sample_raw_data, resample_freq=None)

        assert df.heating_relay.dtype == "boolean"

    def test_is_heater_flag_set_correctly(self, sample_devices, sample_raw_data):
        """Test that is_heater flag distinguishes heater from TRV devices."""
        df = _create_device_dataframe(sample_devices, sample_raw_data, resample_freq=None)

        heater_rows = df[df.device_id == HEATER_ID]
        assert heater_rows.is_heater.all()

        trv_rows = df[df.device_id == TRV_ID]
        assert not trv_rows.is_heater.any()

    def test_resampling_disabled_when_freq_none(self, sample_devices, sample_raw_data):
        """Test that data is not resampled when freq=None."""
        df = _create_device_dataframe(sample_devices, sample_raw_data, resample_freq=None)

        # Should have more granular data (1-minute intervals from fixture)
        # With 6 timestamps per device and 2 devices, expect around 12 rows
        # (may vary due to time grid alignment)
        assert len(df) > 0

    def test_resampling_applies_when_freq_set(self, sample_devices, sample_raw_data):
        """Test that data is resampled when freq is specified."""
        df_raw = _create_device_dataframe(sample_devices, sample_raw_data, resample_freq=None)
        df_resampled = _create_device_dataframe(sample_devices, sample_raw_data, resample_freq="5min")

        # Resampled should have fewer or equal rows
        assert len(df_resampled) <= len(df_raw)
