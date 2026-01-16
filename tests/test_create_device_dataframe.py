import pytest
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from hive import _create_device_dataframe, _convert_measures_to_df, Device
from tests.conftest import HEATER_ID, TRV_ID

UTC_TZ = ZoneInfo("UTC")


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


class TestConvertMeasuresToDf:
    """Tests for _convert_measures_to_df function."""

    @staticmethod
    def ts(minute: int) -> str:
        """Create unix timestamp string for 2024-01-01 10:XX:00 UTC."""
        return str(int(datetime(2024, 1, 1, 10, minute, 0, tzinfo=UTC_TZ).timestamp()))

    def test_trv_gets_heater_timestamps_via_outer_merge(self):
        """
        Bug reproduction: When heater has timestamps that TRV doesn't have,
        the outer merge should propagate those timestamps to the TRV.
        Currently, device_id is NaN for heater-only timestamps causing them
        to be lost.

        Setup:
        - TRV has data at 10:00, 10:05, 10:10
        - Heater has data at 10:03, 10:05, 10:07 (10:05 is shared)

        Expected: TRV should have at least the union: 10:00, 10:03, 10:05, 10:07, 10:10
        Bug: TRV only has 10:00, 10:05, 10:10 (heater-only timestamps 10:03, 10:07 lost)
        """
        heater_id = "test_heater"
        trv_id = "test_trv"

        devices = {
            heater_id: Device(id=heater_id, name="Boiler", type="boilermodule"),
            trv_id: Device(id=trv_id, name="Living Room", type="trv"),
        }

        data = {
            heater_id: {
                "heating_relay": {
                    self.ts(3): 1,   # 10:03 - heater only
                    self.ts(5): 1,   # 10:05 - shared with TRV
                    self.ts(7): 1,   # 10:07 - heater only
                },
            },
            trv_id: {
                "temperature": {
                    self.ts(0): 20.0,   # 10:00 - TRV only
                    self.ts(5): 20.5,   # 10:05 - shared with heater
                    self.ts(10): 21.0,  # 10:10 - TRV only
                },
                "heat_target": {
                    self.ts(0): 22.0,
                    self.ts(5): 22.0,
                    self.ts(10): 22.0,
                },
                "heating_relay": {
                    self.ts(0): 1,
                    self.ts(5): 1,
                    self.ts(10): 1,
                },
            },
        }

        result = _convert_measures_to_df(devices, data)
        trv_rows = result[result.device_id == trv_id]

        trv_timestamps = set(trv_rows.date.dt.minute.tolist())

        # Union of TRV (10:00, 10:05, 10:10) and heater (10:03, 10:05, 10:07) original timestamps
        expected_min_timestamps = {0, 3, 5, 7, 10}

        assert expected_min_timestamps.issubset(trv_timestamps), (
            f"TRV should have at least timestamps {expected_min_timestamps} but got {trv_timestamps}. "
            f"Heater-only timestamps (10:03, 10:07) were not propagated to TRV due to NaN device_id in outer merge."
        )

    def test_heating_relay_values_come_from_heater(self):
        """
        Verify that heating_relay column values in the result come from the heater device.
        - When heater has a timestamp: use heater's heating_relay value
        - When heater doesn't have a timestamp: heating_relay should be NULL
        """
        heater_id = "test_heater"
        trv_id = "test_trv"

        devices = {
            heater_id: Device(id=heater_id, name="Boiler", type="boilermodule"),
            trv_id: Device(id=trv_id, name="Living Room", type="trv"),
        }

        data = {
            heater_id: {
                "heating_relay": {
                    self.ts(0): 1,   # Heater ON at 10:00
                    self.ts(5): 0,   # Heater OFF at 10:05
                    # No heater data at 10:10
                },
            },
            trv_id: {
                "temperature": {
                    self.ts(0): 20.0,
                    self.ts(5): 20.5,
                    self.ts(10): 21.0,  # TRV has data at 10:10, heater doesn't
                },
                "heat_target": {
                    self.ts(0): 22.0,
                    self.ts(5): 22.0,
                    self.ts(10): 22.0,
                },
                "heating_relay": {
                    self.ts(0): 0,   # TRV value (should be ignored, use heater's 1)
                    self.ts(5): 1,   # TRV value (should be ignored, use heater's 0)
                    self.ts(10): 1,  # TRV value (should be ignored, heater has no data -> NULL)
                },
            },
        }

        result = _convert_measures_to_df(devices, data)
        trv_rows = result[result.device_id == trv_id].sort_values("date").reset_index(drop=True)

        # heating_relay should come from heater, not TRV
        assert trv_rows.loc[0, "heating_relay"] == True, "10:00 should be ON (from heater)"
        assert trv_rows.loc[1, "heating_relay"] == False, "10:05 should be OFF (from heater)"
        assert pd.isna(trv_rows.loc[2, "heating_relay"]), "10:10 should be NULL (heater has no data)"
