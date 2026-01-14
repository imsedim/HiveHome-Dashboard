import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from hive import add_heating_stats, resample_heating_data

UTC_TZ = ZoneInfo("UTC")


def create_test_df(start_time: datetime, heating_relay_values: list[bool], device_id: str = "test_device") -> pd.DataFrame:
    """Create a DataFrame matching the format expected by add_heating_stats and resample_heating_data."""
    n = len(heating_relay_values)
    return pd.DataFrame({
        "date": pd.to_datetime([start_time + timedelta(minutes=i) for i in range(n)], utc=True),
        "device_id": device_id,
        "heating_relay": pd.array(heating_relay_values, dtype="boolean"),
        "heating_demand": pd.array([False] * n, dtype="boolean"),
        "heating_demand_percentage": [0.0] * n,
        "temperature": [20.0] * n,
        "heat_target": [21.0] * n,
    })


class TestHeatingMinutesCalculation:
    """Tests for add_heating_stats function."""

    def test_forward_looking_heating_minutes(self):
        """heating_minutes represents time from current to next timestamp."""
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ), [True, True, False])
        df = add_heating_stats(df, heater_id="test_device")

        assert df.loc[0, "heating_minutes"] == 1.0  # 10:00 -> 10:01
        assert df.loc[1, "heating_minutes"] == 1.0  # 10:01 -> 10:02
        assert pd.isna(df.loc[2, "heating_minutes"])  # heating_relay is False

    def test_heating_minutes_with_gaps(self):
        """Test heating_minutes with gaps in heating."""
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ), [True, True, False, False, True])
        df = add_heating_stats(df, heater_id="test_device")

        assert df.loc[0, "heating_minutes"] == 1.0
        assert df.loc[1, "heating_minutes"] == 1.0
        assert pd.isna(df.loc[2, "heating_minutes"])
        assert pd.isna(df.loc[3, "heating_minutes"])
        assert df.loc[4, "heating_minutes"] == 0.0  # last row has no next timestamp


class TestHeatingEndCalculation:
    """Tests for heating_end calculation in add_heating_stats."""

    def test_heating_end_matches_segment_boundary(self):
        """heating_end should be the timestamp after the last heating minute in segment."""
        # ON ON OFF - segment ends after minute 1
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ), [True, True, False])
        df = add_heating_stats(df, heater_id="test_device")

        # heating_end should be 10:02 (next_date of the last ON row)
        assert df.loc[0, "heating_end"] == pd.Timestamp("2024-01-01 10:02:00", tz=UTC_TZ)
        assert df.loc[1, "heating_end"] == pd.Timestamp("2024-01-01 10:02:00", tz=UTC_TZ)
        assert pd.isna(df.loc[2, "heating_end"])  # OFF row has no heating_end

    def test_heating_end_corresponds_to_start_and_length(self):
        """heating_end - heating_start should equal heating_length (in minutes)."""
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
                           [True, True, True, False, False, True, True, False])
        df = add_heating_stats(df, heater_id="test_device")

        # Check first segment (rows 0-2)
        for i in range(3):
            start = df.loc[i, "heating_start"]
            end = df.loc[i, "heating_end"]
            length = df.loc[i, "heating_length"]
            calculated_length = (end - start).total_seconds() / 60
            assert calculated_length == length, f"Row {i}: {calculated_length} != {length}"

        # Check second segment (rows 5-6)
        for i in range(5, 7):
            start = df.loc[i, "heating_start"]
            end = df.loc[i, "heating_end"]
            length = df.loc[i, "heating_length"]
            calculated_length = (end - start).total_seconds() / 60
            assert calculated_length == length, f"Row {i}: {calculated_length} != {length}"

    def test_heating_end_null_when_not_heating(self):
        """heating_end should be null when heating_relay is False."""
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
                           [False, True, True, False])
        df = add_heating_stats(df, heater_id="test_device")

        assert pd.isna(df.loc[0, "heating_end"])
        assert pd.notna(df.loc[1, "heating_end"])
        assert pd.notna(df.loc[2, "heating_end"])
        assert pd.isna(df.loc[3, "heating_end"])

    def test_multiple_segments_have_different_heating_ends(self):
        """Each heating segment should have its own heating_end."""
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
                           [True, True, False, True, True, True, False])
        df = add_heating_stats(df, heater_id="test_device")

        # First segment: rows 0-1, ends at 10:02
        assert df.loc[0, "heating_end"] == pd.Timestamp("2024-01-01 10:02:00", tz=UTC_TZ)
        assert df.loc[1, "heating_end"] == pd.Timestamp("2024-01-01 10:02:00", tz=UTC_TZ)

        # Second segment: rows 3-5, ends at 10:06
        assert df.loc[3, "heating_end"] == pd.Timestamp("2024-01-01 10:06:00", tz=UTC_TZ)
        assert df.loc[4, "heating_end"] == pd.Timestamp("2024-01-01 10:06:00", tz=UTC_TZ)
        assert df.loc[5, "heating_end"] == pd.Timestamp("2024-01-01 10:06:00", tz=UTC_TZ)


class TestResampleHeatingData:
    """Tests for resample_heating_data function."""

    def test_correct_binning(self):
        """
        closed='left', label='left' correctly bins forward-looking data.
        Bin [10:00, 10:05) contains timestamps 10:00-10:04, labeled "10:00".
        """
        # Pattern: ON ON ON OFF OFF ON ON OFF
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
                           [True, True, True, False, False, True, True, False])
        df = add_heating_stats(df, heater_id="test_device")
        resampled = resample_heating_data(df, freq="5min")

        assert len(resampled) == 2
        assert resampled.iloc[0]["heating_minutes"] == 3.0  # bin 10:00: 3 minutes
        assert resampled.iloc[1]["heating_minutes"] == 2.0  # bin 10:05: 2 minutes

    def test_bin_labels(self):
        """Bins are labeled with start time (label='left')."""
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
                           [True, True, True, False, False, True, True, False, False, False])
        df = add_heating_stats(df, heater_id="test_device")
        resampled = resample_heating_data(df, freq="5min")

        dates = resampled.date.tolist()
        assert dates[0] == pd.Timestamp("2024-01-01 10:00:00", tz=UTC_TZ)
        assert dates[1] == pd.Timestamp("2024-01-01 10:05:00", tz=UTC_TZ)

    def test_boundary_timestamp_assignment(self):
        """Timestamp 10:05 goes to bin [10:05, 10:10), not [10:00, 10:05)."""
        # Timestamps: 10:00, ..., 10:07 - heating ON at 10:04, 10:05, 10:06
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
                           [False, False, False, False, True, True, True, False])
        df = add_heating_stats(df, heater_id="test_device")
        resampled = resample_heating_data(df, freq="5min")

        # 10:04 in bin [10:00, 10:05), 10:05-10:06 in bin [10:05, 10:10)
        assert resampled.iloc[0]["heating_minutes"] == 1.0  # bin 10:00
        assert resampled.iloc[1]["heating_minutes"] == 2.0  # bin 10:05 (10:07 has 0 mins)

    def test_total_preserved(self):
        """Total heating_minutes preserved after resampling."""
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
                           [True, True, True, False, False, True, True, False])
        df = add_heating_stats(df, heater_id="test_device")
        total = df.heating_minutes.sum()

        resampled = resample_heating_data(df, freq="5min")
        assert resampled.heating_minutes.sum() == total