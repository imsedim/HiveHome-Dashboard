import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from hive import _add_heating_stats, _resample_heating_data

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
        df = _add_heating_stats(df, heater_id="test_device")

        assert df.loc[0, "heating_minutes"] == 1.0  # 10:00 -> 10:01
        assert df.loc[1, "heating_minutes"] == 1.0  # 10:01 -> 10:02
        assert pd.isna(df.loc[2, "heating_minutes"])  # heating_relay is False

    def test_heating_minutes_with_gaps(self):
        """Test heating_minutes with gaps in heating."""
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ), [True, True, False, False, True])
        df = _add_heating_stats(df, heater_id="test_device")

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
        df = _add_heating_stats(df, heater_id="test_device")

        # heating_end should be 10:02 (next_date of the last ON row)
        assert df.loc[0, "heating_end"] == pd.Timestamp("2024-01-01 10:02:00", tz=UTC_TZ)
        assert df.loc[1, "heating_end"] == pd.Timestamp("2024-01-01 10:02:00", tz=UTC_TZ)
        assert pd.isna(df.loc[2, "heating_end"])  # OFF row has no heating_end

    def test_heating_end_corresponds_to_start_and_length(self):
        """heating_end - heating_start should equal heating_length (in minutes)."""
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
                           [True, True, True, False, False, True, True, False])
        df = _add_heating_stats(df, heater_id="test_device")

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
        df = _add_heating_stats(df, heater_id="test_device")

        assert pd.isna(df.loc[0, "heating_end"])
        assert pd.notna(df.loc[1, "heating_end"])
        assert pd.notna(df.loc[2, "heating_end"])
        assert pd.isna(df.loc[3, "heating_end"])

    def test_multiple_segments_have_different_heating_ends(self):
        """Each heating segment should have its own heating_end."""
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
                           [True, True, False, True, True, True, False])
        df = _add_heating_stats(df, heater_id="test_device")

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
        df = _add_heating_stats(df, heater_id="test_device")
        resampled = _resample_heating_data(df, freq="5min")

        assert len(resampled) == 2
        assert resampled.iloc[0]["heating_minutes"] == 3.0  # bin 10:00: 3 minutes
        assert resampled.iloc[1]["heating_minutes"] == 2.0  # bin 10:05: 2 minutes

    def test_bin_labels(self):
        """Bins are labeled with start time (label='left')."""
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
                           [True, True, True, False, False, True, True, False, False, False])
        df = _add_heating_stats(df, heater_id="test_device")
        resampled = _resample_heating_data(df, freq="5min")

        dates = resampled.date.tolist()
        assert dates[0] == pd.Timestamp("2024-01-01 10:00:00", tz=UTC_TZ)
        assert dates[1] == pd.Timestamp("2024-01-01 10:05:00", tz=UTC_TZ)

    def test_boundary_timestamp_assignment(self):
        """Timestamp 10:05 goes to bin [10:05, 10:10), not [10:00, 10:05)."""
        # Timestamps: 10:00, ..., 10:07 - heating ON at 10:04, 10:05, 10:06
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
                           [False, False, False, False, True, True, True, False])
        df = _add_heating_stats(df, heater_id="test_device")
        resampled = _resample_heating_data(df, freq="5min")

        # 10:04 in bin [10:00, 10:05), 10:05-10:06 in bin [10:05, 10:10)
        assert resampled.iloc[0]["heating_minutes"] == 1.0  # bin 10:00
        assert resampled.iloc[1]["heating_minutes"] == 2.0  # bin 10:05 (10:07 has 0 mins)

    def test_total_preserved(self):
        """Total heating_minutes preserved after resampling."""
        df = create_test_df(datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
                           [True, True, True, False, False, True, True, False])
        df = _add_heating_stats(df, heater_id="test_device")
        total = df.heating_minutes.sum()

        resampled = _resample_heating_data(df, freq="5min")
        assert resampled.heating_minutes.sum() == total


def create_multi_device_df(start_time: datetime, boiler_relay: list[bool],
                           trv_demands: dict[str, list[bool]]) -> pd.DataFrame:
    """
    Create a multi-device DataFrame with boiler and TRVs.

    Args:
        start_time: Starting timestamp
        boiler_relay: Boiler heating_relay values (list of bool)
        trv_demands: Dict mapping TRV device_id to heating_demand values

    The boiler's heating_relay is broadcast to all TRVs. A TRV's effective
    heating_relay = boiler_relay & trv_demand (computed by add_heating_stats).
    """
    n = len(boiler_relay)
    dfs = []

    # Create boiler rows
    boiler_df = pd.DataFrame({
        "date": pd.to_datetime([start_time + timedelta(minutes=i) for i in range(n)], utc=True),
        "device_id": "boiler",
        "heating_relay": pd.array(boiler_relay, dtype="boolean"),
        "heating_demand": pd.array([False] * n, dtype="boolean"),
        "heating_demand_percentage": [0.0] * n,
        "temperature": [20.0] * n,
        "heat_target": [21.0] * n,
    })
    dfs.append(boiler_df)

    # Create TRV rows - heating_relay comes from boiler, heating_demand from TRV
    for trv_id, demands in trv_demands.items():
        assert len(demands) == n, f"TRV {trv_id} demands must match boiler length"
        trv_df = pd.DataFrame({
            "date": pd.to_datetime([start_time + timedelta(minutes=i) for i in range(n)], utc=True),
            "device_id": trv_id,
            "heating_relay": pd.array(boiler_relay, dtype="boolean"),
            "heating_demand": pd.array(demands, dtype="boolean"),
            "heating_demand_percentage": [100.0 if d else 0.0 for d in demands],
            "temperature": [20.0] * n,
            "heat_target": [21.0] * n,
        })
        dfs.append(trv_df)

    return pd.concat(dfs, ignore_index=True).sort_values(["device_id", "date"]).reset_index(drop=True)


class TestTopHeatingDevice:
    """Tests for top_device_id and top_device_minutes computation."""

    def test_basic_top_device(self):
        """Single boiler segment with multiple TRVs - top device is the one with most minutes."""
        # Boiler ON for 5 minutes (10:00 - 10:04)
        # TRV_A demands for 3 minutes (10:00-10:02)
        # TRV_B demands for 4 minutes (10:00-10:03)
        # TRV_B should be top device with 4 minutes
        df = create_multi_device_df(
            datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
            boiler_relay=[True, True, True, True, True, False],
            trv_demands={
                "trv_a": [True, True, True, False, False, False],
                "trv_b": [True, True, True, True, False, False],
            }
        )
        df = _add_heating_stats(df, heater_id="boiler")

        # Check boiler rows have top device info
        boiler_rows = df[df.device_id == "boiler"]
        heating_boiler = boiler_rows[boiler_rows.heating_relay]

        assert heating_boiler.iloc[0]["top_device_id"] == "trv_b"
        assert heating_boiler.iloc[0]["top_device_minutes"] == 4.0

    def test_no_trv_active(self):
        """Boiler on but no TRV demanding heat - both columns should be null."""
        df = create_multi_device_df(
            datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
            boiler_relay=[True, True, True, False],
            trv_demands={
                "trv_a": [False, False, False, False],
                "trv_b": [False, False, False, False],
            }
        )
        df = _add_heating_stats(df, heater_id="boiler")

        boiler_rows = df[df.device_id == "boiler"]
        heating_boiler = boiler_rows[boiler_rows.heating_relay]

        assert pd.isna(heating_boiler.iloc[0]["top_device_id"])
        assert pd.isna(heating_boiler.iloc[0]["top_device_minutes"])

    def test_tie_handling(self):
        """Two TRVs with equal minutes - should pick first in device_id sort order."""
        # Both TRVs active for exactly 2 minutes
        df = create_multi_device_df(
            datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
            boiler_relay=[True, True, True, False],
            trv_demands={
                "trv_a": [True, True, False, False],
                "trv_z": [True, True, False, False],
            }
        )
        df = _add_heating_stats(df, heater_id="boiler")

        boiler_rows = df[df.device_id == "boiler"]
        heating_boiler = boiler_rows[boiler_rows.heating_relay]

        # trv_a comes before trv_z alphabetically
        assert heating_boiler.iloc[0]["top_device_id"] == "trv_a"
        assert heating_boiler.iloc[0]["top_device_minutes"] == 2.0

    def test_multiple_boiler_segments(self):
        """Each boiler segment has independent top device."""
        # Segment 1: 10:00-10:02, TRV_A=2min, TRV_B=1min -> top=TRV_A
        # Segment 2: 10:04-10:06, TRV_A=1min, TRV_B=2min -> top=TRV_B
        df = create_multi_device_df(
            datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
            boiler_relay=[True, True, True, False, True, True, True, False],
            trv_demands={
                "trv_a": [True, True, False, False, True, False, False, False],
                "trv_b": [True, False, False, False, True, True, False, False],
            }
        )
        df = _add_heating_stats(df, heater_id="boiler")

        boiler_rows = df[df.device_id == "boiler"].sort_values("date")

        # First segment (rows where heating_start is 10:00)
        seg1 = boiler_rows[boiler_rows.heating_start == pd.Timestamp("2024-01-01 10:00:00", tz=UTC_TZ)]
        assert seg1.iloc[0]["top_device_id"] == "trv_a"
        assert seg1.iloc[0]["top_device_minutes"] == 2.0

        # Second segment (rows where heating_start is 10:04)
        seg2 = boiler_rows[boiler_rows.heating_start == pd.Timestamp("2024-01-01 10:04:00", tz=UTC_TZ)]
        assert seg2.iloc[0]["top_device_id"] == "trv_b"
        assert seg2.iloc[0]["top_device_minutes"] == 2.0

    def test_fragmented_trv_within_boiler_segment(self):
        """TRV with multiple heating segments within single boiler segment - minutes are summed."""
        # Boiler ON continuously: 10:00-10:09 (10 minutes)
        # TRV_A: ON 10:00-10:01, OFF 10:02-10:03, ON 10:04-10:05 = 2+2 = 4 minutes
        # TRV_B: ON 10:00-10:02 = 3 minutes
        # TRV_A should win with 4 minutes total (even though fragmented)
        df = create_multi_device_df(
            datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
            boiler_relay=[True, True, True, True, True, True, True, True, True, True, False],
            trv_demands={
                "trv_a": [True, True, False, False, True, True, False, False, False, False, False],
                "trv_b": [True, True, True, False, False, False, False, False, False, False, False],
            }
        )
        df = _add_heating_stats(df, heater_id="boiler")

        boiler_rows = df[df.device_id == "boiler"]
        heating_boiler = boiler_rows[boiler_rows.heating_relay]

        # TRV_A has 4 minutes total (2 + 2), TRV_B has 3 minutes
        assert heating_boiler.iloc[0]["top_device_id"] == "trv_a"
        assert heating_boiler.iloc[0]["top_device_minutes"] == 4.0

    def test_trv_rows_have_null_top_device(self):
        """TRV rows should always have null top_device_id and top_device_minutes."""
        df = create_multi_device_df(
            datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
            boiler_relay=[True, True, True, False],
            trv_demands={
                "trv_a": [True, True, False, False],
            }
        )
        df = _add_heating_stats(df, heater_id="boiler")

        trv_rows = df[df.device_id == "trv_a"]
        assert trv_rows["top_device_id"].isna().all()
        assert trv_rows["top_device_minutes"].isna().all()

    def test_top_device_preserved_after_resampling(self):
        """top_device_id and top_device_minutes are preserved after resampling."""
        df = create_multi_device_df(
            datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
            boiler_relay=[True, True, True, True, True, False],
            trv_demands={
                "trv_a": [True, True, True, True, False, False],
            }
        )
        df = _add_heating_stats(df, heater_id="boiler")
        resampled = _resample_heating_data(df, freq="5min")

        boiler_resampled = resampled[resampled.device_id == "boiler"]
        # First bin (10:00) should have top device info
        first_bin = boiler_resampled[boiler_resampled.date == pd.Timestamp("2024-01-01 10:00:00", tz=UTC_TZ)]
        assert first_bin.iloc[0]["top_device_id"] == "trv_a"
        assert first_bin.iloc[0]["top_device_minutes"] == 4.0


class TestTopDeviceSharePct:
    """Tests for top_device_share_pct calculation."""

    def test_share_pct_two_trvs(self):
        """Share percentage with two TRVs: 6 and 4 minutes -> top device = 60%."""
        # Boiler ON for 10 minutes
        # TRV_A: 6 minutes, TRV_B: 4 minutes
        # Total TRV minutes = 10, top device share = 6/10 = 60%
        df = create_multi_device_df(
            datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
            boiler_relay=[True] * 10 + [False],
            trv_demands={
                "trv_a": [True] * 6 + [False] * 5,
                "trv_b": [True] * 4 + [False] * 7,
            }
        )
        df = _add_heating_stats(df, heater_id="boiler")

        boiler_rows = df[df.device_id == "boiler"]
        heating_boiler = boiler_rows[boiler_rows.heating_relay]

        assert heating_boiler.iloc[0]["top_device_id"] == "trv_a"
        assert heating_boiler.iloc[0]["top_device_share_pct"] == 60

    def test_share_pct_single_trv(self):
        """Single TRV active -> share percentage = 100%."""
        df = create_multi_device_df(
            datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
            boiler_relay=[True, True, True, True, False],
            trv_demands={
                "trv_a": [True, True, True, True, False],
            }
        )
        df = _add_heating_stats(df, heater_id="boiler")

        boiler_rows = df[df.device_id == "boiler"]
        heating_boiler = boiler_rows[boiler_rows.heating_relay]

        assert heating_boiler.iloc[0]["top_device_share_pct"] == 100

    def test_share_pct_null_when_no_trvs(self):
        """No TRVs active -> share percentage should be null."""
        df = create_multi_device_df(
            datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
            boiler_relay=[True, True, True, False],
            trv_demands={
                "trv_a": [False, False, False, False],
            }
        )
        df = _add_heating_stats(df, heater_id="boiler")

        boiler_rows = df[df.device_id == "boiler"]
        heating_boiler = boiler_rows[boiler_rows.heating_relay]

        assert pd.isna(heating_boiler.iloc[0]["top_device_share_pct"])

    def test_share_pct_rounded_to_integer(self):
        """Share percentage is rounded to integer."""
        # 3 and 2 minutes -> 3/5 = 60% (exactly)
        # 5 and 3 minutes -> 5/8 = 62.5% -> rounds to 62%
        df = create_multi_device_df(
            datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
            boiler_relay=[True] * 8 + [False],
            trv_demands={
                "trv_a": [True] * 5 + [False] * 4,
                "trv_b": [True] * 3 + [False] * 6,
            }
        )
        df = _add_heating_stats(df, heater_id="boiler")

        boiler_rows = df[df.device_id == "boiler"]
        heating_boiler = boiler_rows[boiler_rows.heating_relay]

        # 5/8 = 62.5%, rounds to 62
        assert heating_boiler.iloc[0]["top_device_share_pct"] == 62

    def test_share_pct_preserved_after_resampling(self):
        """top_device_share_pct is preserved after resampling."""
        df = create_multi_device_df(
            datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
            boiler_relay=[True] * 6 + [False],
            trv_demands={
                "trv_a": [True] * 4 + [False] * 3,
                "trv_b": [True] * 2 + [False] * 5,
            }
        )
        df = _add_heating_stats(df, heater_id="boiler")
        resampled = _resample_heating_data(df, freq="5min")

        boiler_resampled = resampled[resampled.device_id == "boiler"]
        first_bin = boiler_resampled[boiler_resampled.date == pd.Timestamp("2024-01-01 10:00:00", tz=UTC_TZ)]

        # 4 / (4+2) = 4/6 = 66.67% -> rounds to 67%
        assert first_bin.iloc[0]["top_device_share_pct"] == 67

    def test_share_pct_trv_rows_null(self):
        """TRV rows should have null top_device_share_pct."""
        df = create_multi_device_df(
            datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC_TZ),
            boiler_relay=[True, True, True, False],
            trv_demands={
                "trv_a": [True, True, False, False],
            }
        )
        df = _add_heating_stats(df, heater_id="boiler")

        trv_rows = df[df.device_id == "trv_a"]
        assert trv_rows["top_device_share_pct"].isna().all()