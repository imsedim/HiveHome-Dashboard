import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from hive import Device

UTC_TZ = ZoneInfo("UTC")

# Real device IDs from the production system (for reference)
HEATER_ID = "caaaf8cd-a18e-47e4-9ecb-10b9d4df3397"
TRV_ID = "8da110bf-43e0-4ce4-9e6e-f641e18b8f58"


@pytest.fixture
def sample_devices():
    """Minimal device fixture matching production structure."""
    return {
        HEATER_ID: Device(id=HEATER_ID, name="Zone 1", type="boilermodule"),
        TRV_ID: Device(id=TRV_ID, name="Diana's room", type="trv"),
    }


@pytest.fixture
def sample_raw_data():
    """
    Minimal raw API response structure.

    Structure mirrors data/raw/*.json format:
    - Keys are device IDs
    - Each device has measure dicts with Unix timestamp string keys
    """
    # Base timestamp: 2024-01-07 10:00:00 UTC
    base_ts = int(datetime(2024, 1, 7, 10, 0, 0, tzinfo=UTC_TZ).timestamp())

    return {
        HEATER_ID: {
            "heating_relay": {
                str(base_ts + 0): 0,
                str(base_ts + 60): 1,
                str(base_ts + 120): 1,
                str(base_ts + 180): 1,
                str(base_ts + 240): 0,
                str(base_ts + 300): 0,
            },
            "temperature": {
                str(base_ts + 0): 18.5,
                str(base_ts + 60): 18.6,
                str(base_ts + 120): 18.7,
                str(base_ts + 180): 18.8,
                str(base_ts + 240): 18.9,
                str(base_ts + 300): 19.0,
            },
            "heat_target": {
                str(base_ts + 0): 21,
                str(base_ts + 60): 21,
                str(base_ts + 120): 21,
                str(base_ts + 180): 21,
                str(base_ts + 240): 21,
                str(base_ts + 300): 21,
            },
        },
        TRV_ID: {
            "heating_demand": {
                str(base_ts + 0): 0,
                str(base_ts + 60): 1,
                str(base_ts + 120): 1,
                str(base_ts + 180): 1,
                str(base_ts + 240): 0,
                str(base_ts + 300): 0,
            },
            "heating_demand_percentage": {
                str(base_ts + 0): 0,
                str(base_ts + 60): 50,
                str(base_ts + 120): 60,
                str(base_ts + 180): 70,
                str(base_ts + 240): 0,
                str(base_ts + 300): 0,
            },
            "temperature": {
                str(base_ts + 0): 17.5,
                str(base_ts + 60): 17.6,
                str(base_ts + 120): 17.7,
                str(base_ts + 180): 17.8,
                str(base_ts + 240): 17.9,
                str(base_ts + 300): 18.0,
            },
            "heat_target": {
                str(base_ts + 0): 20,
                str(base_ts + 60): 20,
                str(base_ts + 120): 20,
                str(base_ts + 180): 20,
                str(base_ts + 240): 20,
                str(base_ts + 300): 20,
            },
        },
    }
