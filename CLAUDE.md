# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Heat is a Streamlit dashboard for monitoring and analyzing Hive Home smart heating data. It fetches historical measurements from the Hive API, processes them into DataFrames, and displays interactive temperature/heating charts.

## Commands

```bash
# Run the application
streamlit run app.py

# Install dependencies
pip3 install -r requirements.txt

# Install dev dependencies (Jupyter, watchdog)
pip install -r requirements.dev.txt
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Streamlit UI (app.py)                       │
│  Date/period selection, device selection, Vega-Lite charts  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               Data Layer (hive.py)                          │
│  - Async Hive API client                                    │
│  - DataFrame processing (resampling, interpolation)         │
│  - Multi-layer caching (pickle, JSON)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          Authentication (cognito.py)                        │
│  AWS Cognito SRP auth with MFA, token refresh               │
└─────────────────────────────────────────────────────────────┘
```

**File responsibilities:**
- `app.py` - Streamlit UI, chart specs, user interactions
- `hive.py` - API client, data fetching, DataFrame processing, caching
- `cognito.py` - AWS Cognito authentication wrapper with device confirmation
- `utils.py` - Timing decorators, date/period helpers, HTML rendering

## Data Flow

1. Cognito auth → tokens cached to `data/tokens.json`
2. Async fetch measurements from Hive API → cached as raw JSON in `data/raw/`
3. Process into DataFrame → interpolate, resample to 5-min intervals → cache to `data/device_data.pickle`
4. Filter/aggregate cached data → render Vega-Lite charts

## Key Measurements

- `heat_target` - Set target temperature
- `heating_relay` - Boiler active state
- `temperature` - Actual temperature
- `heating_demand` / `heating_demand_percentage` - TRV demand

## Important Details

- Time is UTC internally, converted to Europe/London for display
- Device ID mapping is hardcoded in `hive.py` around line 198
- Heating segments break at midnight for daily metrics
- Uses quadratic interpolation for temperature, forward-fill for relay states
