# Bugs
- [x] fix heating_length calc bugs (forward-looking resampling and heating_relay propagation to trvs)
- [x] test coverage
- [x] Pandas gives errors on load
- [x] Rework get_device_data() to use last actual datapoint for next batch (so, don't use last dataframe date since then measurements are skipped)
- [x] merge_device_data updates period.to with latest measurement timestamp
- [x] fix is_heater colour to red
- [x] optimise refresh speed
  - [x] 30% optimize authentication
  - [X] 30% optimize data fetch - avoid useless merge of history
  - [X] 30% optimize post processing - do no reprocess data older than last fetch round up to 1 day


# Heating costs
- [x] Display heating lenght and heating cost per day (include standing charge)
- [x] Estimate kW rate per minute of heating using ML
  - [x] Download dataset from Octopus
  - [x] Create function to calculate heating length for X days for EON
  - [x] Create training dataset from Octopus + EON
    - [x] For Octopus, include daily aggregations (diff rolling windows)
    - [x] Add duration of period (hours) as a feature (since water usage depends on it)
  - [x] Fit and visualise linear function

# Tasks
- [ ] Rethink aggregate plots - what story does this visualisation tell? 
  - [ ] ??? For month / year / week - add granularity controls (by week/month)
- [x] add top heating device stats
- [x] high-level stats: % or Hours each TRV was active - vertical bar plot? 
- [x] Highlight segments of device line where heating was demanded / received
- [x] Make daily plot always show all devices (would be transparent if not selected)
    - [x] Revert target temperature color to what it was (greenish)
    - [x] Add toggle for heating_demand_percentage (too noisy)
- [x] Add weekly period
- [x] Move calculations of trv's heating length to hive.py
  - [x] break heating segment at midnight to allow daily calcs
- [x] Collect and display heating_demand_percentage
  - [x] Store raw data dumps for future consumption
- [x] Calculate length of heating_relay patch for heater
- [x] Calculate length of heating_relay_heater intersected with heating_demand for trvs
- [x] Add spinner for device data update on streamlit
- [x] Reimplement authentication without username / password in files 
- [x] Understand heating_relay for TRVs and boiler
- [x] Display heating_relay for boiler
- [x] Display target temperature 
- [x] Display heating_relay for TRVs




