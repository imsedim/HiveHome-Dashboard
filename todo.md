# Bugs
- [x] Pandas gives errors on load
- [x] Rework get_device_data() to use last actual datapoint for next batch (so, don't use last dataframe date since then measurements are skipped)
- [x] merge_device_data updates period.to with latest measurement timestamp

# Heating costs
- [ ] Display heating lenght and heating cost per day (include standing charge)
- [ ] Add heating cost to heating_relay patches tooltip
- [ ] Estimate kW rate per minute of heating using ML
  - [x] Download dataset from Octopus
  - [ ] Create function to calculate heating length for X days for EON
  - [ ] Create training dataset from Octopus + EON
    - [ ] For Octopus, include daily aggregations
    - [ ] for fun, isolate periods where only one TRV was active and fit price per minute there
      - [ ] for fun, figure out the cost of 'waisted boiler cycles' for undershots
  - [ ] Fit and visualise linear function
  - [ ] Fit something non-linear
- [ ] Add settings page to store heating prices and kW per minute

# Tasks
- [ ] Highlight segments of device line where heating was demanded / received
- [ ] Make daily plot always show all devices (would be transparent if not selected)
    - [ ] Change device selector to multiple-choice
- [ ] Add weekly period with zoomable plots
- [ ] Rethink aggregate plots - what story does this visualisation tell? 
  - [ ] ??? For month / year / week - add granularity controls (by week/month)
- [ ] Move calculations of trv's heating length to hive.py
  - [ ] break heating segment at midnight to allow daily calcs
- [ ] try candlestick or area with y,y2 plot for months / years
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




