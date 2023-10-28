# Bugs
[ ] Pandas gives errors on load
    /Users/sedim/Documents/Projects/Heat/hive.py:125: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  df = (pd.concat((pd.concat(pd.DataFrame(device_data.get(m, {}).items(), columns=["date", "value"]).assign(measure=m)
[ ] On month or year view, no heating_relay statistics since April'23

# Tasks
[ ] Calculate length of heating_relay patch for heater
[ ] Calculate length of heating_relay_heater intersected with heating_demand for trvs
[ ] either candlestick or area with y,y2 plot for months / years
[x] Add spinner for device data update on streamlit
[x] Reimplement authentication without username / password in files 
[x] Understand heating_relay for TRVs and boiler
[x] Display heating_relay for boiler
[x] Display target temperature 
[x] Display heating_relay for TRVs




