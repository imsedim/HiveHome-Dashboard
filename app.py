import asyncio
from decimal import Decimal
import random
from cognito import BotoSession
from utils import DATE_TYPE_DAY, render_date_title, timeit, write_date_picker
from datetime import date
import streamlit as st
import hive
import pandas as pd

st.set_page_config(page_title="Heating Dashboard",
                   layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
header {visibility: hidden;}
footer {visibility: hidden;}
.css-1oe6wy4 {padding-top: 0rem;}
.css-ng1t4o {padding-top: 0rem;}
.css-18e3th9 {padding-top: 0rem;}
.st-emotion-cache-z5fcl4 {
    padding: 0rem 1rem 0rem 2.5rem;
}
.st-emotion-cache-10oheav {
  padding-top: 0rem;
}
div.stCheckbox {margin-top: 1px; margin-bottom: -20px;}
div.stCheckbox label span {height:18px;width:18px}
div[data-testid=stHorizontalBlock] > div > div > div[data-testid=stVerticalBlock] > div > div.row-widget.stRadio > label {display:none;}
div[data-testid=stHorizontalBlock] > div > div > div[data-testid=stVerticalBlock] > div > div.row-widget.stRadio > div > label div {font-size:10pt; white-space: nowrap;}

</style>
"""


@st.cache_resource
def get_cache():
    return dict()


async def get_data(period: tuple, refresh: bool = False) -> pd.DataFrame:
    if refresh:
        get_cache.clear()
    cache = get_cache()

    cached_value = cache.get(period)
    if cached_value is not None:
        return cached_value

    data = cache.get(None)
    if data is None:
        try:
            data = await hive.get_device_data(refresh)
        except:
            raise
        if data is None:
            return None
        data["date"] = data["date"].dt.tz_convert('Europe/London')
        data["trv_heating_relay"] = data.heating_relay & ~data.is_heater
        cache[None] = data

    date_from, date_to = pd.to_datetime(period).tz_localize('Europe/London')
    data = data.query("@date_from <= date < @date_to")
    cache[period] = data
    return data

def erase_cache():
    hive.CACHED_DEVICE_DATA_FILE.exists() and hive.CACHED_DEVICE_DATA_FILE.unlink()
    get_cache.clear()

def _calculate_price(heating_length: Decimal, days: int = 0,
                     price_per_kw: Decimal = 0.0689682, standing_charge: Decimal = 0.2746842) -> Decimal:
    # kw = heating_length * 7.07610677 + 0.25658080647633885
    kw = heating_length * 6.30897471 + 3.4544852333845384
    return kw * price_per_kw + standing_charge * days


async def app():
    st.markdown(CSS, unsafe_allow_html=True)

    *date_cols, _, col1, col2, _, col3 = st.columns((27.5, 75, 22.5, 20, 60, 80, 40, 300))

    period = write_date_picker(date_cols)
    device_data = await get_data(period)
    # data_is_expired = device_data is not None and (date_from == (today := _this(DATE_TYPE_DAY))) and device_data.date.max() < make_utc(today)
    refresh = col1.button(("❗ " if device_data is None else "⚠️ " if device_data.empty else "") + "Refresh")
    interpolate = col2.checkbox("Interpolate", value=True)
    display_demand_percentage = col2.checkbox("Demand %", value=False)
    demand_overlay = col2.checkbox("Demand Overlay", value=False)

    if refresh or (st.session_state.get("auth_state") == "complete"):
        if (st.session_state.get("auth_state") == "complete") or check_auth():
            st.session_state["auth_state"] = None
            with st.spinner('Loading data'):
                device_data = await get_data(period, True)
        else:
            st.session_state["auth_state"] = "pending"

    if st.session_state.get("auth_state") == "pending":
        authenticate()

    if device_data is None:
        st.stop()

    devices = await hive.get_devices()
    devices = {d.id: d for d in sorted(devices.values(), key=lambda x: x.name)
               if d.is_trv or d.is_heater}

    device_id = st.radio("Device", [""] + list(devices.keys()),
                         format_func=lambda x: devices[x].name if x else "All devices", key="device_id",
                         label_visibility="collapsed", horizontal=True)
    device = devices[device_id] if device_id else None
    is_daily_chart = st.session_state["date_type"] == DATE_TYPE_DAY
    device_data = prepare_data(devices, device_data, device_id, not is_daily_chart)

    heating_duration = get_heating_duration(is_daily_chart, device_id, devices, device_data)
    heating_hours, heating_minutes = int(heating_duration) // 60, int(heating_duration) % 60
    heating_duration_str = (f"{heating_hours}h" if heating_hours else "") + (f" {heating_minutes}m" if heating_minutes else "")

    col3.metric(f"{device.name if device else 'Total'} heating", heating_duration_str, delta=f"£{_calculate_price(heating_duration/60):.2f}", delta_color="inverse")

    spec = get_daily_chart_spec([d.name for _, d in devices.items()], device, interpolate, display_demand_percentage, demand_overlay) if is_daily_chart else get_agg_chart_spec(device)
    st.vega_lite_chart(device_data, spec, use_container_width=False)

    with st.expander("Dataframe"):
        dump = device_data.query("device_id==@device_id").drop(columns=["device_name"]) if device_id else device_data
        st.dataframe(dump.drop(columns=["device_id", "is_heater"]))

    st.button("Erase Cache", on_click=erase_cache)


def authenticate():
    def _start_auth():
        t = hive.authenticate(hive.Credentials(username=st.session_state["login_username"],
                                               password=st.session_state["login_password"]))
        if t.id_token:
            st.session_state["auth_state"] = "complete"

    def _complete_auth():
        t = hive.authenticate(hive.Credentials(username=st.session_state["login_username"],
                                               mfa_code=st.session_state["login_mfa_code"]))
        if t.id_token:
            st.session_state["auth_state"] = "complete"

    tokens = hive.load_cached_tokens()
    credentials = hive.load_credentials()

    with st.form("Login"):
        st.text_input("Username", value=credentials.username, key="login_username")
        if not tokens.mfa_tokens:
            st.text_input("Password",  type="password", value=credentials.password, key="login_password")
            st.form_submit_button("Start", type="primary", on_click=_start_auth)
        else:
            st.text_input("SMS Code", key="login_mfa_code")
            st.form_submit_button("Complete", type="primary", on_click=_complete_auth)
    return False


def check_auth() -> bool:
    t = hive.load_cached_tokens()
    if t.id_token is None:
        return False
    try:
        hive.authenticate(None)
        return True
    except:
        return False


@timeit
def prepare_data(devices: dict, device_data: pd.DataFrame, device_id: str | None, aggregate: bool) -> pd.DataFrame:
    non_heating_source_rows = device_data["device_id"] != (device_id or hive.get_heater_id(devices))

    device_data = device_data.copy()
    device_measures = [x for x in hive.MEASURE_NAMES if x != "temperature"] if device_id else ["heating_relay"]
    device_data.loc[non_heating_source_rows, device_measures] = None

    if aggregate:
        device_data = (device_data.pipe(lambda x: x.query("device_id == @device_id") if device_id else x)
                       .assign(date=lambda x: x.date.dt.floor('d'))
                       .groupby(["device_id", "device_name", "date"])
                       .agg(low=('temperature', 'min'), high=('temperature', 'max'),
                            open=('temperature', 'first'), close=('temperature', 'last'))
                       .assign(heating_length=(device_data.drop_duplicates(subset=["device_id", "heating_start"])
                                               .assign(date=lambda x: x.heating_start.dt.tz_convert('Europe/London').dt.floor('d'))
                                               .groupby(["device_id", "device_name", "date"])
                                               .heating_length.sum()))
                       .assign(heating_length=lambda x: x.heating_length.fillna(0))
                       .assign(heating_relay=lambda x: x.heating_length > 0)
                       .reset_index())

    heating_minutes_str = ((heating_minutes := device_data.heating_length.dropna().astype(int)) % 60).astype(str) + "m"
    device_data = device_data.assign(heating_length_str=lambda x:
                                     heating_minutes_str
                                     .where(heating_minutes < 60, (heating_minutes // 60).astype(str) + "h " + heating_minutes_str)
                                     .where(x.heating_relay, ""))
    return device_data


def get_heating_duration(is_daily_chart: bool, device_id: str | None, devices: dict[str, hive.Device], device_data: pd.DataFrame) -> str:
    heating_source = device_id or hive.get_heater_id(devices)
    heating_duration = (device_data[device_data.device_id == heating_source]
                        .pipe(lambda x: x.drop_duplicates(subset=["heating_start", "heating_length"]) if is_daily_chart else x)
                        .heating_length.sum())

    return heating_duration


def get_agg_chart_spec(device: hive.Device | None) -> dict:
    spec = {"height": 460,
            "width": 1300,
            "title": f"Temperature for {device.name if device else 'all devices'} {render_date_title()} {' '.rjust(random.randint(1,10))}",
            "view": {"stroke": "transparent"},
            "layer": [
                # {"layer": [{"mark": {"type": "line", "tooltip": True, "point": False, "interpolate": "monotone", "strokeWidth": 3}},
                #            {"mark": {"type": "line", "tooltip": True, "point": False, "interpolate": "monotone", "strokeWidth": 3},
                #             "encoding": {"y": {"field": "high", "type": "quantitative", "stack": None,
                #                                "scale": {"zero": False}, "axis": {"grid": True, "title": "Max Temperature"}}}}],
                #  "encoding": {"y": {"field": "low", "type": "quantitative", "stack": None,
                #                     "scale": {"zero": False}, "axis": {"grid": True, "title": "Min Temperature"}},
                {"mark": {"type": "bar", "tooltip": True, "point": False, "interpolate": "monotone", "opacity": 1},
                 "encoding": {"y": {"field": "low", "type": "quantitative", "stack": None,
                                    "scale": {"zero": False}, "axis": {"grid": True, "title": "Min Temperature"}},
                              "y2": {"field": "high", "type": "quantitative", "stack": None,
                                     "scale": {"zero": False}, "axis": {"grid": True, "title": "Max Temperature"}},
                              "tooltip": [{"field": "date", "title": "time",  "type": "temporal", "timeUnit": "hoursminutes"},
                                          {"field": "low", "title": "min temperature"},
                                          {"field": "high", "title": "max temperature"},
                                          {"field": "device_name"}]}},

                {"mark": {"type": "bar", "transparent": True, "opacity": 0.7, "tooltip": True, "width": 15},
                 "encoding": {"y": {"field": "heating_length", "type": "quantitative",
                                    "scale": {"zero": False, "domain": [0, 2500]}, "axis": None},
                              "tooltip": [{"field": "date", "type": "temporal", "timeUnit": "datemonthyear", "title": "date"},
                                          {"field": "heating_length_str", "title": "heating length"}]}}
            ],
            "encoding": {"x": {"field": "date", "type": "temporal", "title": "time", "timeUnit": "datemonthyear",
                               "axis": {"ticks": True, "grid": True, "gridOpacity": 0.35}},
                         "color": {"field": "device_name", "type": "nominal", "title": ""}},
            "resolve": {"scale": {"y": "independent"}}}

    if device:
        spec["encoding"].pop("color")
        spec["layer"][1]["encoding"]["color"] = {"value": "red"}

        spec["layer"][0]["mark"]["color"] = {"x1": 1, "y1": 1, "x2": 1, "y2": 0, "gradient": "linear",
                                             "stops": [{"offset": 0, "color": "blue"}, {"offset": 1, "color": "green"}]}

    #     spec["layer"].append({"mark": {"type": "area", "interpolate": "step-after", "transparent": False, "opacity": 0.2, "tooltip": False, "color": "green"},
    #                           "encoding": {"y": {"field": "heating_demand", "type": "quantitative", "scale": {"zero": False, "domain": [0, 40]}, "axis": None}}})

    #     spec["layer"][0]["layer"].append({"mark": {"type": "area", "tooltip": True, "point": False, "interpolate": "step-after", "opacity": 0.05,
    #                                                "line": {"color": "#468B52", "strokeDash": [8, 8], "strokeWidth": 2, "opacity": 0.75},
    #                                                "color": {"x1": 1, "y1": 1, "x2": 1, "y2": 0, "gradient": "linear",
    #                                                          "stops": [{"offset": 0, "color": "white"}, {"offset": 1, "color": "darkred"}]}},
    #                                       "encoding": {"y": {"field": "heat_target", "type": "quantitative", "scale": {"zero": False}}}})

    return spec


def get_daily_chart_spec(device_names: list[str], device: hive.Device | None, interpolate: bool, display_demand_percentage: bool, demand_overlay: bool) -> dict:
    device_names = sorted(device_names)
    # blue80 blue40 red80 red40 blueGreen80 green40 orange80 orange50 purple80 gray40
    # 84c9ff #0068c9 #ff2b2b #ffabab #29b09d #7defa1 #ff8700 #ffd16a #6d3fc0 #d5dae5
    colours = ["#84c9ff", "#0068c9", "#6d3fc0", "#ffd16a", "#ff2b2b", "#ffabab",  "#ff8700", "#29b09d", "#7defa1", "#ffd16a", "#6d3fc0", "#d5dae5", "#ff8700"]
#
    spec = {"height": 460,
            "width": 1300,
            "title": f"Temperature for {device.name if device else 'all devices'} {render_date_title()} {' '.rjust(random.randint(1,10))}",
            "view": {"stroke": "transparent"},
            "transform": [{"window": [{"field": "temperature",
                                       "op": "mean",
                                       "as": "rolling_mean"}],
                           "frame": [-4, 4],
                           "groupby": ["device_id"]}],
            "layer": [{"layer": [{"mark": {"type": "line", "tooltip": True, "point": False, "interpolate": "monotone", "size": 3},
                                  "encoding": {"color": {"field": "device_name", "type": "nominal", "title": "", "scale": {"domain": device_names, "range": colours}}},
                                  "params": [{"name": "zoom_x", "bind": "scales",
                                              "select": {"type": "interval", "encodings": ["x"]}}]}],
                       "encoding": {"y": {"field": ["temperature", "rolling_mean"][interpolate], "type": "quantitative", "stack": None,
                                          "scale": {"zero": False, "domainMax": 25}, "axis": {"grid": True, "title": "Temperature"}},
                                    "tooltip": [{"field": "date", "title": "time",  "type": "temporal", "timeUnit": "hoursminutes"},
                                                {"field": "temperature"},
                                                {"field": "heat_target"},
                                                {"field": "heating_demand"},
                                                {"field": "heating_demand_percentage"},
                                                {"field": "device_name"}]}},
                      {"mark": {"type": "area", "interpolate": "step-after", "opacity": 0.7, "tooltip": True},
                       "encoding": {"y": {"field": "heating_relay", "type": "quantitative",
                                    "scale": {"zero": False, "domain": [0, 20]}, "axis": None},
                                    "tooltip": [{"field": "heating_start", "type": "temporal", "timeUnit": "hoursminutes", "title": "start"},
                                                {"field": "heating_length_str", "title": "length"}],
                                    "color": {"value": "red"}}}],
            "encoding": {"x": {"field": "date", "type": "temporal", "timeUnit": "hoursminutes", "title": "time",
                               "axis": {"ticks": True, "grid": True, "gridOpacity": 0.35}},
                         "detail": {"field": "device_name", "type": "nominal", "title": ""}},
            "resolve": {"scale": {"y": "independent"}}}

    if device:
        spec["layer"][0]["layer"][0]["encoding"].update({"opacity": {"condition": {"test": f"datum['device_id'] == '{device.id}'", "value": 1}, "value": 0.4},
                                                         "size": {"condition": {"test": f"datum['device_id'] == '{device.id}'", "value": 3}, "value": 1.2}, })

        spec["layer"].append({"mark": {"type": "area", "interpolate": "step-after", "transparent": False, "opacity": 0.2, "tooltip": False, "color": "green"},
                              "encoding": {"y": {"field": "heating_demand", "type": "quantitative", "scale": {"zero": False, "domain": [0, 30]}, "axis": None}}})

        if demand_overlay:
            spec["layer"].append({"mark": {"type": "area", "interpolate": "step-after", "transparent": False, "opacity": 0.15, "tooltip": False, "color": "darkred"},
                                  "encoding": {"y": {"field": "heating_relay", "type": "quantitative", "scale": {"zero": False, "domain": [0, 1]}, "axis": None}}})

        spec["layer"][0]["layer"].append({"mark": {"type": "area", "tooltip": True, "point": False, "interpolate": "step-after", "opacity": 0.05,
                                                   "line": {"color": "#468B52", "strokeDash": [8, 8], "strokeWidth": 2, "opacity": 0.75},
                                                   "color": {"x1": 1, "y1": 1, "x2": 1, "y2": 0, "gradient": "linear",
                                                             "stops": [{"offset": 0, "color": "white"}, {"offset": 1, "color": "darkred"}]}},
                                          "encoding": {"y": {"field": "heat_target", "type": "quantitative", "scale": {"zero": False}}}})

        if device.is_trv and display_demand_percentage:
            spec["layer"].append({"mark": {"type": "line", "strokeWidth": 0.8, "opacity": 0.4},
                                  "encoding": {"y": {"field": "heating_demand_percentage", "type": "quantitative",  "interpolate": "linear",
                                                     "scale": {"domain": [0, 100]}, "axis": {"title": "Heating Demand %"}},
                                               "color": {"field": "device_name"}}})
    elif demand_overlay:
        spec["layer"][0]["layer"].insert(0, {"mark": {"type": "trail", "filled": True},
                                             "encoding": {"color": {"field": "device_name"},
                                                          "size": {"condition": {"test": "datum['trv_heating_relay']", "value": 14}, "value": 1}}})
    return spec


@st.cache_resource
def get_boto_session():
    return BotoSession()


async def main():
    hive.BOTO_SESSION = get_boto_session()
    async with hive.init_session():
        await app()

asyncio.run(main())
