import asyncio
from decimal import Decimal
import random
from cognito import BotoSession
from utils import DATE_TYPE_DAY, _this, make_utc, render_date_title, write_date_picker
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

async def app():
    st.markdown(CSS, unsafe_allow_html=True)

    *date_cols, _, col1, col2, _, col3 = st.columns((27.5, 75, 22.5, 20, 60, 80, 40, 300))

    device_data, data_is_expired = None, False
    try:
        device_data = await hive.get_device_data(False)
    except:
        pass

    date_from, date_to = write_date_picker(date_cols)
    data_is_expired = device_data is not None and (date_from == (today := _this(DATE_TYPE_DAY))) and device_data.date.max() < make_utc(today)
    refresh = col1.button(("❗ " if device_data is None else "⚠️ " if data_is_expired else "") + "Refresh")
    interpolate = col2.checkbox("Interpolate", value=True)
    display_demand_percentage = col2.checkbox("Demand %", value=False)
    demand_overlay = col2.checkbox("Demand Overlay", value=False)

    if refresh or (st.session_state.get("auth_state") == "complete"):
        if (st.session_state.get("auth_state") == "complete") or check_auth():
            st.session_state["auth_state"] = None
            with st.spinner('Loading data'):
                device_data = await hive.get_device_data(True)
        else:
            st.session_state["auth_state"] = "pending"

    if st.session_state.get("auth_state") == "pending":
        authenticate()

    if device_data is None:
        st.stop()

    devices = await hive.get_devices()
    device_id = st.radio("Device", [""] + sorted((d.id for d in devices.values() if d.is_trv or d.is_heater), key=lambda x: devices[x].name),
                         format_func=lambda x: devices[x].name if x else "All devices", key="device_id",
                         label_visibility="collapsed", horizontal=True)
    device = devices[device_id] if device_id else None
    is_daily_chart = st.session_state["date_type"] == DATE_TYPE_DAY
    device_data = prepare_data(devices, device_data, date_from, date_to, device_id, not is_daily_chart)
    spec = get_daily_chart_spec(device, interpolate, display_demand_percentage, demand_overlay) if is_daily_chart else get_agg_chart_spec(device)
    st.vega_lite_chart(device_data, spec, use_container_width=False)

    with st.expander("Dataframe"):
        st.dataframe(device_data)


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

    tokens = hive._load_cached_tokens()
    credentials = hive._load_credentials()

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
    t = hive._load_cached_tokens()
    if t.id_token is None:
        return False
    try:
        hive.authenticate(None)
        return True
    except:
        return False


def prepare_data(devices: dict, device_data: pd.DataFrame, date_from: date, date_to: date, device_id: str | None, aggregate: bool) -> pd.DataFrame:
    date_from, date_to = pd.to_datetime([date_from, date_to]).tz_localize('Europe/London')
    print(f"{date_from}({date_from.timestamp()}) / {date_to}({date_to.timestamp()})")

    device_data = (device_data.assign(date=lambda x: x.date.dt.tz_convert('Europe/London'))
                   .query("@date_from <= date < @date_to"))

    heating_source = device_id or hive._get_heater_id(devices)
    device_data = device_data.assign(trv_heating_relay=lambda x: x.heating_relay & ~x.is_heater,
                                     heating_relay=lambda x: x.heating_relay.where(x.device_id == heating_source, None))

    if device_id:
        device_specific_columns = ["heat_target", "heating_demand", "heating_demand_percentage", "heating_relay"]
        device_data.loc[device_data["device_id"] != device_id, device_specific_columns] = None

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
    return device_data.drop(columns=["is_heater", "heating_end"])



def get_agg_chart_spec(device_name: str | None) -> dict:
    spec = {"height": 460,
            "width": 1300,
            "title": f"Temperature for {device_name or 'all devices'} {render_date_title()} {' '.rjust(random.randint(1,10))}",
            "view": {"stroke": "transparent"},
            "layer": [
                # {"layer": [{"mark": {"type": "line", "tooltip": True, "point": False, "interpolate": "monotone", "strokeWidth": 3}},
                #            {"mark": {"type": "line", "tooltip": True, "point": False, "interpolate": "monotone", "strokeWidth": 3},
                #             "encoding": {"y": {"field": "high", "type": "quantitative", "stack": None,
                #                                "scale": {"zero": False}, "axis": {"grid": True, "title": "Max Temperature"}}}}],
                #  "encoding": {"y": {"field": "low", "type": "quantitative", "stack": None,
                #                     "scale": {"zero": False}, "axis": {"grid": True, "title": "Min Temperature"}},
                {"mark": {"type": "area", "tooltip": True, "point": False, "interpolate": "monotone", "opacity": 1},
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
            "encoding": {"x": {"field": "date", "type": "temporal", "title": "time",
                               "axis": {"ticks": True, "grid": True, "gridOpacity": 0.35}},
                         "color": {"field": "device_name", "type": "nominal", "title": ""}},
            "resolve": {"scale": {"y": "independent"}}}

    if device_name:
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


def get_daily_chart_spec(device: hive.Device | None, interpolate: bool, display_demand_percentage: bool, demand_overlay: bool) -> dict:
    spec = {"height": 460,
            "width": 1300,
            "title": f"Temperature for {device.name if device else 'all devices'} {render_date_title()} {' '.rjust(random.randint(1,10))}",
            "view": {"stroke": "transparent"},
            "transform": [{"window": [{"field": "temperature",
                                       "op": "mean",
                                       "as": "rolling_mean"}],
                           "frame": [-7, 7],
                           "groupby": ["device_id"]}],
            "layer": [{"layer": [{"mark": {"type": "line", "tooltip": True, "point": False, "interpolate": "monotone", "size": 3},
                                  "encoding": {"color": {"field": "device_name", "type": "nominal", "title": ""}},
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
        spec["layer"][0]["layer"].insert(0, {"mark": {"type": "point", "filled": True, "size": 144,
                                                      "shape": "M0,.5L.6,.8L.5,.1L1,-.3L.3,-.4L0,-1L-.3,-.4L-1,-.3L-.5,.1L-.6,.8L0,.5Z"},
                                             "encoding": {"color": {"value": "darkred"},
                                                          "opacity": {"condition": {"test": "datum['trv_heating_relay']", "value": 1}, "value": 0}}})
    return spec

@st.cache_resource
def get_boto_session():
    return BotoSession()
  

async def main():
    hive.BOTO_SESSION = get_boto_session()
    async with hive.init_session():
        await app()

asyncio.run(main())
