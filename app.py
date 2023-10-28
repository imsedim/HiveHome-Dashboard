import random
from utils import DATE_TYPE_DAY, _this, make_utc, render_date_title, write_date_picker
from datetime import date
import streamlit as st
import hive
import pandas as pd

def app():
    st.set_page_config(page_title="Heating Dashboard",
                       layout="wide",
                       initial_sidebar_state="collapsed")

    *date_cols, _, col1, col2 = st.columns((27.5, 75, 22.5, 20, 60, 420))

    device_data, data_is_expired = None, False
    try:
        with st.spinner('Loading data'):
            device_data = hive.get_device_data(False)
    except:
        pass

    date_from, date_to = write_date_picker(date_cols)
    data_is_expired = device_data is not None and (date_from == (today := _this(DATE_TYPE_DAY))) and device_data.date.max() < make_utc(today)
    refresh = col1.button(("❗ " if device_data is None else "⚠️ " if data_is_expired else "") + "Refresh")
    interpolate = col2.checkbox("Interpolate", value=True)

    if refresh or (st.session_state.get("auth_state") == "complete"):
        st.session_state["auth_state"] = None
        if check_auth():
            with st.spinner('Loading data'):
                device_data = hive.get_device_data(True)
        else:
            st.session_state["auth_state"] = "pending"

    if st.session_state.get("auth_state") == "pending":
        authenticate()

    if device_data is None:
        st.stop()

    devices = hive.get_devices()
    device_id = st.selectbox("Device", [""] + [d.id for d in devices.values() if d.is_trv or d.is_heater],
                             format_func=lambda x: devices[x].name if x else "", key="device_id")

    is_daily_chart = st.session_state["date_type"] == DATE_TYPE_DAY
    device_data = prepare_data(device_data, date_from, date_to, device_id, not is_daily_chart)
    spec = get_daily_chart_spec(device_id, interpolate) if is_daily_chart else get_agg_chart_spec(device_id)
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


def prepare_data(device_data: pd.DataFrame, date_from: date, date_to: date, device_id: str | None, aggregate: bool):
    devices = hive.get_devices()
    date_from, date_to = pd.to_datetime([date_from, date_to]).tz_localize('Europe/London')
    device_data = device_data.assign(date=lambda x: x.date.dt.tz_convert('Europe/London'))
    print(f"{date_from}({date_from.timestamp()}) / {date_to}({date_to.timestamp()})")
    device_data = device_data.query("@date_from <= date < @date_to")

    if device_id:
        heater_id = next(d.id for d in devices.values() if d.is_heater)
        if devices[device_id].is_trv:
            device_data = (pd.merge(device_data.query("device_id == @device_id").drop(columns=["heating_relay"]),
                                    device_data.query("device_id == @heater_id").loc[:, ["date", "heating_relay"]], on="date")
                           .assign(heating_relay=lambda x: x.heating_relay & x.heating_demand.fillna(False)))
            hive._add_heating_stats(device_data)
        else:
            device_data = device_data.query("device_id == @device_id")

    if aggregate:
        device_data = (device_data.assign(date=lambda x: x.date.dt.floor('d'))
                       .groupby(["device_id", "device_name", "date"])
                       .agg(low=('temperature', 'min'), high=('temperature', 'max'),
                            open=('temperature', 'first'), close=('temperature', 'last'))
                       .assign(heating_length=(device_data.drop_duplicates(subset=["device_id", "heating_start"])
                                               .assign(date=lambda x: x.heating_start.dt.floor('d'))
                                               .groupby(["device_id", "device_name", "date"])
                                               .heating_length.sum()))
                       .assign(heating_length=lambda x: x.heating_length.fillna(pd.Timedelta(0)))
                       .reset_index())

    heating_minutes = ((heating_seconds := device_data.heating_length.dt.total_seconds().dropna().astype(int)) % 3600 // 60).astype(str) + "m"
    heating_length_str = heating_minutes.where(heating_seconds < 3600,
                                               (heating_seconds // 3600).astype(str) + "h " + heating_minutes)
    return device_data.assign(heating_length=heating_seconds/60,
                              heating_length_str=lambda x: heating_length_str.where(x.heating_relay, "") if "heating_relay" in x else heating_length_str)


def get_agg_chart_spec(device_id: str | None) -> dict:
    devices = hive.get_devices()
    spec = {"height": 460,
            "width": 1420,
            "title": f"Temperature for {devices[device_id].name if device_id else 'all devices'} {render_date_title()} {' '.rjust(random.randint(1,10))}",
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

    if device_id:
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


def get_daily_chart_spec(device_id: str | None, interpolate: bool) -> dict:
    devices = hive.get_devices()

    spec = {"height": 460,
            "width": 1420,
            "title": f"Temperature for {devices[device_id].name if device_id else 'all devices'} {render_date_title()} {' '.rjust(random.randint(1,10))}",
            "view": {"stroke": "transparent"},
            "transform": [{"window": [{"field": "temperature",
                                       "op": "mean",
                                       "as": "rolling_mean"}],
                           "frame": [-7, 7],
                           "groupby": ["device_id"]}],
            "layer": [{"layer": [{"mark": {"type": "line", "tooltip": True, "point": False, "interpolate": "monotone", "strokeWidth": 3}}],
                       "encoding": {"y": {"field": ["temperature", "rolling_mean"][interpolate], "type": "quantitative", "stack": None,
                                          "scale": {"zero": False}, "axis": {"grid": True, "title": "Temperature"}},
                                    "tooltip": [{"field": "date", "title": "time",  "type": "temporal", "timeUnit": "hoursminutes"},
                                                {"field": "temperature"},
                                                {"field": "heat_target"},
                                                {"field": "heating_demand"},
                                                {"field": "device_name"}]}},
                      #    "params": [{"name": "zoom_x",
                      #                "select": {"type": "interval", "encodings": ["x"]},
                      #                "bind": "scales"}],
                      {"mark": {"type": "area", "interpolate": "step-after", "transparent": True, "opacity": 0.7, "tooltip": True},
                       "encoding": {"y": {"field": "heating_relay", "type": "quantitative",
                                          "scale": {"zero": False, "domain": [0, 20]}, "axis": None},
                                    "tooltip": [{"field": "heating_start", "type": "temporal", "timeUnit": "hoursminutes", "title": "start"},
                                                {"field": "heating_length_str", "title": "length"}]
                                    }}],
            "encoding": {"x": {"field": "date", "type": "temporal", "timeUnit": "hoursminutes", "title": "time",
                               "axis": {"ticks": True, "grid": True, "gridOpacity": 0.35}},
                         "color": {"field": "device_name", "type": "nominal", "title": ""}},
            "resolve": {"scale": {"y": "independent"}}}

    if device_id:
        spec["encoding"].pop("color")
        spec["layer"][1]["encoding"]["color"] = {"value": "red"}

        spec["layer"].append({"mark": {"type": "area", "interpolate": "step-after", "transparent": False, "opacity": 0.2, "tooltip": False, "color": "green"},
                              "encoding": {"y": {"field": "heating_demand", "type": "quantitative", "scale": {"zero": False, "domain": [0, 40]}, "axis": None}}})

        spec["layer"][0]["layer"].append({"mark": {"type": "area", "tooltip": True, "point": False, "interpolate": "step-after", "opacity": 0.05,
                                                   "line": {"color": "#468B52", "strokeDash": [8, 8], "strokeWidth": 2, "opacity": 0.75},
                                                   "color": {"x1": 1, "y1": 1, "x2": 1, "y2": 0, "gradient": "linear",
                                                             "stops": [{"offset": 0, "color": "white"}, {"offset": 1, "color": "darkred"}]}},
                                          "encoding": {"y": {"field": "heat_target", "type": "quantitative", "scale": {"zero": False}}}})
    return spec


app()
