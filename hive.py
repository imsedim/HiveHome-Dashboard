import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from typing import Any
import json
from aiohttp import ClientResponseError, ClientSession
from cognito import Cognito, MFAChallengeException
import pandas as pd
import pydantic
import boto3
from utils import time_it, timeit

SESSION: ClientSession = None
BOTO_SESSION: boto3.Session = None

LOCAL_TZ = ZoneInfo("Europe/London")
UTC_TZ = ZoneInfo("UTC")
DATA_DIR = Path("data")
MEASURE_NAMES = ["heat_target", "heating_relay", "temperature", "heating_demand", "heating_demand_percentage"]
RAW_DEVICE_DATA_DIR = DATA_DIR / "raw"
CACHED_TOKENS_FILE = DATA_DIR / "tokens.json"
CACHED_DEVICE_DATA_FILE = DATA_DIR / "device_data.pickle"

if not DATA_DIR.exists():
    DATA_DIR.mkdir()
    RAW_DEVICE_DATA_DIR.mkdir()

# DATA_OVERLAP_DELTA = timedelta(minutes=15)


class Credentials(pydantic.BaseModel):
    username: str | None = None
    password: str | None = None
    mfa_code: str | None = None


class AuthenticationResult(pydantic.BaseModel):
    id_token: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    mfa_tokens: dict | None = None
    mfa_username: str | None = None
    device_key: str | None = None
    device_group_key: str | None = None


class Device(pydantic.BaseModel):
    id: str
    name: str
    type: str

    @property
    def is_trv(self) -> bool:
        return self.type == "trv"

    @property
    def is_heater(self) -> bool:
        return self.type == "boilermodule"


def init_session() -> ClientSession:
    global SESSION
    SESSION = ClientSession(raise_for_status=True)
    return SESSION


def load_cached_tokens() -> AuthenticationResult:
    return (AuthenticationResult.model_validate_json(CACHED_TOKENS_FILE.read_text()) if CACHED_TOKENS_FILE.exists() else
            AuthenticationResult())


def load_credentials() -> Credentials:
    credentials_file = Path("data/credentials.json")
    return Credentials.model_validate_json(credentials_file.read_text()) if credentials_file.exists() else Credentials()


def _create_cognito(username: str | None = None, id_token: str | None = None, access_token: str | None = None,
                    refresh_token: str | None = None, device_key: str | None = None, device_group_key: str | None = None) -> Cognito:
    return Cognito(user_pool_id='eu-west-1_SamNfoWtf', client_id='3rl4i0ajrmtdm8sbre54p9dvd9',
                   username=username, id_token=id_token, access_token=access_token, refresh_token=refresh_token,
                   device_key=device_key, device_group_key=device_group_key, session=BOTO_SESSION)


@timeit
def start_authentication(username: str, password: str) -> AuthenticationResult:
    assert username
    assert password

    u = _create_cognito(username=username)

    try:
        u.authenticate(password)
    except MFAChallengeException as mfa_challenge:
        print(f" *** SMS MFA required")
        return AuthenticationResult(mfa_tokens=mfa_challenge.get_tokens(), mfa_username=u.mfa_username)
    else:
        return AuthenticationResult(id_token=u.id_token, access_token=u.access_token, refresh_token=u.refresh_token)


@timeit
def verify_authentication(tokens: AuthenticationResult) -> AuthenticationResult:
    u = _create_cognito(id_token=tokens.id_token, access_token=tokens.access_token, refresh_token=tokens.refresh_token,
                        device_key=tokens.device_key, device_group_key=tokens.device_group_key)
    u.check_token()
    return AuthenticationResult(id_token=u.id_token, access_token=u.access_token, refresh_token=u.refresh_token,
                                device_key=tokens.device_key, device_group_key=tokens.device_group_key)


@timeit
def complete_authentication(username: str, mfa_code: str, mfa_tokens: Any) -> AuthenticationResult:
    assert username
    assert mfa_code
    assert mfa_tokens

    u = _create_cognito(username=username)
    u.respond_to_sms_mfa_challenge(mfa_code, mfa_tokens)
    u.confirm_device()
    u.update_device_status(True)
    return AuthenticationResult(id_token=u.id_token, access_token=u.access_token, refresh_token=u.refresh_token,
                                device_key=u.device_key, device_group_key=u.device_group_key)


def authenticate(credentials: Credentials) -> AuthenticationResult:
    tokens = load_cached_tokens()

    try:
        tokens = (verify_authentication(tokens) if tokens.id_token else
                  start_authentication(credentials.username, credentials.password) if not tokens.mfa_tokens else
                  complete_authentication(tokens.mfa_username, credentials.mfa_code, tokens.mfa_tokens))

        CACHED_TOKENS_FILE.write_text(tokens.model_dump_json())
    except Exception:
        (CACHED_TOKENS_FILE.unlink() if CACHED_TOKENS_FILE.exists() else None)
        raise

    return tokens


def get_auth_token() -> str:
    tokens = load_cached_tokens()
    assert tokens.id_token
    return tokens.id_token


def md5_hash(input: str) -> str:
    return hashlib.md5(input.encode()).hexdigest()


async def get_measurements(device: str, start: datetime, end: datetime) -> tuple[str, dict | None]:
    params = {"from": int(start.timestamp()),
              "to": int(end.timestamp())}
    return await call_api_async(f"https://measurements.tsdb.prod.bgchprod.info/device/{device}", params=params)


async def get_devices() -> dict[str, Device]:
    file = DATA_DIR / "devices.json"

    if file.exists():
        data = json.loads(file.read_text())
    else:
        data = await call_api_async("https://beekeeper-uk.hivehome.com/1.0/devices")
        file.write_text(json.dumps(data, indent=2))

    return {x["id"]: Device(id=x["id"], type=x["type"], name=x["state"]["name"])
            for x in data}


async def fetch_device_data(device: str, start_time: datetime) -> tuple[str, dict]:
    try:
        data = await get_measurements(device, start_time, datetime.now(UTC_TZ))
        return (device, data)
    except ClientResponseError as e:
        if e.status != 404:
            raise


def merge_device_data(data: dict | None, new_data: dict) -> dict:
    if not data:
        return new_data

    max_time_entries = []
    for key, time_entries in new_data.items():
        if key not in ["id", "period"]:
            data.setdefault(key, {}).update(time_entries)
            if time_entries:
                max_time_entries.append(max(time_entries.keys()))

    data["period"]["to"] = max(max_time_entries, default=data["period"]["from"])

    return data


def _create_device_dataframe(devices: dict[str, Device], data: dict, resample_freq: str = "5min") -> pd.DataFrame:
    def convert_measures_to_df() -> pd.DataFrame:
        device_mapping = {"65e0f239-21d7-4bac-a96f-96bc3520b682": "c31bc5f9-5962-4636-ba0f-c43406c2d029"}
        return (pd.concat((pd.concat(pd.DataFrame(mv.items(), columns=["date", "value"]).assign(measure=m)
                                     for m, mv in device_data.items()
                                     if mv and m in MEASURE_NAMES)
                           .assign(date=lambda x: pd.to_datetime(x.date.astype(int), unit='s', utc=True).dt.floor('T'))
                           .assign(device_id=device_id,
                                   value=lambda x: x.value.where((x.measure != "heating_relay") | devices[device_id].is_heater, None))
                           for _d, device_data in data.items()
                           for device_id in [device_mapping.get(_d, _d)]), ignore_index=True)
                .pivot_table(index=["date", "device_id"], columns="measure", values="value")
                .reset_index()
                .assign(heating_relay=lambda x: x.heating_relay if "heating_relay" in x else None,
                        heating_demand=lambda x: x.heating_demand if "heating_demand" in x else None,
                        heat_target=lambda x: x.heat_target if "heat_target" in x else None)
                .assign(heating_relay=lambda x: x.heating_relay.astype('boolean'),
                        heating_demand=lambda x: x.heating_demand.astype('boolean')))

    def unify_date(df: pd.DataFrame) -> pd.DataFrame:
        date_index = set(df["date"].unique()) | set(pd.date_range(df["date"].min().round("5min"), df["date"].max(), freq="5min"))

        df_index = (pd.MultiIndex.from_product([df["device_id"].unique(), date_index], names=["device_id", "date"])
                    .to_frame(index=False))

        return df_index.merge(df, on=["device_id", "date"], how="left")

    def add_heating_stats(df: pd.DataFrame) -> pd.DataFrame:
        heater_map = {_id: d.is_heater for _id, d in devices.items()}
        df = (df.assign(is_heater=lambda x: x.device_id.map(heater_map).astype('boolean'))
                .sort_values("date"))
        df = (pd.merge(df.drop(columns=["heating_relay"]),
                       df.loc[df.is_heater, ["date", "heating_relay"]], on="date")
              .assign(heating_relay=lambda x: x.heating_relay.where(x.is_heater, x.heating_relay & x.heating_demand.fillna(False))))

        for _, device_group in df[["device_id"]].groupby("device_id"):
            heating_grouping = (df.loc[device_group.index]
                                .assign(heating_relay=lambda x: x.heating_relay.astype(bool))
                                .assign(next_date=lambda x: x.date.shift(-1).fillna(x.date))
                                .pipe(lambda x: x.groupby([x.date.dt.tz_convert('Europe/London').dt.floor('D'), (x.heating_relay != x.heating_relay.shift()).cumsum()])))
            df.loc[device_group.index, "heating_start"] = heating_grouping.date.transform('first').where(df.heating_relay, None)
            heating_end = heating_grouping.next_date.transform('last').where(df.heating_relay, None)
            df.loc[device_group.index, "heating_length"] = (heating_end - df["heating_start"]).dt.total_seconds()/60 + 1
        return df

    def fill_gaps(df) -> pd.DataFrame:
        df = df.sort_values(["device_id", "date"])
        df["temperature_int"] = df.groupby("device_id")["temperature"].transform(lambda x: x.interpolate())
        df[MEASURE_NAMES] = df.groupby("device_id")[MEASURE_NAMES].ffill().bfill()
        return df

    def resample(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
        return (df
                .groupby(["device_id", pd.Grouper(key="date", freq=freq, label="right", closed="right")], as_index=False)
                .agg({x: ['last', 'median'][x == "temperature"] for x in MEASURE_NAMES + ["is_heater", "heating_start", "heating_length"]}))

    df = convert_measures_to_df()
    df = unify_date(df)
    df = fill_gaps(df)
    df = add_heating_stats(df)
    df = resample(df, resample_freq) if resample_freq else df

    device_names = {_id: d.name for _id, d in devices.items()}
    return df.assign(device_name=lambda x: x.device_id.map(device_names))


async def get_product_state() -> dict[str, dict]:
    products = await fetch_products()
    result = {x["props"]["trvs"][0]: x for x in products if x["type"] == "trvcontrol"}
    heating = next(x for x in products if x["type"] == "heating")
    result[heating["props"]["zone"]] = heating
    return result


async def fetch_products():
    products = await call_api_async("https://beekeeper.hivehome.com/1.0/products")
    Path("data/products.json").write_text(json.dumps(products, indent=2))
    return products


async def get_current_device_data(devices: dict[str, Device]) -> pd.DataFrame:
    product_state = await get_product_state()
    return (pd.DataFrame(dict(date=pd.to_datetime(datetime.now(UTC_TZ), utc=True).floor('T'),
                              device_id=k,
                              device_name=devices[k].name,
                              temperature=v["props"]["temperature"],
                              heat_target=v["state"]["target"],
                              is_heater=v["type"] == "heating",
                              heating_relay=(v["props"]["working"] if v["type"] == "heating" else None))
                         for k, v in product_state.items())
            .assign(heating_relay=lambda x: x.heating_relay.astype('boolean'),
                    is_heater=lambda x: x.is_heater.astype('boolean')))


@timeit
async def get_device_data(refresh: bool = False) -> pd.DataFrame:
    history_data: pd.DataFrame = pd.read_pickle(CACHED_DEVICE_DATA_FILE) if CACHED_DEVICE_DATA_FILE.exists() else pd.DataFrame()
    print("get_device_data: history", history_data.shape)
    if not refresh:
        return history_data if not history_data.empty else None

    devices = await get_devices()

    if history_data.empty:
        cutoff_date = datetime(2023, 3, 1, 0, 0, 0, tzinfo=UTC_TZ)
        history_data = parse_raw_device_data(devices)

    if not history_data.empty:
        cutoff_date = history_data.iloc[-1]["date"].tz_convert(LOCAL_TZ).floor('D').tz_convert('UTC').to_pydatetime()
        history_data = history_data[history_data["date"] < cutoff_date]

    with time_it("Requesting data"):
        print(f"Fetching data since {cutoff_date}")
        async with asyncio.TaskGroup() as tg:
            # current_task = tg.create_task(get_current_device_data(devices))
            devices_tasks = [tg.create_task(fetch_device_data(device_id, cutoff_date))
                             for device_id, device in devices.items()
                             if device.is_heater or device.is_trv]

    device_data_dict = dict([t.result() for t in devices_tasks if t.result()])
    suffix = ["daily", "catchup"][cutoff_date < datetime.now(LOCAL_TZ).replace(hour=0, minute=0, second=0, microsecond=0)]
    (RAW_DEVICE_DATA_DIR / f"{datetime.today():%Y-%m-%d} {suffix}.json").write_text(json.dumps(device_data_dict))

    device_data = _create_device_dataframe(devices, device_data_dict)
    # device_data = pd.concat([device_data, current_task.result()], ignore_index=True)

    if not history_data.empty:
        device_data = pd.concat([history_data, device_data], ignore_index=True)

    device_data.to_pickle(CACHED_DEVICE_DATA_FILE)

# todo:
# in app.py:
#   resample to lower date resolution AFTER heating stats (median)

    return device_data


async def call_api_async(url: str, method: str = "get", params: dict | None = None) -> dict:
    async def _request():
        session = SESSION or init_session()
        async with session.request(method, url, params=params, headers={"Authorization": get_auth_token()}) as r:
            return await r.json()

    with time_it(f"> {method.upper()} {urlparse(url).path}"):
        try:
            return await _request()
        except ClientResponseError as e:
            if e.status == 401:
                authenticate(None)
                return await _request()
            raise


def parse_raw_device_data(devices: dict[str, Device]) -> pd.DataFrame:
    files = sorted(RAW_DEVICE_DATA_DIR.glob("*.json"), key=lambda x: x.name)
    dataframes = [_create_device_dataframe(devices, json.loads(x.read_text()))
                  for x in files]
    for df, next_df in zip(dataframes, dataframes[1:]):
        df.drop(df[df["date"] >= next_df["date"].min()].index, inplace=True)

    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()


def get_heater_id(devices: dict[str, Device]) -> str:
    return next(k for k, v in devices.items() if v.is_heater)
