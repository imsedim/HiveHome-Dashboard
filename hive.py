import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
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
import humanize

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

START_OF_HISTORY = datetime(2023, 3, 1, 0, 0, 0, tzinfo=UTC_TZ)


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
    """Initialize global aiohttp session. Mutates SESSION global."""
    
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
    """Begin Cognito auth flow. Returns partial result with mfa_tokens if SMS MFA required."""

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
    """Verify and refresh tokens if expired. May update id_token and access_token."""

    u = _create_cognito(id_token=tokens.id_token, access_token=tokens.access_token, refresh_token=tokens.refresh_token,
                        device_key=tokens.device_key, device_group_key=tokens.device_group_key)
    u.check_token()
    return AuthenticationResult(id_token=u.id_token, access_token=u.access_token, refresh_token=u.refresh_token,
                                device_key=tokens.device_key, device_group_key=tokens.device_group_key)


@timeit
def complete_authentication(username: str, mfa_code: str, mfa_tokens: Any) -> AuthenticationResult:
    """Complete MFA flow. Registers device with Cognito for future remembered sessions."""

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
    """Authenticate or refresh session. Writes tokens to cache file; deletes cache on failure."""

    tokens = load_cached_tokens()

    try:
        tokens = (verify_authentication(tokens) if tokens.id_token else
                  start_authentication(credentials.username, credentials.password) if not tokens.mfa_tokens else
                  complete_authentication(tokens.mfa_username, credentials.mfa_code, tokens.mfa_tokens))

        CACHED_TOKENS_FILE.write_text(tokens.model_dump_json())
    except:
        if CACHED_TOKENS_FILE.exists():
            CACHED_TOKENS_FILE.unlink()
        raise

    return tokens


def get_auth_token() -> str:
    tokens = load_cached_tokens()
    assert tokens.id_token
    return tokens.id_token


async def get_measurements(device: str, start: datetime, end: datetime) -> dict:
    params = {"from": int(start.timestamp()),
              "to": int(end.timestamp())}
    return await call_api_async(f"https://measurements.tsdb.prod.bgchprod.info/device/{device}", params=params)


async def get_devices() -> dict[str, Device]:
    """Fetch device list. Caches to devices.json permanently (delete file to refresh)."""

    file = DATA_DIR / "devices.json"

    if file.exists():
        data = json.loads(file.read_text())
    else:
        data = await call_api_async("https://beekeeper-uk.hivehome.com/1.0/devices")
        file.write_text(json.dumps(data, indent=2))

    return {x["id"]: Device(id=x["id"], type=x["type"], name=x["state"]["name"])
            for x in data}


async def fetch_device_data(device: str, start_time: datetime, end_time: datetime) -> tuple[str, dict] | None:
    """Fetch measurements for device. Returns None silently on 404 (device has no data)."""

    try:
        data = await get_measurements(device, start_time, end_time)
        return (device, data)
    except ClientResponseError as e:
        if e.status != 404:
            raise

def add_heating_stats(df: pd.DataFrame, heater_id: str) -> pd.DataFrame:
    """
    Calculate heating statistics for each row.

    heating_minutes is forward-looking: value at timestamp T represents
    heating from T until the next timestamp.
    """
    df = df.assign(is_heater=lambda x: x.device_id == heater_id,
                    heating_relay=lambda x: x.heating_relay.where(x.is_heater, x.heating_relay & x.heating_demand.fillna(False)),
                    next_date=lambda x: x.groupby("device_id").date.shift(-1).fillna(x.date),
                    heating_minutes=lambda x: (x.next_date - x.date).dt.total_seconds().div(60).where(x.heating_relay, None))

    heating_groups = (df.assign(heating_relay=lambda x: x.heating_relay.astype(float))
                      .pipe(lambda x: x.groupby(["device_id",
                                                 x.date.dt.tz_convert('Europe/London').dt.floor('D'),
                                                 (x.heating_relay != x.groupby("device_id").heating_relay.shift()).cumsum()])))
    return (df.assign(heating_start=heating_groups.date.transform('first').where(df.heating_relay, None),
                     heating_end=heating_groups.next_date.transform('last').where(df.heating_relay, None),
                     heating_length=heating_groups.heating_minutes.transform('sum').where(df.heating_relay, None))
            .drop(columns=["next_date"]))


def resample_heating_data(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Resample heating data to specified frequency."""
    if freq is None:
        return df

    aggs = ({"heating_minutes": ("heating_minutes", "sum"),
             "heating_relay": ("heating_relay", "max"),
             "heating_start": ("heating_start", "first"),
             "heating_end": ("heating_end", "first"),
             "heating_length": ("heating_length", "first"),
             "t_low": (["temperature", "t_low"]["t_low" in df], "min"),
             "t_high": (["temperature", "t_high"]["t_high" in df], "max")})
    copy_columns = ["heat_target", "heating_demand", "is_heater",
                    "temperature", "heating_demand_percentage"]

    
    return (df.groupby(["device_id", pd.Grouper(key="date", freq=freq, label="left", closed="left")], as_index=False)
            .agg(**aggs)
            .merge(df[["device_id", "date"] + copy_columns], on=["device_id", "date"])
            .assign(temperature=lambda x: x.temperature.round(2),
                    heating_demand_percentage=lambda x: x.heating_demand_percentage.round()))

def _create_device_dataframe(devices: dict[str, Device], data: dict, resample_freq: str = "5min") -> pd.DataFrame:
    """
    Transform raw API measurements into processed DataFrame.
    Applies time grid alignment, gap interpolation, and heating stats calculation.
    Uses hardcoded device_mapping for legacy device ID remapping.
    """

    def convert_measures_to_df() -> pd.DataFrame:
        def create_measure_df(values: dict, measure: str, device_id: str) -> pd.DataFrame:
            return (pd.DataFrame(values.items(), columns=["date", "value"])
                    .assign(measure=measure, device_id=device_id,
                            date=lambda x: pd.to_datetime(x.date.astype(int), unit='s', utc=True).dt.floor('min')))

        heater_id = get_heater_id(devices)
        heating_df = (create_measure_df(data[heater_id]["heating_relay"], None, None)
                      .rename(columns={"value": "heating_relay"})
                      .loc[:, ["date", "heating_relay"]]
                      .set_index("date")
                      .sort_index()
                      .resample("1min")
                      .ffill()
                      .reset_index())

        device_mapping = {"65e0f239-21d7-4bac-a96f-96bc3520b682": "c31bc5f9-5962-4636-ba0f-c43406c2d029"}
        measures_df = (pd.concat((pd.concat(create_measure_df(mv, m, device_id)
                                            for m, mv in device_data.items()
                                            if mv and m in MEASURE_NAMES)
                                  for _d, device_data in data.items()
                                  for device_id in [device_mapping.get(_d, _d)]), ignore_index=True)
                       .pivot_table(index=["date", "device_id"], columns="measure", values="value")
                       .reset_index())

        return (measures_df
                .drop(columns="heating_relay")
                .merge(heating_df, on="date", how="outer")
                .assign(heating_demand=lambda x: x.heating_demand if "heating_demand" in x else None,
                        heat_target=lambda x: x.heat_target if "heat_target" in x else None)
                .assign(heating_relay=lambda x: x.heating_relay.astype('boolean'),
                        heating_demand=lambda x: x.heating_demand.astype('boolean')))

    def add_time_grid(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
        date_index = (df.assign(day=lambda x: df.date.dt.normalize())
                      .groupby(["device_id", "day"])
                      .agg(start=('date', 'min'), end=('date', 'max'))
                      .assign(start=lambda x: x.start.where(x.start - x.start.dt.normalize() > pd.Timedelta(minutes=30), x.start.dt.normalize()),
                              end=lambda x: x.end.where(x.end - x.start.dt.normalize() < pd.Timedelta(hours=23, minutes=30), (x.end.dt.normalize() + pd.Timedelta(hours=23, minutes=59))))
                      .stack()
                      .reset_index(level=2, drop=True)
                      .to_frame("date")
                      .reset_index()
                      .set_index("date")
                      .groupby(["device_id", "day"], group_keys=False)
                      .resample(freq)
                      .nearest()
                      .drop(columns="day")
                      .reset_index())
        return df.merge(date_index, on=["device_id", "date"], how="outer")

    def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("date")
        interpolate_measures = ["temperature", "heating_demand_percentage"]
        df[interpolate_measures] = df.groupby("device_id")[interpolate_measures].transform(lambda x: x.interpolate(method="quadratic").round(5))

        # bug: potentially back/forward filling heating_relay for intra-day outage gaps
        df[MEASURE_NAMES] = df.groupby(["device_id", df.index.normalize()])[MEASURE_NAMES].ffill()
        df[MEASURE_NAMES] = df.groupby(["device_id", (df.index - pd.Timedelta(hours=1)).normalize()])[MEASURE_NAMES].transform(lambda x: x.ffill().bfill())

        return df.reset_index()

    df = convert_measures_to_df()
    df = add_time_grid(df)
    df = df.sort_values(["device_id", "date"])
    df = fill_gaps(df)
    df = add_heating_stats(df, get_heater_id(devices))
    df = resample_heating_data(df, resample_freq)

    device_names = {_id: d.name for _id, d in devices.items()}
    return df.assign(device_name=lambda x: x.device_id.map(device_names))


async def get_product_state() -> dict[str, dict]:
    """Fetch live state of devices, keyed by device ID (not product ID)."""

    products = await fetch_products()
    result = {x["props"]["trvs"][0]: x for x in products if x["type"] == "trvcontrol"}
    heating = next(x for x in products if x["type"] == "heating")
    result[heating["props"]["zone"]] = heating
    return result


async def fetch_products():
    """Fetch products from API. Always writes to products.json (for debugging)."""
    products = await call_api_async("https://beekeeper.hivehome.com/1.0/products")
    Path("data/products.json").write_text(json.dumps(products, indent=2))
    return products


# async def get_current_device_data(devices: dict[str, Device]) -> pd.DataFrame:
#     product_state = await get_product_state()
#     return (pd.DataFrame(dict(date=pd.to_datetime(datetime.now(UTC_TZ), utc=True).floor('min'),
#                               device_id=k,
#                               device_name=devices[k].name,
#                               temperature=v["props"]["temperature"],
#                               heat_target=v["state"]["target"],
#                               is_heater=v["type"] == "heating",
#                               heating_relay=(v["props"]["working"] if v["type"] == "heating" else None))
#                          for k, v in product_state.items())
#             .assign(heating_relay=lambda x: x.heating_relay.astype('boolean'),
#                     is_heater=lambda x: x.is_heater.astype('boolean')))

async def get_current_device_data() -> dict[str, dict]:
    product_state = await get_product_state()
    # date=pd.to_datetime(datetime.now(UTC_TZ), utc=True).floor('min'),
    return {k: {"temperature": v["props"]["temperature"],
                "heat_target": v["state"]["target"],
                ["heating_demand", "heating_relay"][v["type"] == "heating"]: v["props"]["working"]}
            for k, v in product_state.items()}


def _load_cached_dataframe() -> pd.DataFrame:
    """Load cached DataFrame from pickle file. Returns empty DataFrame if no cache exists."""
    return pd.read_pickle(CACHED_DEVICE_DATA_FILE) if CACHED_DEVICE_DATA_FILE.exists() else pd.DataFrame()


def _save_cached_dataframe(df: pd.DataFrame) -> None:
    """Save DataFrame to pickle cache file."""
    df.to_pickle(CACHED_DEVICE_DATA_FILE)


def _save_raw_data(filename: str, data: dict) -> None:
    """Save raw API response to JSON file in data/raw/ directory."""
    (RAW_DEVICE_DATA_DIR / filename).write_text(json.dumps(data))


async def _fetch_all_device_data(devices: dict[str, Device], start: datetime, end: datetime) -> dict:
    """Fetch measurement data for all devices in parallel. Returns dict of device_id -> data."""
    async with asyncio.TaskGroup() as tg:
    #   current_task = tg.create_task(get_current_device_data())
        tasks = [tg.create_task(fetch_device_data(device_id, start, end))
                 for device_id, device in devices.items()
                 if device.is_heater or device.is_trv]
    return dict([t.result() for t in tasks if t.result()])


@timeit
async def get_device_data(refresh: bool = False) -> pd.DataFrame:
    """
    Main entry point for device data. Returns cached DataFrame or fetches fresh data.
    On refresh: fetches from API, archives raw JSON to data/raw/, caches processed pickle.
    Incrementally fetches only data newer than last full day in cache.
    """

    history_data = _load_cached_dataframe()
    if not refresh:
        return history_data if not history_data.empty else None

    devices = await get_devices()

    history_data = history_data if not history_data.empty else parse_raw_device_data(devices)

    # round down to the beginning of day to make sure heating stats are recalculated correctly for the whole day
    # skip 10 rows to ignore "current device state"
    cutoff_date = (history_data.iloc[-10]["date"]
                   .tz_convert(LOCAL_TZ)
                   .floor('D')
                   .tz_convert('UTC')
                   .to_pydatetime()
                   if not history_data.empty else None)

    # extra hour to fill gaps around midnight at the start of the batch
    batch_start = (cutoff_date - timedelta(hours=1)) if not history_data.empty else START_OF_HISTORY
    batch_end = datetime.now(UTC_TZ)
    with time_it("Requesting data"):
        print(f"Fetching data from {batch_start:%Y-%m-%d} to {batch_end:%Y-%m-%d %H:%M:%S}")
        device_data_dict = await _fetch_all_device_data(devices, batch_start, batch_end)

    last_reportings = {device_id: datetime.fromtimestamp(max(int(next(reversed(device_data[m]))) for m in MEASURE_NAMES if m in device_data), tz=UTC_TZ)
        for device_id, device_data in device_data_dict.items()}
    print("\n".join(f"  >> Last report for <{devices[device_id].name}> was {humanize.naturaltime(last_date)}"
                    for device_id, last_date in last_reportings.items()))

    # substract 1 second to represent the closed end of interval
    data_end = min(last_reportings.values())
    new_cutoff_date = data_end.astimezone(LOCAL_TZ).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(seconds=1)

    # if full day's worth of data is fetched since last.
    if history_data.empty or (new_cutoff_date > cutoff_date):
        print(f"New data cutoff: {new_cutoff_date: %Y-%m-%d}")
        _save_raw_data(f"{new_cutoff_date:%Y-%m-%d}.json", device_data_dict)

    device_data = _create_device_dataframe(devices, device_data_dict)
    if not history_data.empty:
        device_data = pd.concat([history_data.query("date < @cutoff_date"),
                                 device_data.query("date >= @cutoff_date")], ignore_index=True)

    _save_cached_dataframe(device_data)
    return device_data


async def call_api_async(url: str, method: str = "get", params: dict | None = None) -> dict:
    """Make authenticated API request. Auto-retries once on 401 after re-authenticating."""

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
    """Rebuild DataFrame from archived raw JSON files in data/raw/. Used for cache recovery."""

    print("Parsing raw device data ", end="")
    files = sorted(RAW_DEVICE_DATA_DIR.glob("*.json"), key=lambda x: x.name)

    dataframes, start_from, end_with = [], None, None
    for file in files:
        print(".", end="")
        df = _create_device_dataframe(devices, json.loads(file.read_text()))

        start_from = end_with
        end_with = [pd.to_datetime(file.stem).tz_localize(LOCAL_TZ) + pd.Timedelta(days=1), None][file == files[-1]]
        query = ["@start_from <= ", ""][start_from is None] + "date" + [" < @end_with", ""][end_with is None]

        df = df.query(query)
        dataframes.append(df)

    print()

    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()


def get_heater_id(devices: dict[str, Device]) -> str:
    return next(k for k, v in devices.items() if v.is_heater)
