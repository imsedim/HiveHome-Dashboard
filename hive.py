import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
from urllib.parse import urlparse
import time
from typing import Any
import json
from aiohttp import ClientResponseError, ClientSession
from cognito import Cognito, MFAChallengeException
import pandas as pd
import pydantic
import boto3

SESSION: ClientSession = None
BOTO_SESSION: boto3.Session = None

DATA_DIR = Path("data")

if not DATA_DIR.exists():
    DATA_DIR.mkdir()
    (DATA_DIR / "raw").mkdir()

CACHED_TOKENS_FILE = DATA_DIR / "tokens.json"
DATA_OVERLAP_DELTA = timedelta(minutes=15)


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


def _load_cached_tokens() -> AuthenticationResult:
    return (AuthenticationResult.model_validate_json(CACHED_TOKENS_FILE.read_text()) if CACHED_TOKENS_FILE.exists() else
            AuthenticationResult())


def _load_credentials() -> Credentials:
    credentials_file = Path("data/credentials.json")
    return Credentials.model_validate_json(credentials_file.read_text()) if credentials_file.exists() else Credentials()


def _create_cognito(username: str | None = None, id_token: str | None = None, access_token: str | None = None,
                    refresh_token: str | None = None, device_key: str | None = None, device_group_key: str | None = None) -> Cognito:
    return Cognito(user_pool_id='eu-west-1_SamNfoWtf', client_id='3rl4i0ajrmtdm8sbre54p9dvd9',
                   username=username, id_token=id_token, access_token=access_token, refresh_token=refresh_token,
                   device_key=device_key, device_group_key=device_group_key, session=BOTO_SESSION)


@contextmanager
def time_it(msg: str) -> Any:
    start = time.time()
    try:
        yield start
    finally:
        print(f"{msg} – {int((time.time() - start)*1000)}ms")


def timeit(func):
    def timed(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            print(f"{func.__name__} – {int((time.time() - start)*1000)}ms")
    return timed


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
    tokens = _load_cached_tokens()

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
    tokens = _load_cached_tokens()
    assert tokens.id_token
    return tokens.id_token


def md5_hash(input: str) -> str:
    return hashlib.md5(input.encode()).hexdigest()


async def get_measurements(device: str, start: datetime, end: datetime) -> tuple[str, dict | None]:
    # print(locals())
    params = {"from": int(start.timestamp()),
              "to": int(end.timestamp())}
    return await call_api_async(f"https://measurements.tsdb.prod.bgchprod.info/device/{device}", params=params)


async def get_devices() -> dict[str, Device]:
    file = Path(f"data/devices.json")

    if file.exists():
        data = json.loads(file.read_text())
    else:
        data = await call_api_async("https://beekeeper-uk.hivehome.com/1.0/devices")
        file.write_text(json.dumps(data, indent=2))

    return {x["id"]: Device(id=x["id"], type=x["type"], name=x["state"]["name"])
            for x in data}


async def fetch_device_data(device: str, return_full_history: bool) -> tuple[str, dict]:
    history_file = Path(f"data/raw/{device}.json")
    if history_file.exists():
        old_data = json.loads(history_file.read_text())
        start_time = pd.to_datetime(int(old_data["period"]["to"]), unit='s', utc=True).to_pydatetime() - DATA_OVERLAP_DELTA
    else:
        old_data, start_time = None, datetime(2023, 3, 1, 0, 0, 0)

    try:
        new_data = await get_measurements(device, start_time, datetime.now())
        history_data = merge_device_data(old_data, new_data)
        history_file.write_text(json.dumps(history_data))
        return (device, history_data if return_full_history else new_data)
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


def create_device_dataframe(devices: dict[str, Device], data: dict) -> pd.DataFrame:
    df = (pd.concat((pd.concat(pd.DataFrame(mv.items(), columns=["date", "value"]).assign(measure=m)
                               for m, mv in device_data.items()
                               if mv and m in ["heat_target", "heating_relay", "temperature", "heating_demand", "heating_demand_percentage"])
                       .assign(date=lambda x: pd.to_datetime(x.date.astype(int), unit='s', utc=True).dt.floor('T'))
                       .assign(device_id=device_id,
                               value=lambda x: x.value.where((x.measure != "heating_relay") | devices[device_id].is_heater, None))
                     for device_id, device_data in data.items()), ignore_index=True)
            .pivot_table(index=["date", "device_id"], columns="measure", values="value")
            .reset_index()
            .assign(heating_relay=lambda x: x.heating_relay if "heating_relay" in x else None,
                    heating_demand=lambda x: x.heating_demand if "heating_demand" in x else None)
            .assign(heating_relay=lambda x: x.heating_relay.astype('boolean'),
                    heating_demand=lambda x: x.heating_demand.astype('boolean'),
                    is_heater=lambda x: x.device_id.apply(lambda y: devices[y].is_heater).astype('boolean'))
            .pipe(lambda x: x.assign(heat_target=None) if "heat_target" not in x else x))

    device_names = {x.id: x.name for x in devices.values()}
    heater_id = _get_heater_id(devices)
    all_dates = pd.MultiIndex.from_product([df["device_id"].unique(), df["date"].unique()], names=["device_id", "date"]).to_frame(index=False)
    return (all_dates.merge(df, on=["device_id", "date"], how="left")
            .assign(device_name=lambda x: x.device_id.map(device_names),
                    is_heater=lambda x: x.device_id == heater_id)
            .sort_values(["date", "device_name"]))


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
    return (pd.DataFrame(dict(date=pd.to_datetime(datetime.utcnow(), utc=True).floor('T'),
                              device_id=k,
                              device_name=devices[k].name,
                              temperature=v["props"]["temperature"],
                              heat_target=v["state"]["target"],
                              is_heater=v["type"] == "heating",
                              heating_relay=(v["props"]["working"] if v["type"] == "heating" else None))
                         for k, v in product_state.items())
            .assign(heating_relay=lambda x: x.heating_relay.astype('boolean'),
                    is_heater=lambda x: x.is_heater.astype('boolean')))


async def get_device_data(refresh: bool = False) -> pd.DataFrame:
    try:
        history_data, fetch_partial_data = pd.read_pickle("data/device_data.pickle"), True
        if not refresh:
            return history_data
    except FileNotFoundError:
        history_data, fetch_partial_data = None, False

    with time_it("Loading device data"):
        devices = await get_devices()
        async with asyncio.TaskGroup() as tg:
            current_task = tg.create_task(get_current_device_data(devices))
            devices_tasks = [tg.create_task(fetch_device_data(device_id, not fetch_partial_data))
                            for device_id, device in devices.items()
                            if device.is_heater or device.is_trv]

    device_data_dict = dict([t.result() for t in devices_tasks if t.result()])
    if fetch_partial_data:
        Path("data/raw/last_device_data.json").write_text(json.dumps(device_data_dict))

    with time_it("Creating dataframe"):
        device_data = create_device_dataframe(devices, device_data_dict)
        if fetch_partial_data:
            split_date = max(device_data.groupby("device_id").date.first())
            device_data = pd.concat([history_data.query("date < @split_date"), device_data.query("date >= @split_date")], ignore_index=True)
        new_df = pd.concat([device_data, current_task.result()], ignore_index=True)

    with time_it("Processing device data"):
        ordinal_cols = ["heating_relay", "heating_demand", "heat_target", "heating_demand_percentage"]
        new_df[ordinal_cols] = new_df[ordinal_cols].fillna(new_df.groupby("device_id")[ordinal_cols].ffill())

        assert new_df["heating_relay"].dtype == "boolean"
        assert new_df["heating_demand"].dtype == "boolean"

        new_df["temperature"] = new_df.groupby("device_id", group_keys=False)["temperature"].apply(lambda x: x.interpolate())
        new_df["temperature"] = new_df["temperature"].fillna(new_df.groupby("device_id")["temperature"].bfill())

        new_df = (pd.merge(new_df.drop(columns=["heating_relay"]),
                        new_df.loc[new_df.is_heater, ["date", "heating_relay"]], on="date")
                    .assign(heating_relay=lambda x: x.heating_relay.where(x.is_heater, x.heating_relay & x.heating_demand.fillna(False))))

        _add_heating_stats(new_df)

    new_df.to_pickle("data/device_data.pickle")
    return new_df


async def call_api_async(url: str, method: str = "get", params: dict | None = None) -> dict:
    async def _request():
        assert SESSION
        async with SESSION.request(method, url, params=params, headers={"Authorization": get_auth_token()}) as r:
            return await r.json()

    with time_it(f"> {method.upper()} {urlparse(url).path}"):
        try:
            return await _request()
        except ClientResponseError as e:
            if e.status == 401:
                authenticate(None)
                return await _request()
            raise


def _add_heating_stats(df: pd.DataFrame):
    for _, device_group in df[["device_id"]].groupby("device_id"):
        heating_grouping = (df.loc[device_group.index]
                            .assign(heating_relay=lambda x: x.heating_relay.astype(bool))
                            .assign(next_date=lambda x: x.date.shift(-1).fillna(x.date))
                            .pipe(lambda x: x.groupby([x.date.dt.tz_convert('Europe/London').dt.floor('D'), (x.heating_relay != x.heating_relay.shift()).cumsum()])))
        df.loc[device_group.index, "heating_start"] = heating_grouping.date.transform('first').where(df.heating_relay, None)
        df.loc[device_group.index, "heating_end"] = heating_grouping.next_date.transform('last').where(df.heating_relay, None)
        df.loc[device_group.index, "heating_length"] = (df["heating_end"] - df["heating_start"]).dt.total_seconds()/60 + 1


def _get_heater_id(devices: dict[str, Device]) -> str:
    return next(k for k, v in devices.items() if v.is_heater)
