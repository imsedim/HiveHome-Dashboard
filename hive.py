from datetime import datetime, timedelta
import hashlib
from pathlib import Path
from urllib.parse import urlparse
import time
from typing import Any, Callable
import json
from pycognito import Cognito
from pycognito.exceptions import SMSMFAChallengeException
import pandas as pd
import requests
import pydantic


DATA_DIR = Path("data")

if not DATA_DIR.exists():
    DATA_DIR.mkdir()

CACHED_TOKENS_FILE = DATA_DIR / "tokens.json"



class Credentials(pydantic.BaseModel):
    username: str | None = None
    password: str | None = None
    mfa_code: str | None = None


class AuthenticationResult(pydantic.BaseModel):
    id_token: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    mfa_tokens: dict | None = None


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


def _load_cached_tokens() -> AuthenticationResult:
    return (AuthenticationResult.model_validate_json(CACHED_TOKENS_FILE.read_text()) if CACHED_TOKENS_FILE.exists() else
            AuthenticationResult())


def _load_credentials() -> Credentials:
    credentials_file = Path("data/credentials.json")
    return Credentials.model_validate_json(credentials_file.read_text()) if credentials_file.exists() else Credentials()


def _create_cognito(username: str | None = None, id_token: str | None = None, access_token: str | None = None,
                    refresh_token: str | None = None):
    return Cognito(user_pool_id='eu-west-1_SamNfoWtf', client_id='3rl4i0ajrmtdm8sbre54p9dvd9',
                   username=username, id_token=id_token, access_token=access_token, refresh_token=refresh_token)


def _timer() -> Callable:
    start = time.time()
    return lambda: int((time.time() - start)*1000)


def start_authentication(username: str, password: str) -> AuthenticationResult:
    assert username
    assert password

    t = _timer()
    u = _create_cognito(username=username)

    try:
        print("Invoke authenticate()")
        u.authenticate(password)
    except SMSMFAChallengeException as mfa_challenge:
        print(f" done [{t()}ms] *** SMS MFA required")
        return AuthenticationResult(mfa_tokens=mfa_challenge.get_tokens())
    else:
        return AuthenticationResult(id_token=u.id_token, access_token=u.access_token, refresh_token=u.refresh_token)


def verify_authentication(tokens: AuthenticationResult) -> AuthenticationResult:
    print("Invoke check_token()")
    u = _create_cognito(id_token=tokens.id_token, access_token=tokens.access_token, refresh_token=tokens.refresh_token)
    u.check_token()
    return AuthenticationResult(id_token=u.id_token, access_token=u.access_token, refresh_token=u.refresh_token)


def complete_authentication(username: str, mfa_code: str, mfa_tokens: Any) -> AuthenticationResult:
    assert username
    assert mfa_code
    assert mfa_tokens
    t = _timer()
    u = _create_cognito(username=username)
    print("Invoke respond_to_sms_mfa_challenge()")
    u.respond_to_sms_mfa_challenge(mfa_code, mfa_tokens)
    print(f"  done [{t()}ms]")
    return AuthenticationResult(id_token=u.id_token, access_token=u.access_token, refresh_token=u.refresh_token)


def authenticate(credentials: Credentials) -> AuthenticationResult:
    tokens = _load_cached_tokens()

    try:
        tokens = (verify_authentication(tokens) if tokens.id_token else
                  start_authentication(credentials.username, credentials.password) if not tokens.mfa_tokens else
                  complete_authentication(credentials.username, credentials.mfa_code, tokens.mfa_tokens))

        CACHED_TOKENS_FILE.write_text(tokens.model_dump_json())
    except Exception:
        CACHED_TOKENS_FILE.unlink()
        raise

    return tokens


def get_auth_token() -> str:
    tokens = _load_cached_tokens()
    assert tokens.id_token
    tokens = authenticate(None)
    return tokens.id_token


def md5_hash(input: str) -> str:
    return hashlib.md5(input.encode()).hexdigest()


def get_measurements(device: str, start: datetime, end: datetime) -> tuple[str, dict | None]:
    params = {"from": int(start.timestamp()),
              "to": int(end.timestamp())}
    return call_api(f"https://measurements.tsdb.prod.bgchprod.info/device/{device}", params=params)


def get_devices() -> dict[str, Device]:
    file = Path(f"data/devices.json")

    if file.exists():
        data = json.loads(file.read_text())
    else:
        data = call_api("https://beekeeper-uk.hivehome.com/1.0/devices")
        file.write_text(json.dumps(data, indent=2))

    return {x["id"]: Device(id=x["id"], type=x["type"], name=x["state"]["name"])
            for x in data}


def fetch_device_data(devices: list[str], start_date: datetime | None) -> dict:
    end_date = datetime.now()
    start_date = start_date or datetime(2023, 3, 1, 0, 0, 0)

    print(f"{start_date}({start_date.timestamp()}) / {end_date}({end_date.timestamp()})")
    device_data = {}
    for device in devices:
        try:
            device_data[device] = get_measurements(device, start_date, end_date)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                continue
            raise

    Path("data/raw/last_device_data.json").write_text(json.dumps(device_data))
    return device_data


def create_device_dataframe(devices: dict[str, Device], data: dict) -> pd.DataFrame:
    device_names = {x.id: x.name for x in devices.values()}
    df = (pd.concat((pd.concat(pd.DataFrame(device_data.get(m, {}).items(), columns=["date", "value"]).assign(measure=m)
                               for m in ["heat_target", "heating_relay", "temperature", "heating_demand"])
                       .assign(date=lambda x: pd.to_datetime(x.date.astype(int), unit='s', utc=True).dt.floor('T'))
                       .assign(device_id=device_id,
                               value=lambda x: x.value.where((x.measure != "heating_relay") | devices[device_id].is_heater, None))
                     for device_id, device_data in data.items()), ignore_index=True)
            .pivot_table(index=["date", "device_id"], columns="measure", values="value")
            .reset_index()
            .assign(heating_relay=lambda x: x.heating_relay.astype('boolean'),
                    heating_demand=lambda x: x.heating_demand.astype('boolean'))
            .pipe(lambda x: x.assign(heat_target=None) if "heat_target" not in x else x))

    all_dates = pd.MultiIndex.from_product([df["device_id"].unique(), df["date"].unique()], names=["device_id", "date"]).to_frame(index=False)
    return (all_dates.merge(df, on=["device_id", "date"], how="left")
            .assign(device_name=lambda x: x.device_id.map(device_names))
            .sort_values(["date", "device_name"]))


def get_product_state() -> dict[str, dict]:
    products = fetch_products()
    result = {x["props"]["trvs"][0]: x for x in products if x["type"] == "trvcontrol"}
    heating = next(x for x in products if x["type"] == "heating")
    result[heating["props"]["zone"]] = heating
    return result


def fetch_products():
    products = call_api("https://beekeeper.hivehome.com/1.0/products")
    Path("data/products.json").write_text(json.dumps(products, indent=2))
    return products


def get_current_device_data(devices: dict[str, Device]) -> pd.DataFrame:
    return (pd.DataFrame(dict(date=pd.to_datetime(datetime.utcnow(), utc=True).floor('T'),
                              device_id=k,
                              device_name=devices[k].name,
                              temperature=v["props"]["temperature"],
                              heat_target=v["state"]["target"],
                              heating_relay=(v["props"]["working"] if v["type"] == "heating" else None))
                         for k, v in get_product_state().items())
            .assign(heating_relay=lambda x: x.heating_relay.astype('boolean')))


def get_device_data(refresh: bool = False) -> pd.DataFrame:
    try:
        last_df = pd.read_pickle("data/device_data.pickle")
        if not refresh:
            return last_df
        batch_start_date = last_df.date.max().to_pydatetime() - timedelta(minutes=15)
    except FileNotFoundError:
        last_df, batch_start_date = None, None

    t = _timer()
    devices = get_devices()
    batch_df = pd.concat([create_device_dataframe(devices, fetch_device_data(devices.keys(), batch_start_date)),
                          get_current_device_data(devices)], ignore_index=True)
    print(f"Loading device data: {t()}ms")
    t = _timer()

    new_df = pd.concat([last_df.query("date < @batch_start_date"), batch_df], ignore_index=True) if last_df is not None else batch_df

    ordinal_cols = ["heating_relay", "heating_demand", "heat_target"]
    new_df[ordinal_cols] = new_df[ordinal_cols].fillna(new_df.groupby("device_id")[ordinal_cols].ffill())

    assert new_df["heating_relay"].dtype == "boolean"
    assert new_df["heating_demand"].dtype == "boolean"

    new_df["temperature"] = new_df.groupby("device_id", group_keys=False)["temperature"].apply(lambda x: x.interpolate())
    new_df["temperature"] = new_df["temperature"].fillna(new_df.groupby("device_id")["temperature"].bfill())

    _add_heating_stats(new_df, new_df["device_id"] == _get_heater_id(devices))

    print(f"Processing device data: {t()}ms")

    new_df.to_pickle("data/device_data.pickle")
    return new_df


def call_api(url: str, method: str = "get", params: dict | None = None) -> dict:
    print(f"Invoke {method.upper()} {urlparse(url).path}")
    t = _timer()
    r = requests.request(method, url, params=params, headers={"Authorization": get_auth_token()})
    r.raise_for_status()
    result = r.json()
    print(f"  done [{t()}ms]")
    return result


def _add_heating_stats(df: pd.DataFrame, source_mask: pd.Series = None):
    source_mask = source_mask if source_mask is not None else slice(None)
    heating_grouping = (df.loc[source_mask]
                        .assign(heating_relay=lambda x: x.heating_relay.astype(bool))
                        .assign(heating_segment=lambda x: (x.date.shift(-1) - x.date).where(x.heating_relay, None))
                        .pipe(lambda x: x.groupby((x.heating_relay != x.heating_relay.shift()).cumsum())))
    df.loc[source_mask, "heating_start"] = heating_grouping.date.transform('first').where(df.heating_relay, None)
    df.loc[source_mask, "heating_length"] = heating_grouping.heating_segment.transform('sum').where(df.heating_relay, None)


def _get_heater_id(devices: dict[str, Device]) -> str:
    return next(k for k, v in devices.items() if v.is_heater)
