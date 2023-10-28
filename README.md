# HIVE HOME DASHBOARD

Simple app for loading heating data from Hive and displaying plots on a Streamlit page

## Install and run dashboard app

Installation: 

```python
pip3 install -r requirements.txt
```

Running: 
```python
streamlit run app.py
```

## Using 'API'


### Authentication
Hive requires multifactor authentication (MFA) so authentication process requires two calls: 

1. Initialise the authentication
```python
import hive

hive.authenticate(hive.Credentials(username="<email>", password="<password>"))
```

2. Complete the authentication using the SMS code received on registered user's phone
```python
import hive

hive.authenticate(hive.Credentials(username="<email>", mfa_code="<SMS CODE>"))
```

The authentication state is automatically stored between calls in the `data/tokens.json`. 

### Getting aggregated data

Method `get_device_data` returns a dataframe containing historic information since 1.03.2023 and current state of all heating devices. There's some processing magic going on (temperature interpolation for missing timestamps, heating windows calculations). First call is SLOW (minutes). As this method stores the cached result in pickled dataframe, the consequent calls are fetching new updates and are much faster. 


```python
import hive

df = hive.get_device_data(True)
print(df.head())
```

### Raw data
```python
import hive
from datetime import datetime

devices = hive.get_devices()
data_dict = hive.fetch_device_data(devices.keys(), datetime(2023, 10, 1))
df = hive.create_device_dataframe(devices, data_dict)
print(df.head())
```


### Caveats
* Authentication only lasts 1 hour and then expires and has to be done again.\
Theoretically, it's possible to use "refresh tokens" like the native hive app does, but I could not manage to make it work. 

* My dataframe post-processing has some bugs so not all heating windows are calculated. 