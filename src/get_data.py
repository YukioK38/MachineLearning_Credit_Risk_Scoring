# This file gets data from the Federal Reserve Bank's API and saves it as a fred.csv
# NOTE: Requires a free API key: https://fred.stlouisfed.org/docs/api/api_key.html

import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv # type: ignore
load_dotenv()


# Setup (run once in your terminal before using this file):
# Windows:    set FRED_API_KEY="your_key_here"
# Mac/Linux:  export FRED_API_KEY="your_key_here"
FRED_API_KEY = os.environ.get("FRED_API_KEY")

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True) # Create folder

FRED_SERIES = {
    "default_rate_total": "DRCLACBS",   # Delinquency Rate on Consumer Loans, All Commercial Banks
    "default_rate_credit_card": "DRCCLACBS",  # Delinquency Rate on Credit Card Loans, All Commercial Banks
    "default_rate_other": "DROCLACBS"  # Delinquency Rate on Other Consumer Loans, All Commercial Banks
}



# Gets a time series using the FRED API. Will be used for the next function
def get_series(code, date_start, date_end) -> pd.DataFrame:
    if not FRED_API_KEY:
        raise EnvironmentError(
            "FRED_API_KEY missing. "
            "Set it in your terminal with: export FRED_API_KEY='your_key_here'"
        )
    
    url = "https://api.stlouisfed.org/fred/series/observations"

    dict = {
        "series_id": code,
        "observation_start": date_start,
        "observation_end": date_end,
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }

    response = requests.get(url, params=dict, timeout=10)
    response.raise_for_status()

    '''
        This is how the FRED API outputs data:
            {
        "realtime_start": "2003-01-01",
        "realtime_end": "2013-01-01",
        "observation_start": "2005-01-01",
        "observation_end": "2013-01-01",
        "units": "lin",
        "observations": [
            {"date": "2005-01-01", "value": "4.51"},
            {"date": "2005-04-01", "value": "4.38"}
        ]
    }   Note that both time and value are strings and they are inside
        of "observations"
    '''

    # get data from observations
    observations = response.json()["observations"]
    df = pd.DataFrame(observations)

    # Keep only the columns we need
    df = df[["date", "value"]]

    # Convert types — FRED returns both as strings
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # FRED uses "." to represent missing values — replace before converting to float
    df["value"] = df["value"].replace(".", pd.NA)
    df["value"] = pd.to_numeric(df["value"])
    
    return(df)



# Gets all the series from series via the FRED API
def get_all_series(series, date_start, date_end) -> pd.DataFrame:
    dfs = []
    for name, code in series.items():
        df = get_series(code, date_start, date_end)
        df = df.rename(columns={"value": name}).set_index("date")
        dfs.append(df)

    result = pd.concat(dfs, axis=1, join='outer').sort_index()
    
    return(result)



# Saves the dataframe in our chosen path ("data/raw") and returns the path 
def save_dataframe(df, name) -> Path:
    output_path = RAW_DATA_DIR / name
    df.to_csv(output_path)
    print(f" Saved on {output_path}")
    return(output_path)



if __name__ == "__main__":
    df = get_all_series(FRED_SERIES, "2003-01-01", "2013-01-01")
    print(df.head()) 
    print(df.tail())
    print(df.shape)
    print(df.isnull().sum())
    save_dataframe(df, "fred.csv")
    print("Completed")
