import os
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from pandas.tseries.offsets import MonthEnd

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()


def download_dataset(dataset_name, path="datasets_month/"):
    """
    Downloads a dataset from Kaggle and saves it to the specified path.
    """
    api.dataset_download_files(dataset_name, path=path, unzip=True)


def adjust_dates(df, freq):
    """
    Adjusts the dates of the DataFrame backward to align with the reported period.
    """
    if freq == "Q":
        # Shift quarterly data to the start of the quarter
        df.index = df.index - pd.offsets.QuarterBegin(startingMonth=1)
    elif freq == "M":
        # Shift monthly data to the start of the month
        df.index = df.index - pd.offsets.MonthBegin(1)
    return df


def resample_data(df, freq, percent_change, interpolate):
    """
    Resamples the data based on the frequency and requirements for percent change and interpolation.
    """
    df = adjust_dates(df, freq)

    if percent_change and interpolate:
        df = (
            df.resample("M").mean().interpolate()
        )  # Quarterly data to monthly data with interpolation
        df["Value"] = (
            df["Value"].pct_change().dropna() * 100
        )  # Calculate monthly percent change
    elif percent_change:
        df = df.resample("M").mean()  # Monthly or quarterly data
        df["Value"] = (
            df["Value"].pct_change().dropna() * 100
        )  # Calculate monthly percent change
    elif interpolate:
        df = (
            df.resample("M").mean().interpolate()
        )  # Quarterly data to monthly data with interpolation
    else:
        df = df.resample("M").mean()  # Average for the month or already monthly data

    return df


def process_dataset(
    file_path, date_col_index, value_col_index, percent_change, interpolate, freq
):
    """
    Process dataset to load, clean, and resample.
    """
    df = pd.read_csv(file_path, usecols=[date_col_index, value_col_index])
    df.columns = ["Date", "Value"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Value"] = df["Value"].replace(
        ",", "", regex=True
    )  # Remove commas from numbers (e.g., 1,000 -> 1000
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df.dropna(subset=["Date", "Value"], inplace=True)
    df.set_index("Date", inplace=True)

    return resample_data(df, freq, percent_change, interpolate)


def merge_data(
    cleaned_data_dir="datasets_month/cleaned", output_file="monthly_merged.csv"
):
    """
    Merges all cleaned data files into a single DataFrame where each column corresponds to a dataset.
    Only includes dates that have entries in all columns.
    """
    files = [
        os.path.join(cleaned_data_dir, f)
        for f in os.listdir(cleaned_data_dir)
        if f.endswith("_cleaned.csv")
    ]
    data_frames = []

    for file in files:
        df = pd.read_csv(file, parse_dates=["Date"])
        data_source_name = os.path.splitext(os.path.basename(file))[0].replace(
            "_cleaned", ""
        )
        df.set_index("Date", inplace=True)
        df.rename(
            columns={"Value": data_source_name}, inplace=True
        )  # Rename column to data source name
        data_frames.append(df)

    # Merge all dataframes on the 'Date' index
    merged_df = pd.concat(
        data_frames, axis=1, join="inner"
    )  # 'inner' join to only keep dates present in all files

    # Save the merged DataFrame
    merged_df.to_csv(os.path.join(cleaned_data_dir, output_file))
    print(f"Merged data saved to {os.path.join(cleaned_data_dir, output_file)}")


def main():
    datasets = {
        # Dataset frequency: daily including weekends
        "simonwildsmith/us-uncertainty-index": [
            "United States uncertainty index",
            0,
            1,
            False,
            False,
            "D",
        ],
        # Dataset frequency: daily not including weekends
        "simonwildsmith/10y-2y-tbill-constant-maturity": [
            "T10Y2Y",
            0,
            1,
            False,
            False,
            "D",
        ],
        # Dataset frequency: daily not including weekends
        "simonwildsmith/us-dollar-index": [
            "US Dollar Index (DXY)",
            0,
            4,
            False,
            False,
            "D",
        ],
        # Dataset frequency: quarterly, data may lie on non-business days
        "simonwildsmith/gdp-growth-quarterly-from-preceding-period": [
            "GDP growth",
            0,
            1,
            False,
            True,
            "Q",
        ],
        # Dataset frequency: monthly, data may lie on non-business days
        "simonwildsmith/us-unemployment-rate": [
            "Unemployment Rate",
            0,
            1,
            False,
            False,
            "M",
        ],
        # Dataset frequency: monthly, data may lie on non-business days
        "simonwildsmith/consumer-price-index-monthly-seasonally-adjusted": [
            "Consumer Price Index",
            0,
            1,
            True,
            False,
            "M",
        ],
        # Dataset frequency: daily including weekends
        "simonwildsmith/federal-funds-effective-rate-1979-march-2024": [
            "Effective Funds Rate DFF",
            0,
            1,
            False,
            False,
            "D",
        ],
        # Dataset frequency: daily not including weekends
        "simonwildsmith/historical-gold-prices-march-2024": [
            "Historical Gold Prices",
            0,
            1,
            True,
            False,
            "D",
        ],
        # Dataset frequency: monthly, data may lie on non-business days
        "simonwildsmith/m2-money-supply": ["M2SL", 0, 1, True, False, "M"],
        # Dataset frequency: monthly, data may lie on non-business days
        "simonwildsmith/umcsent": ["UMCSENT", 0, 1, False, False, "M"],
        # Dataset frequency: daily not including weekends
        "simonwildsmith/market-yield-on-us-treasury-securities-10-year": [
            "DGS10",
            0,
            1,
            False,
            False,
            "D",
        ],
        # Data frequency: monthly, data may lie on non-business days
        "simonwildsmith/inflation-expectation-12-month": [
            "MICH",
            0,
            1,
            False,
            False,
            "M",
        ],
        # Data frequency: monthly, data may lie on non-business days
        "simonwildsmith/mainland-papers-china-economic-uncertainty-index": [
            "CHNMAINLANDEPU",
            0,
            1,
            False,
            False,
            "M",
        ],
        # Data frequency: daily not including weekends
        "simonwildsmith/sp-500": ["SP500_Historical_Prices", 0, 4, True, False, "D"],
    }

    cleaned_data_dir = "datasets_month/cleaned"
    os.makedirs(cleaned_data_dir, exist_ok=True)

    for key, (name, date_col, value_col, pct_change, interp, freq) in datasets.items():
        print(f"Processing {name}...")
        download_dataset(key)
        file_path = os.path.join("datasets_month", name + ".csv")
        df = process_dataset(file_path, date_col, value_col, pct_change, interp, freq)
        output_path = os.path.join(cleaned_data_dir, name + "_cleaned.csv")
        df.to_csv(output_path)
        print(f"Saved cleaned data to {output_path}")

    merge_data()


if __name__ == "__main__":
    main()
