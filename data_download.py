import os
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from pandas.tseries.offsets import BDay

# take a look at potentially including UMCSENT, data to feb 2024
# review code to handle quartely data noted on non business days

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()


def download_dataset(dataset_name, path="datasets/"):
    """
    Downloads a dataset from Kaggle and saves it to the specified path.
    """
    api.dataset_download_files(dataset_name, path=path, unzip=True)


def load_and_clean_data(
    file_path, date_col_index, value_col_index, data_freq="D", percent_change=False
):
    """
    Loads a dataset from a CSV file, cleans it, and returns a DataFrame.
    The dataset is cleaned by:
    1. Parsing the date column and value column
    2. Reindexing to include all calendar days
    3. Forward filling to maintain data continuity
    4. Extending monthly/quarterly data to daily
    5. Optionally calculating percent change from the previous period
    """
    try:
        df = pd.read_csv(file_path)
        df = df.iloc[:, [date_col_index, value_col_index]]
        df.columns = ["Date", "Value"]
        df["Date"] = pd.to_datetime(
            df["Date"], errors="coerce"
        )  # Ensure date parsing with error handling
        df["Value"] = df["Value"].replace(
            ",", "", regex=True
        )  # Remove commas from numbers (e.g., 1,000 -> 1000
        df["Value"] = pd.to_numeric(
            df["Value"], errors="coerce"
        )  # Convert Value to numeric, handling errors
        df.dropna(
            subset=["Date", "Value"], inplace=True
        )  # Drop rows where Date or Value could not be parsed

        # Reindex to include all calendar days
        all_days = pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq="D")
        df.set_index("Date", inplace=True)
        df = df.reindex(all_days)
        df["Value"] = df["Value"].ffill()  # Forward fill to maintain data continuity

        if data_freq in ["M", "Q"]:
            # Extend monthly/quarterly data to daily
            if data_freq == "M":
                period = "M"
            elif data_freq == "Q":
                period = "Q"

            # Calculate percent change from the previous period if requested
            if percent_change:
                # Calculate percent change from the previous period
                period_df = df.resample(period).first()
                period_df["Percent Change"] = period_df["Value"].pct_change() * 100
                period_df.dropna(inplace=True)
                period_df["Period"] = period_df.index

                # Assign the calculated percent change to all days within the period
                df["Percent Change"] = np.nan
                for idx in period_df.index:
                    df.loc[
                        df.index.to_period(period) == idx.to_period(period),
                        "Percent Change",
                    ] = period_df.loc[idx, "Percent Change"]

                df["Value"] = df["Percent Change"]
                df.drop(columns=["Percent Change"], inplace=True)
            else:
                df = df.resample("D").ffill()

            # Filter to only include business days
            business_days = df.index.to_series().dt.dayofweek < 5
            business_df = df[business_days].copy()
            # business_df = business_df.drop(columns=["Percent Change"], errors='ignore')

        elif data_freq == "D":
            # Filter to only include business days
            business_days = df.index.to_series().dt.dayofweek < 5
            business_df = df[business_days].copy()

            if percent_change:
                business_df["Value"] = business_df["Value"].pct_change() * 100
                business_df.dropna(inplace=True)  # Removes NaNs created by pct_change

        return business_df.reset_index().rename(columns={"index": "Date"})
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def merge_data(cleaned_data_dir="datasets/cleaned", output_file="merged.csv"):
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
        df = pd.read_csv(file)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        # Ensure the column names are unique by naming them after the files
        df.rename(
            columns={"Value": os.path.splitext(os.path.basename(file))[0]}, inplace=True
        )
        data_frames.append(df)

    # Merge all dataframes on the 'Date' index
    merged_df = pd.concat(
        data_frames, axis=1, join="inner"
    )  # 'inner' join to only keep dates present in all files

    # Save the merged DataFrame
    merged_df.reset_index().to_csv(
        os.path.join(cleaned_data_dir, output_file), index=False
    )
    print(f"Merged data saved to {os.path.join(cleaned_data_dir, output_file)}")


def main():
    # List of datasets on Kaggle
    # Each dataset is a tuple with the following elements:
    # 1. Dataset name on Kaggle
    # 2. Index of the date column in the dataset
    # 3. Index of the value column in the dataset
    # 4. Whether to calculate percent change from the previous business day
    # 5. Whether to interpolate between change points
    datasets = {
        # Dataset frequency: daily including weekends
        "simonwildsmith/us-uncertainty-index": [
            "United States uncertainty index",
            0,
            1,
            "D",
            False,
        ],
        # Dataset frequency: daily not including weekends
        "simonwildsmith/10y-2y-tbill-constant-maturity": ["T10Y2Y", 0, 1, "D", False],
        # Dataset frequency: daily not including weekends
        "simonwildsmith/us-dollar-index": ["US Dollar Index (DXY)", 0, 4, "D", False],
        # Dataset frequency: quarterly, data may lie on non-business days
        "simonwildsmith/gdp-growth-quarterly-from-preceding-period": [
            "GDP growth",
            0,
            1,
            "Q",
            False,
        ],
        # Dataset frequency: monthly, data may lie on non-business days
        "simonwildsmith/us-unemployment-rate": ["Unemployment Rate", 0, 1, "M", False],
        # Dataset frequency: monthly, data may lie on non-business days
        "simonwildsmith/consumer-price-index-monthly-seasonally-adjusted": [
            "Consumer Price Index",
            0,
            1,
            "M",
            True,
        ],
        # Dataset frequency: daily including weekends
        "simonwildsmith/federal-funds-effective-rate-1979-march-2024": [
            "Effective Funds Rate DFF",
            0,
            1,
            "D",
            False,
        ],
        # Dataset frequency: daily not including weekends
        "simonwildsmith/historical-gold-prices-march-2024": [
            "Historical Gold Prices",
            0,
            1,
            "D",
            True,
        ],
        # Dataset frequency: monthly, data may lie on non-business days
        "simonwildsmith/m2-money-supply": ["M2SL", 0, 1, "M", True],
        # Dataset frequency: monthly, data may lie on non-business days
        "simonwildsmith/umcsent": ["UMCSENT", 0, 1, "M", False],
        # Dataset frequency: daily not including weekends
        "simonwildsmith/market-yield-on-us-treasury-securities-10-year": [
            "DGS10",
            0,
            1,
            "D",
            False,
        ],
        # Data frequency: monthly, data may lie on non-business days
        "simonwildsmith/inflation-expectation-12-month": ["MICH", 0, 1, "M", False],
        # Data frequency: monthly, data may lie on non-business days
        "simonwildsmith/mainland-papers-china-economic-uncertainty-index": [
            "CHNMAINLANDEPU",
            0,
            1,
            "M",
            False,
        ],
        # Data frequency: daily not including weekends
        "simonwildsmith/sp-500": ["SP500_Historical_Prices", 0, 4, "D", True],
    }

    cleaned_data_dir = "datasets/cleaned"

    if not os.path.exists(cleaned_data_dir):
        os.makedirs(cleaned_data_dir)

    for dataset_name, (
        download_name,
        date_col_index,
        value_col_index,
        data_freq,
        percent_change,
    ) in datasets.items():
        print(f"Downloading {dataset_name}...")
        download_dataset(dataset_name)

        file_path = os.path.join("datasets", download_name + ".csv")

        df = load_and_clean_data(
            file_path, date_col_index, value_col_index, data_freq, percent_change
        )

        if df is not None:
            clean_file_path = os.path.join(
                "datasets", "cleaned", download_name + "_cleaned.csv"
            )
            df.to_csv(clean_file_path, index=False)
            print(f"Saved cleaned data to {clean_file_path}")

    merge_data()


if __name__ == "__main__":
    main()
