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
    file_path, date_col_index, value_col_index, percent_change=False, interpolate=True
):
    """
    Loads a dataset from a CSV file, cleans it, and returns a DataFrame.
    The dataset is cleaned by:
    1. Parsing the date column and value column
    2. Reindexing to include all calendar days
    3. Forward filling to maintain data continuity
    4. Optionally interpolating between change points
    5. Optionally calculating percent change from the previous business day
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

        # Filter to only include business days before identifying change points
        business_days = all_days.to_series().dt.dayofweek < 5
        business_df = df[business_days].copy()

        # Interpolate between change points if requested
        if interpolate:
            # Detect where actual changes occur
            changes = business_df["Value"].diff().fillna(0) != 0
            change_dates = changes[changes].index

            # Include the first row explicitly as a change point
            if change_dates.empty or change_dates[0] != business_df.index.min():
                change_dates = change_dates.insert(0, business_df.index.min())

            # Perform linear interpolation between change points
            for start, end in zip(change_dates, change_dates[1:]):
                if end > start:  # Ensure there is more than one day to interpolate
                    indices = business_df.loc[start:end].index
                    business_df.loc[indices, "Value"] = np.linspace(
                        business_df.at[start, "Value"],
                        business_df.at[end, "Value"],
                        len(indices),
                    )

        # Calculate percent change on business days if requested
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
            False,
            False,
        ],
        # Dataset frequency: daily not including weekends
        "simonwildsmith/10y-2y-tbill-constant-maturity": ["T10Y2Y", 0, 1, False, False],
        # Dataset frequency: daily not including weekends
        "simonwildsmith/us-dollar-index": ["US Dollar Index (DXY)", 0, 4, False, False],
        # Dataset frequency: quarterly, data may lie on non-business days
        "simonwildsmith/gdp-growth-quarterly-from-preceding-period": [
            "GDP growth",
            0,
            1,
            False,
            True,
        ],
        # Dataset frequency: monthly, data may lie on non-business days
        "simonwildsmith/us-unemployment-rate": ["Unemployment Rate", 0, 1, False, True],
        # Dataset frequency: monthly, data may lie on non-business days
        "simonwildsmith/consumer-price-index-monthly-seasonally-adjusted": [
            "Consumer Price Index",
            0,
            1,
            True,
            True,
        ],
        # Dataset frequency: daily including weekends
        "simonwildsmith/federal-funds-effective-rate-1979-march-2024": [
            "Effective Funds Rate DFF",
            0,
            1,
            False,
            False,
        ],
        # Dataset frequency: daily not including weekends
        "simonwildsmith/historical-gold-prices-march-2024": [
            "Historical Gold Prices",
            0,
            1,
            True,
            False,
        ],
        # Dataset frequency: monthly, data may lie on non-business days
        "simonwildsmith/m2-money-supply": ["M2SL", 0, 1, True, True],
        # Dataset frequency: monthly, data may lie on non-business days
        "simonwildsmith/umcsent": ["UMCSENT", 0, 1, False, True],
    }

    cleaned_data_dir = "datasets/cleaned"

    if not os.path.exists(cleaned_data_dir):
        os.makedirs(cleaned_data_dir)

    for dataset_name, (
        download_name,
        date_col_index,
        value_col_index,
        percent_change,
        interpolate,
    ) in datasets.items():
        print(f"Downloading {dataset_name}...")
        download_dataset(dataset_name)

        file_path = os.path.join("datasets", download_name + ".csv")

        df = load_and_clean_data(
            file_path, date_col_index, value_col_index, percent_change, interpolate
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
