import yfinance as yf

def download_sp500_data():
    # Define the ticker symbol for the S&P 500
    ticker = "^GSPC"  # This is the ticker symbol for the S&P 500 Index

    # Download the historical data for the S&P 500
    # Adjust the period as needed, e.g., "1y" for 1 year, "5y" for 5 years, "max" for maximum data
    data = yf.download(ticker, start="1978-01-01", end="2024-04-23")

    # Save the data to a CSV file
    data.to_csv("SP500_Historical_Prices.csv")

    print("Data downloaded and saved to SP500_Historical_Prices.csv.")

# Call the function
download_sp500_data()
