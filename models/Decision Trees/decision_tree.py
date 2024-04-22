import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

"""
Step 1: Load the data
"""

# When working on codespaces
data_path = "datasets/cleaned/merged.csv"

# When working on personal machine
# ...

data = pd.read_csv(data_path)

"""
Step 2: Preprocess the data
"""

# Drop missing values
if data.isnull().values.any():
    num_rows_before = data.shape[0]
    data = data.dropna()
    num_rows_after = data.shape[0]
    num_rows_removed = num_rows_before - num_rows_after
    print(f"Number of rows removed: {num_rows_removed}")

# Split the data into features and target
X = data.drop(columns=["Date", "Historical Gold Prices_cleaned"])
y = data["Historical Gold Prices_cleaned"]

# Split the data into training and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split the data into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Normalize the features
scaler = StandardScaler()
scaler.fit(X_train)

# Scale the training, validation, and test data
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define training parameters
train_params = {"max_depth": 7, "random_state": 42}

# Initialize the DecisionTreeRegressor
model = DecisionTreeRegressor(**train_params)

# Train the model using the scaled training data
model.fit(X_train, y_train)

# Predict the target values
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calculate the mean squared errors
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Training MSE: {train_mse}")
print(f"Validation MSE: {val_mse}")
print(f"Test MSE: {test_mse}")
