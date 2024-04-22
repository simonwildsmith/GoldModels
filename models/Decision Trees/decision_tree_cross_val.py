import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize the features
scaler = StandardScaler()
scaler.fit(X_train)

# Scale the training, validation, and test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define training parameters
train_params = {"max_depth": 1, "random_state": 42}

# Initialize the DecisionTreeRegressor
model = DecisionTreeRegressor(**train_params)

# Perform cross-vlaidation on the training set
scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="neg_mean_squared_error")

# Train the model using the scaled training data
model.fit(X_train_scaled, y_train)

# Predict the target values
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Test MSE: {test_mse:.4f}")
print(f"Cross-validated MSE: {-scores.mean():.4f} (+/- {scores.std()*2:.4f})")
