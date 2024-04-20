import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools
from itertools import combinations

"""
Step 1: Load the data
"""

# Load the data
data_path = "datasets/cleaned/merged.csv"
data = pd.read_csv(data_path)
print(data.head())

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

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


def train_and_evaluate(X_train, y_train, X_val, y_val, num_epochs=50):
    # Convert data to tensors
    train_features = torch.tensor(X_train.values, dtype=torch.float32)  # Change here
    train_targets = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    val_features = torch.tensor(X_val.values, dtype=torch.float32)  # Change here
    val_targets = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # DataLoader setup
    train_loader = DataLoader(
        TensorDataset(train_features, train_targets), batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_features, val_targets), batch_size=64, shuffle=False
    )

    # Model setup
    model = LinearRegressionModel(train_features.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Tracking losses
    epoch_losses = {"epoch": [], "train_loss": [], "val_loss": []}

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * inputs.size(0)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                total_val_loss += val_loss.item() * inputs.size(0)

        # Record epoch losses
        epoch_losses["epoch"].append(epoch)
        epoch_losses["train_loss"].append(total_train_loss / len(train_loader.dataset))
        epoch_losses["val_loss"].append(total_val_loss / len(val_loader.dataset))

    return epoch_losses


# Prepare feature combinations excluding empty set
feature_names = X_train.columns.tolist()  # Ensure X_train is a DataFrame
all_combinations = []
for r in range(1, len(feature_names) + 1):
    all_combinations.extend(combinations(feature_names, r))

# Dictaionary to store results
all_results_df = pd.DataFrame()

for combo in all_combinations:
    print(
        f"Processing combination: ({all_combinations.index(combo)+1}/{len(all_combinations)})"
    )

    # Select the features for this combination
    X_train_subset = X_train[list(combo)]
    X_val_subset = X_val[list(combo)]

    # Train and evaluate the model
    combo_results = train_and_evaluate(X_train_subset, y_train, X_val_subset, y_val)

    # Create a DataFrame from the results
    df = pd.DataFrame(combo_results)
    df["features"] = str(combo)  # Add a column with the feature combination
    df["combo_index"] = all_combinations.index(
        combo
    )  # Optional: add combo index for easier tracking

    # Append the current DataFrame to the all_results_df
    all_results_df = pd.concat([all_results_df, df], ignore_index=True)

# Save all results to a single CSV file
all_results_df.to_csv("all_combinations_results.csv", index=False)
