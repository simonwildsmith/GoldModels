import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

# Load the data
data_path = "datasets/cleaned/merged.csv"
data = pd.read_csv(data_path)
print(data.head())

# Preprocess the data
if data.isnull().values.any():
    num_rows_before = data.shape[0]
    data = data.dropna()
    num_rows_after = data.shape[0]
    print(f"Number of rows removed: {num_rows_before - num_rows_after}")

# Split the data into features and target
X = data.drop(columns=["Date", "Historical Gold Prices_cleaned"])
y = data["Historical Gold Prices_cleaned"]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Setup cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

model_params = {
    "learning_rate": 0.0005,
    "num_epochs": 200,
    "batch_size": 32,
}

# Define the PyTorch model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    def forward(self, x):
        return self.linear(x)

# Initialize model components
criterion = nn.MSELoss()

# Train and evaluate using cross-validation
fold = 1
all_fold_results = []

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    train_features = torch.tensor(X_train, dtype=torch.float32)
    train_targets = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    test_features = torch.tensor(X_test, dtype=torch.float32)
    test_targets = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    train_dataset = TensorDataset(train_features, train_targets)
    test_dataset = TensorDataset(test_features, test_targets)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=model_params["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=model_params["batch_size"], shuffle=False)
    
    input_size = X_train.shape[1]
    model = LinearRegressionModel(input_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=model_params["learning_rate"])
    
    # Train the model
    for epoch in range(model_params["num_epochs"]):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{model_params['num_epochs']}, Loss: {loss.item():.4f}")

    # Evaluate the model
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
    test_loss /= len(test_loader.dataset)
    all_fold_results.append(test_loss)
    print(f"Fold {fold}, Test Loss: {test_loss:.4f}")
    fold += 1

# Display average test loss across folds
print("Average Test Loss:", np.mean(all_fold_results))

# Additional plotting and model saving code can be added here.
