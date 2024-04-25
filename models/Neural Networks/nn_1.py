import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold



# Load the data
# data_path = "datasets/cleaned/merged.csv"
data_path = "C:/Users/s_wil/OneDrive/Documents/Macro/datasets/cleaned/merged.csv"
data = pd.read_csv(data_path)

# Preprocess the data
if data.isnull().values.any():
    num_rows_before = data.shape[0]
    data = data.dropna()
    num_rows_after = data.shape[0]
    print(f"Number of rows removed: {num_rows_before - num_rows_after}")

# Split the data into features and target
X = data.drop(columns=["Date", "Historical Gold Prices_cleaned"])
y = data["Historical Gold Prices_cleaned"]


class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Initialize K-Fold Cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Loop through the folds
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    # Split the data
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    # Scale the data within the fold
    scaler = StandardScaler()
    X_train_fold_scaled = scaler.fit_transform(X_train_fold)
    X_val_fold_scaled = scaler.transform(X_val_fold)

    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_fold_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_fold_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_fold.values, dtype=torch.float32).reshape(-1, 1)
    y_val_tensor = torch.tensor(y_val_fold.values, dtype=torch.float32).reshape(-1, 1)

    # Initialize the model and loss function
    model = RegressionNN(input_dim=X_train_tensor.shape[1])
    criteron = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Arrays to store losses
    train_losses, val_losses = [], []

    # Train the model
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criteron(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criteron(val_outputs, y_val_tensor)
            val_losses.append(val_loss.item())
        
    print(f"Fold {fold+1}, Validation Loss: {val_loss.item():.4f}")
    plot_losses(train_losses, val_losses)


