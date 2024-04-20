import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt


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

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

"""
Setup Pytorch for a Simple Linear Model
"""

# Convert arrays to PyTorch tensors
train_features = torch.tensor(X_train, dtype=torch.float32)
train_targets = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
val_features = torch.tensor(X_val, dtype=torch.float32)
val_targets = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
test_features = torch.tensor(X_test, dtype=torch.float32)
test_targets = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create Tensor Datasets
train_dataset = TensorDataset(train_features, train_targets)
val_dataset = TensorDataset(val_features, val_targets)
test_dataset = TensorDataset(test_features, test_targets)

# Create Data Loaders
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

"""
Step 3: Define the model
"""


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


# Initialize the model
input_size = X_train.shape[1]
model = LinearRegressionModel(input_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)

"""
Step 4: Train the model concurrently with the validation set
"""


def train_and_evaluate(
    model, train_loader, val_loader, criterion, optimizer, num_epochs
):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        val_loss = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f})")

    return train_losses, val_losses

def evaluate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(val_loader.dataset)

def predict(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():  # Inference mode, gradients not needed
        for inputs, _ in test_loader:
            outputs = model(inputs)
            predictions.extend(
                outputs.view(-1).tolist()
            )  # Flatten outputs and convert to list
    return predictions


"""
Step 5: Plot the losses
"""

num_epochs = 300
train_losses, val_losses = train_and_evaluate(
    model, train_loader, val_loader, criterion, optimizer, num_epochs
)

# Plotting the training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
plt.grid(True)

# Evaluate the model on the test set
test_loss = evaluate_model(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}")

"""
Step 6: Make predictions
"""

# Make Predictios on the test set and plot them
predictions = predict(model, test_loader)

assert len(predictions) == len(
    y_test
), "Number of predictions must match number of test samples"

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="True")
plt.plot(predictions, label="Predicted")
plt.xlabel("Sample")
plt.ylabel("Price")
plt.title("True vs Predicted Prices")
plt.legend()
plt.show()
plt.grid(True)
