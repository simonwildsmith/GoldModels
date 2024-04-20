from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

"""
Step 1: Load the data
"""

# Load the data
data_path = "C:/Users/s_wil/OneDrive/Documents/Macro/datasets/cleaned/merged.csv"
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

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

train_params = {
    "degree": 4,
    "batch_size": 32,
    "lrate": 0.0001,
    "num_epochs": 400,
}

# Create the polynomial feature transformer
poly = PolynomialFeatures(degree=train_params["degree"], include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)
X_test_poly = poly.transform(X_test)

# Convert the numpy arrays to DataFrames for easier handling later
X_train_poly = pd.DataFrame(X_train_poly, columns=poly.get_feature_names_out(X.columns))
X_val_poly = pd.DataFrame(X_val_poly, columns=poly.get_feature_names_out(X.columns))
X_test_poly = pd.DataFrame(X_test_poly, columns=poly.get_feature_names_out(X.columns))

print(f"Polynomial features generated: {X_train_poly.head()}")

# Normalize the polynomial features
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_val_poly_scaled = scaler.transform(X_val_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

# Convert the scaled features and targets into PyTorch tensors
train_features = torch.tensor(X_train_poly_scaled, dtype=torch.float32)
train_targets = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
val_features = torch.tensor(X_val_poly_scaled, dtype=torch.float32)
val_targets = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
test_features = torch.tensor(X_test_poly_scaled, dtype=torch.float32)
test_targets = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoaders
train_loader = DataLoader(
    TensorDataset(train_features, train_targets),
    batch_size=train_params["batch_size"],
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(val_features, val_targets),
    batch_size=train_params["batch_size"],
    shuffle=False,
)
test_loader = DataLoader(
    TensorDataset(test_features, test_targets),
    batch_size=train_params["batch_size"],
    shuffle=False,
)


class PolyLinearModel(nn.Module):
    def __init__(self, input_size):
        super(PolyLinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


# Initialize the model
input_size = X_train_poly_scaled.shape[1]
model = PolyLinearModel(input_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=train_params["lrate"])


# Train the model
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

        # Validation loss
        val_loss = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)

        # print number of epochs completed of total
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

    return train_losses, val_losses


def evaluate_model(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
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
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.view(-1).tolist())
    return predictions


# Start training
num_epochs = train_params["num_epochs"]
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

# print the model weights next to the feature names
feature_names = X_train_poly.columns
weights = model.linear.weight.data.numpy().flatten()
bias = model.linear.bias.data.numpy().flatten()

# append the model params, final losses, and weights for each feature to a csv
train_params["test_loss"] = test_loss
train_params["final_train_loss"] = train_losses[-1]
train_params["final_val_loss"] = val_losses[-1]
train_params["weights"] = weights.tolist()
train_params["bias"] = bias.tolist()

# Create a dictionary of feature names and their corresponding weights
feature_weights = {feature: weight for feature, weight in zip(feature_names, weights)}

# Append the feature weights to the model_params dictionary
train_params["feature_weights"] = feature_weights

model_params_list = [train_params]  # Convert the dictionary to a list of dictionaries

model_params_df = pd.DataFrame(model_params_list)

# Create a DataFrame with the correct shape
model_params_df.to_csv(
    "C:/Users/s_wil/OneDrive/Documents/Macro/results/polynomial_regression/polynomial_linear_regression.csv",
    mode="a",
    header=False,
    index=False,
)
