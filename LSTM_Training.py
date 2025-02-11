import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import time

# 1. Data Loading and Preprocessing
df = pd.read_csv("dataset/preprocess_dataset.csv", parse_dates=["DateTime"])

# Filter data between 8 AM and 5 PM
df["Hour"] = df["DateTime"].dt.hour
df = df[(df["Hour"] >= 8) & (df["Hour"] < 17)]

# Resample to 1-hour intervals
df = df.set_index("DateTime").resample("H").agg({"MW": "mean", "Capacity": "mean"})
df = df.reset_index()

# Feature Engineering (Cyclic Time of Day)
minutes = df['DateTime'].dt.hour * 60 + df['DateTime'].dt.minute
df['Time_sin'] = np.sin(2 * np.pi * minutes / (15 * 60 * 24))
df['Time_cos'] = np.cos(2 * np.pi * minutes / (15 * 60 * 24))

# Day of Year (Cyclic)
day_of_year = df['DateTime'].dt.dayofyear
df['Day_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
df['Day_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)


# Prepare data for LSTM
df = df.set_index('DateTime')

# Scaling 
scaler_MW = MinMaxScaler()
df['MW'] = scaler_MW.fit_transform(df['MW'].values.reshape(-1, 1))

scaler_Capacity = MinMaxScaler()
df['Capacity'] = scaler_Capacity.fit_transform(df['Capacity'].values.reshape(-1, 1))

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 2. Data Splitting and Sequence Creation
train_data = df['2013-01-01':'2020-12-31']
test_data = df['2021-01-01':'2021-12-31']

print(train_data.head())
print(test_data.head())

seq_length = 9  # Number of hours in a day (8 AM to 5 PM)

X_train, y_train = create_sequences(train_data.values, seq_length)
X_test, y_test = create_sequences(test_data.values, seq_length)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# 3. LSTM Model Building and Training (PyTorch)
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)  # batch_first=True
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]  # Get the last time step's output
        output = self.linear(output)
        return output

input_dim = X_train.shape[2]  # Number of features
hidden_dim = 50  # Adjust as needed
output_dim = y_train.shape[1]  # Number of output features

model = LSTMModel(input_dim, hidden_dim, output_dim)

criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

num_epochs = 50  # Adjust as needed
batch_size = 32

start_time = time.time()
for epoch in range(num_epochs):
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
end_time = time.time()
training_time = end_time - start_time

# 4. Prediction and Evaluation (PyTorch)
start_time = time.time()
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_pred = model(X_test)
end_time = time.time()
testing_time = end_time - start_time


# Inverse scaling
y_pred_MW = scaler_MW.inverse_transform(y_pred[:, 0].numpy().reshape(-1, 1))
y_test_MW = scaler_MW.inverse_transform(y_test[:, 0].numpy().reshape(-1, 1))


rmse = np.sqrt(mean_squared_error(y_test_MW, y_pred_MW))
mae = mean_absolute_error(y_test_MW, y_pred_MW)
mse = mean_squared_error(y_test_MW, y_pred_MW)
r2 = r2_score(y_test_MW, y_pred_MW)

print(f"Training Time: {training_time:.2f} seconds")
print(f"Testing Time: {testing_time:.2f} seconds")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R-squared: {r2:.4f}")