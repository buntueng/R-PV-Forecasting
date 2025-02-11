import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset
df = pd.read_csv("dataset/preprocess_dataset.csv")
df["DateTime"] = pd.to_datetime(df["DateTime"])
df.set_index("DateTime", inplace=True)

# Split into Training & Testing
train_data = df["2013-01-01":"2017-12-31"].values
test_data = df["2018-01-01":"2018-12-31"].values

# show length of train and test data
print(f"Train Data Length: {len(train_data)}")
print(f"Test Data Length: {len(test_data)}")

# Convert to PyTorch Tensor
train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)

# Define Dataset Class
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, input_size=9, output_size=9):
        self.data = data
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.data) - self.input_size - self.output_size

    def __getitem__(self, index):
        x = self.data[index : index + self.input_size]
        y = self.data[index + self.input_size : index + self.input_size + self.output_size, 0]  # Forecast MW column
        return x, y

# Create Dataloader
batch_size = 32
train_dataset = TimeSeriesDataset(train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use last LSTM output
        return out

# Model Parameters
input_size = train_data.shape[1]  # Number of features
hidden_size = 64
num_layers = 2
output_size = 9  # Forecast 9 future points
learning_rate = 0.001
num_epochs = 5

# Initialize Model
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# summary of the model
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
train_losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, Y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save Training History
pd.DataFrame({"Epoch": range(1, num_epochs+1), "Train_Loss": train_losses}).to_csv("training_history.csv", index=False)

# Save Model
torch.save(model.state_dict(), "lstm_model.pth")
print("Model saved as 'lstm_model.pth'.")


# Model Evaluation
model.eval()
with torch.no_grad():
    X_test = test_tensor[:9].unsqueeze(0).to(device)  # Initial input sequence (1, 9, features)
    predictions = []

    for i in range(len(test_tensor) - 9):
        pred = model(X_test).cpu().numpy().flatten()  # Get the prediction for the next step
        predictions.append(pred)

        # Convert prediction to tensor and reshape to match expected input
        # pred_tensor = torch.tensor(pred).unsqueeze(0).to(device)  # Shape:
        # pred_tensor = pred_tensor.repeat(1, 1, 5)  # Repeat along feature dimension to get
        # Assuming 'pred' contains the 9 predicted values
        pred_tensor = torch.tensor(pred).reshape(1, 1, 9).to(device)  # Reshape to
        
        
        print("X_test shape:", X_test[:, 1:,:].shape)
        print("pred_tensor shape:", pred_tensor[:, -1:,:].shape)

        # Shift input sequence: remove the first time step and add new prediction
        X_test = torch.cat((X_test[:, 1:,:], pred_tensor), dim=1)

predictions = np.array(predictions)


# Get Actual Values for Comparison
actual_values = test_data[9:, 0]  # MW column

# Calculate Error Metrics
rmse = np.sqrt(mean_squared_error(actual_values, predictions[:, 0]))
mae = mean_absolute_error(actual_values, predictions[:, 0])
r2 = r2_score(actual_values, predictions[:, 0])

# Save Evaluation Metrics
with open("evaluation_metrics.txt", "w") as f:
    f.write(f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2 Score: {r2:.4f}")

print(f"Evaluation Metrics:\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# Save Actual vs Forecasting Results
results_df = pd.DataFrame({"Actual": actual_values, "Forecast": predictions[:, 0]})
results_df.to_csv("forecast_results.csv", index=False)

print("Forecasting results saved to 'forecast_results.csv'.")
