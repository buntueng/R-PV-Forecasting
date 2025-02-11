import os
import time
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from thop import profile
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Data loading and preprocessing
def load_and_preprocess_data(data_dir, filename, sequence_length):
    data_path = os.path.join(data_dir, filename)
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None, None, None, None

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Hour'] = df['DateTime'].dt.hour
    df = df[(df['Hour'] >= 8) & (df['Hour'] <= 17)]

    input_features = ['MW', 'Time_sin', 'Time_cos', 'Day_sin', 'Day_cos']
    output_feature = 'MW_NextDay'

    daily_data = df.groupby(df['DateTime'].dt.date)[input_features + [output_feature]].apply(lambda x: x.values).to_dict()

    reshaped_daily_data = {}
    for date, values in daily_data.items():
        reshaped_daily_data[date] = {}
        reshaped_daily_data[date]['X'] = values[:,:-1]
        reshaped_daily_data[date]['y'] = values[:, -1].reshape(10)

    train_dates = [date for date in reshaped_daily_data if date.year < 2018]
    test_dates = [date for date in reshaped_daily_data if date.year == 2018]

    X_train = np.stack([reshaped_daily_data[date]['X'] for date in train_dates])
    y_train = np.stack([reshaped_daily_data[date]['y'] for date in train_dates])

    X_test = np.stack([reshaped_daily_data[date]['X'] for date in test_dates])
    y_test = np.stack([reshaped_daily_data[date]['y'] for date in test_dates])

    def create_sequences(X, y, seq_length):
        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:i + seq_length])
            ys.append(y[i + seq_length])
        return np.array(Xs), np.array(ys)

    sequence_length = 1
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


# LSTM Model definition
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        output, (hn, cn) = self.lstm(x, (h0, c0))
        output = output[:, -1,:]
        output = self.fc(output)
        return output


# Training loop
def train_model(model, X_train, y_train, criterion, optimizer, num_epochs, sequence_length, input_dimension, device):
    training_times = []
    training_losses = []
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0

        for i in range(len(X_train)):
            inputs = X_train[i].to(device)
            targets = y_train[i].to(device)
            
            # Reshape the target tensor:
            targets = targets.unsqueeze(0)  # Add batch dimension

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        end_time = time.time()
        epoch_time = end_time - start_time
        training_times.append(epoch_time)
        epoch_loss = running_loss / len(X_train)
        training_losses.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
    return training_times, training_losses


# Evaluation function
def evaluate_model(model, X_test, y_test, device):
    model.eval()  # Set to evaluation mode
    predictions = []
    with torch.no_grad():
        for i in range(len(X_test)):
            inputs = X_test[i].to(device)
            outputs = model(inputs)
            predictions.append(outputs)

    predictions = torch.cat(predictions, dim=0) # Concatenate the list of tensors into one tensor
    predictions = predictions.cpu().numpy()  # Convert to NumPy

    # convert y_test to numpy
    y_test = y_test.cpu().numpy()
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, rmse, mae, r2, predictions


# Main execution (Part 1 - Everything before the actual training)
if __name__ == "__main__":
    # clean all output files
    model_file = os.path.join(os.path.dirname(__file__),"trained_lstm_model.pth")
    training_metrics_file = os.path.join(os.path.dirname(__file__),"training_metrics.csv")
    predictions_file = os.path.join(os.path.dirname(__file__),"predictions.csv")
    parameters_file = os.path.join(os.path.dirname(__file__),"parameters.txt")
    
    if os.path.exists(model_file):
        os.remove(model_file)
    
    if os.path.exists(training_metrics_file):
        os.remove(training_metrics_file)
    
    if os.path.exists(predictions_file):
        os.remove(predictions_file)
        
    if os.path.exists(parameters_file):
        os.remove(parameters_file)
    
    # load the data
    current_directory = os.path.dirname(__file__)
    data_directory = os.path.join(current_directory,"..","dataset")
    file_name = "preprocess_dataset.csv"
    sequence_length = 1
    input_dimension = 5  # Number of input features
    hidden_dimension = 64
    number_of_layers = 2
    output_dimension = 10  # Number of output values (10 hours)
    num_epochs = 300
    learning_rate = 0.001

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_data(
        data_directory, file_name, sequence_length
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LSTMModel(input_dimension, hidden_dimension, number_of_layers, output_dimension).to(device)
    model.to(device)  # Move model to device ONCE

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    training_times, training_losses = train_model(
        model, X_train_tensor, y_train_tensor, criterion, optimizer, num_epochs, sequence_length, input_dimension, device
    )

    # Calculate FLOPs and parameters
    dummy_input = torch.randn(1, sequence_length, input_dimension).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")
    # Save FLOPs and parameters
    with open(parameters_file, "a+") as f:
        f.write(f"FLOPs: {flops}\n")
        f.write(f"Parameters: {params}\n")

    # Save training times and losses
    training_time_df = pd.DataFrame({"epoch": range(1, len(training_times) + 1), "training_time": training_times, "training_loss": training_losses})
    training_time_df.to_csv(training_metrics_file, index=False)

    # Save the trained model
    torch.save(model.state_dict(), model_file)

    # Evaluate the model
    mse, rmse, mae, r2, predictions = evaluate_model(model, X_test_tensor, y_test_tensor, device)
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R-squared: {r2}")
    
    
    # Save predictions (optional)
    # predictions_df = pd.DataFrame(predictions, columns=[f"Hour_{i+8}" for i in range(10)])  # Assuming hours 8-17
    # predictions_df.to_csv(predictions_file, index=False)
    
    # load dataset in dataframe
    df = pd.read_csv(os.path.join(data_directory, file_name), parse_dates=['DateTime'],usecols=['DateTime','MW'])
    # Filter data for the year 2018
    df_2018 = df[df["DateTime"].dt.year == 2018].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # rename columns
    df_2018.columns = ['DateTime', 'MW_actual']

    # # Add a new column for predictions (initialized to NaN)
    # df_2018["MW_predict"] = float("nan")
    
    # add the last dummy row to fill the predictions of the last day
    predictions = np.append(predictions, [0,0,0,0,0,0,0,0,0,0])
    predictions = predictions.flatten()
    # # Add the predictions to the DataFrame
    df_2018["MW_predict"] = predictions
    
    # # Save the DataFrame to a CSV file
    df_2018.to_csv(predictions_file, index=False)