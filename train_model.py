import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

class TelemetryDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DriverNN(nn.Module):
    def __init__(self, input_size):
        super(DriverNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # Output: 2 for continuous (steering, accel) + 5 for gear probabilities
        )
    
    def forward(self, x):
        output = self.network(x)
        # Split the output into continuous and discrete parts
        continuous = output[:, :2]  # First two outputs for steering and acceleration
        gear = output[:, 2:]  # Last 5 outputs for gear probabilities
        
        # Apply tanh to continuous outputs to bound them between -1 and 1
        continuous = torch.tanh(continuous)
        
        # Apply softmax to gear output to get probabilities for each gear
        gear = torch.softmax(gear, dim=1)
        
        return torch.cat([continuous, gear], dim=1)

def load_and_preprocess_data(data_dir):
    all_data = []
    all_labels = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            print(f"\nProcessing {filename}...")
            df = pd.read_csv(file_path)
            
            # Print initial shape
            print(f"Initial shape: {df.shape}")
            
            # Process track sensor data
            if 'Track' in df.columns:
                print("Processing Track sensor data...")
                # Split the track sensor string into individual values and take the first value
                df['Track'] = df['Track'].apply(lambda x: float(str(x).split()[0]) if isinstance(x, str) else x)
            
            # Extract features using the actual column names (removed WheelSpinVel)
            feature_columns = [
                'Track', 'Angle', 'Speed X', 'Speed Y', 'Speed Z',
                'TrackPos', 'RPM', 'Gear'
            ]
            
            # Extract labels using the actual column names
            label_columns = ['Steer', 'Acceleration', 'Gear']
            
            # Check which columns actually exist
            available_features = [col for col in feature_columns if col in df.columns]
            available_labels = [col for col in label_columns if col in df.columns]
            
            print(f"Available features: {available_features}")
            print(f"Available labels: {available_labels}")
            
            if not available_features or not available_labels:
                print(f"Warning: Missing required columns in {filename}")
                continue
            
            # Convert all columns to float
            for col in available_features + available_labels:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ensure gear values are in range 1-5
            df['Gear'] = df['Gear'].clip(1, 5)
            
            # Print number of NaN values in each column
            print("\nNaN values in each column:")
            print(df[available_features + available_labels].isna().sum())
            
            # Drop any rows with NaN values
            df = df.dropna(subset=available_features + available_labels)
            print(f"Shape after dropping NaN values: {df.shape}")
            
            if len(df) == 0:
                print(f"Warning: No valid data left in {filename} after cleaning")
                continue
            
            features = df[available_features].values
            labels = df[available_labels].values
            
            all_data.append(features)
            all_labels.append(labels)
            
            print(f"Added {len(features)} samples from {filename}")
    
    if not all_data:
        raise ValueError("No valid data found in any CSV files")
    
    # Combine all data
    X = np.vstack(all_data)
    y = np.vstack(all_labels)
    
    print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, scaler

def custom_loss(outputs, targets):
    # Split outputs into continuous and discrete parts
    continuous_outputs = outputs[:, :2]  # steering and acceleration
    gear_outputs = outputs[:, 2:]  # gear probabilities
    
    # Split targets similarly
    continuous_targets = targets[:, :2]
    gear_targets = targets[:, 2].long() - 1  # Convert gear to 0-based index for cross entropy
    
    # MSE loss for continuous outputs
    continuous_loss = nn.MSELoss()(continuous_outputs, continuous_targets)
    
    # Cross entropy loss for gear
    gear_loss = nn.CrossEntropyLoss()(gear_outputs, gear_targets)
    
    # Combine losses (you can adjust the weights if needed)
    total_loss = continuous_loss + gear_loss
    return total_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_driver_model.pth')

def main():
    # Load and preprocess data
    print("Starting data loading and preprocessing...")
    X, y, scaler = load_and_preprocess_data('data')
    
    print("\nSplitting data into train and validation sets...")
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    
    # Create datasets and dataloaders
    train_dataset = TelemetryDataset(X_train, y_train)
    val_dataset = TelemetryDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Initialize model
    input_size = X.shape[1]
    print(f"\nInitializing model with input size: {input_size}")
    model = DriverNN(input_size)
    
    # Loss function and optimizer
    criterion = custom_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nStarting training...")
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer)
    
    # Save the scaler for future use
    print("\nSaving scaler...")
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    print("Done!")

if __name__ == "__main__":
    main() 