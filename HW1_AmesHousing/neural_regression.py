import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess data
data = pd.read_csv('data/ames_housing_train.csv')

def preprocess_data(df):
    # Separate features that are numeric
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_features = numeric_features.drop('SalePrice') if 'SalePrice' in numeric_features else numeric_features
    
    # Handle missing values in numeric features
    X_numeric = df[numeric_features].fillna(df[numeric_features].mean())
    
    # Get categorical features
    categorical_features = df.select_dtypes(include=['object']).columns
    
    # Create dummy variables for categorical features
    X_categorical = pd.get_dummies(df[categorical_features], drop_first=True)
    
    # Combine numeric and categorical features
    X = pd.concat([X_numeric, X_categorical], axis=1)
    
    return X

# Custom Dataset class
class AmesHousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values.reshape(-1, 1))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Neural Network Model
class HousingNN(nn.Module):
    def __init__(self, input_dim):
        super(HousingNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Preprocess data
X = preprocess_data(data)
y = data['SalePrice']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Create datasets
train_dataset = AmesHousingDataset(X_train_scaled, y_train)
val_dataset = AmesHousingDataset(X_val_scaled, y_val)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = HousingNN(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 100
train_losses = []
val_losses = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item()
    
    # Calculate average losses
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Calculate RMSE, MSE and R² on validation set
model.eval()
with torch.no_grad():
    val_predictions = model(torch.FloatTensor(X_val_scaled))
    val_predictions_np = val_predictions.numpy()
    val_true_np = y_val.values.reshape(-1, 1)
    
    # Calculate MSE
    val_mse = criterion(val_predictions, torch.FloatTensor(val_true_np)).item()
    
    # Calculate RMSE
    val_rmse = np.sqrt(val_mse)
    
    # Calculate R²
    val_r2 = 1 - (np.sum((val_true_np - val_predictions_np) ** 2) / np.sum((val_true_np - np.mean(val_true_np)) ** 2))
    
    print('\nValidation Metrics:')
    print(f'MSE: ${val_mse:.2f}')
    print(f'RMSE: ${val_rmse:.2f}')
    print(f'R²: {val_r2:.4f}')

# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('neural_network_training.png')
plt.close()

# Save the model
torch.save(model.state_dict(), 'ames_housing_nn_model.pth') 