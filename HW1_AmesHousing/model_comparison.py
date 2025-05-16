import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
from neural_regression import HousingNN, preprocess_data

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load and preprocess data
data = pd.read_csv('data/ames_housing_train.csv')
X = preprocess_data(data)
y = data['SalePrice']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"MSE: ${mse:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    return {'model': model_name, 'mse': mse, 'rmse': rmse, 'r2': r2}

# 1. Evaluate Ridge Regression (L2)
ridge = Ridge(alpha=1.0)  # Using default alpha
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)
ridge_metrics = evaluate_model(y_test, ridge_pred, "Ridge Regression (L2)")

# 2. Evaluate Lasso Regression (L1)
lasso = Lasso(alpha=0.001)  # Using default alpha
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)
lasso_metrics = evaluate_model(y_test, lasso_pred, "Lasso Regression (L1)")

# 3. Evaluate Neural Network
# Load the trained model
nn_model = HousingNN(input_dim=X_train.shape[1])
nn_model.load_state_dict(torch.load('ames_housing_nn_model.pth'))
nn_model.eval()

with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    nn_pred = nn_model(X_test_tensor).numpy()
    nn_metrics = evaluate_model(y_test, nn_pred, "Neural Network")

# Collect all metrics
all_metrics = [ridge_metrics, lasso_metrics, nn_metrics]
metrics_df = pd.DataFrame(all_metrics)

# Find the best model for each metric
best_mse = metrics_df.loc[metrics_df['mse'].idxmin()]
best_rmse = metrics_df.loc[metrics_df['rmse'].idxmin()]
best_r2 = metrics_df.loc[metrics_df['r2'].idxmax()]

print("\nBest Models by Metric:")
print(f"Best MSE: {best_mse['model']} (${best_mse['mse']:.2f})")
print(f"Best RMSE: {best_rmse['model']} (${best_rmse['rmse']:.2f})")
print(f"Best R²: {best_r2['model']} ({best_r2['r2']:.4f})")

# Create comparison plots
plt.figure(figsize=(15, 5))

# Plot 1: RMSE Comparison
plt.subplot(131)
plt.bar([m['model'] for m in all_metrics], [m['rmse'] for m in all_metrics])
plt.title('RMSE Comparison')
plt.xticks(rotation=45)
plt.ylabel('RMSE ($)')

# Plot 2: MSE Comparison
plt.subplot(132)
plt.bar([m['model'] for m in all_metrics], [m['mse'] for m in all_metrics])
plt.title('MSE Comparison')
plt.xticks(rotation=45)
plt.ylabel('MSE ($)')

# Plot 3: R² Comparison
plt.subplot(133)
plt.bar([m['model'] for m in all_metrics], [m['r2'] for m in all_metrics])
plt.title('R² Comparison')
plt.xticks(rotation=45)
plt.ylabel('R²')

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Create scatter plots of predicted vs actual values
plt.figure(figsize=(15, 5))

# Ridge
plt.subplot(131)
plt.scatter(y_test, ridge_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Ridge: Predicted vs Actual')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')

# Lasso
plt.subplot(132)
plt.scatter(y_test, lasso_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Lasso: Predicted vs Actual')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')

# Neural Network
plt.subplot(133)
plt.scatter(y_test, nn_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Neural Network: Predicted vs Actual')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')

plt.tight_layout()
plt.savefig('prediction_comparison.png')
plt.close() 