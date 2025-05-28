import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Load the training data
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

# Preprocess the data
X = preprocess_data(data)
y = data['SalePrice']

# Initialize KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Define lambda values to try
lambda_values = np.logspace(-6, 6, 60)  # Creates 20 evenly spaced values on log scale from 10^-6 to 10^6
mean_test_rmse_scores = []

# For each lambda value
for lambda_val in lambda_values:
    # Initialize lists to store performance metrics for this lambda
    test_rmse_scores = []
    
    # Perform 10-fold cross validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        # Split data for this fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Ridge model with current lambda
        ridge = Ridge(alpha=lambda_val)
        ridge.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_test = ridge.predict(X_test_scaled)
        
        # Calculate RMSE
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_rmse_scores.append(test_rmse)
    
    # Calculate mean RMSE across folds for this lambda
    mean_test_rmse = np.mean(test_rmse_scores)
    mean_test_rmse_scores.append(mean_test_rmse)
    print(f"Lambda = {lambda_val:.6f}, Mean Test RMSE = {mean_test_rmse:.2f}")

# Find the best lambda
best_lambda_idx = np.argmin(mean_test_rmse_scores)
best_lambda = lambda_values[best_lambda_idx]
best_rmse = mean_test_rmse_scores[best_lambda_idx]

print(f"\nBest Lambda: {best_lambda:.6f}")
print(f"Best RMSE: {best_rmse:.2f}")

# Plot RMSE vs Lambda
plt.figure(figsize=(10, 6))
plt.semilogx(lambda_values, mean_test_rmse_scores, '-o')
plt.plot(best_lambda, best_rmse, 'r*', markersize=15, label=f'Best λ = {best_lambda:.6f}')
plt.xlabel('Lambda (α)')
plt.ylabel('Mean Test RMSE')
plt.title('Ridge Regression: RMSE vs Lambda')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('ridge_lambda_vs_rmse.png')
plt.close() 