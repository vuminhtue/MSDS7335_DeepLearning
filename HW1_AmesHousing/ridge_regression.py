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

# Initialize lists to store performance metrics
train_mse_scores = []
test_mse_scores = []
train_r2_scores = []
test_r2_scores = []
feature_importance_list = []

# Perform 10-fold cross validation
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    # Split data for this fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ridge model
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = ridge.predict(X_train_scaled)
    y_pred_test = ridge.predict(X_test_scaled)
    
    # Calculate performance metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Store metrics
    train_mse_scores.append(train_mse)
    test_mse_scores.append(test_mse)
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)
    
    # Store feature importance
    feature_importance_list.append(pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': np.abs(ridge.coef_)
    }))
    
    print(f"\nFold {fold} Results:")
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Test MSE: {test_mse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")

# Calculate and print mean and std of performance metrics
print("\nOverall Cross-Validation Results:")
print(f"Mean Training MSE: {np.mean(train_mse_scores):.2f} (±{np.std(train_mse_scores):.2f})")
print(f"Mean Test MSE: {np.mean(test_mse_scores):.2f} (±{np.std(test_mse_scores):.2f})")
print(f"Mean Training R²: {np.mean(train_r2_scores):.4f} (±{np.std(train_r2_scores):.4f})")
print(f"Mean Test R²: {np.mean(test_r2_scores):.4f} (±{np.std(test_r2_scores):.4f})")

# Calculate average feature importance across folds
avg_feature_importance = pd.concat(feature_importance_list).groupby('Feature')['Coefficient'].mean().reset_index()
top_10_features = avg_feature_importance.nlargest(10, 'Coefficient')

print("\nTop 10 Most Important Features (Averaged across folds):")
print(top_10_features)

# Plot average feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(top_10_features)), top_10_features['Coefficient'])
plt.xticks(range(len(top_10_features)), top_10_features['Feature'], rotation=45, ha='right')
plt.title('Top 10 Feature Importance (Ridge) - Averaged Across 10 Folds')
plt.tight_layout()
plt.savefig('ridge_feature_importance.png')
plt.close() 