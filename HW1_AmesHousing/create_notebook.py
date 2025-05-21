import json
import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add markdown title cell
title_cell = nbf.v4.new_markdown_cell("""# Ames Housing Dataset Modeling

This notebook contains model development for the Ames Housing dataset, including:
- Missing value imputation
- Data preprocessing
- Train/test splitting
- 10-fold cross validation with multiple models:
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - PyTorch Neural Network""")

# Add imports cell
imports_cell = nbf.v4.new_code_cell("""# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)""")

# Add data loading section
data_loading_title = nbf.v4.new_markdown_cell("## 1. Data Loading")
data_loading_cell = nbf.v4.new_code_cell("""# Load the dataset (adjust path as needed)
try:
    df = pd.read_csv('data/train.csv')
except FileNotFoundError:
    # Try alternative path
    try:
        df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Please ensure the Ames Housing dataset (train.csv) is in the current directory or data/ folder")

# Display basic information
print(f"Dataset shape: {df.shape}")
df.head()""")

missing_values_cell = nbf.v4.new_code_cell("""# Check for missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print(f"Number of features with missing values: {len(missing_values)}")
missing_values""")

# Add missing value imputation section
imputation_title = nbf.v4.new_markdown_cell("## 2. Missing Values Imputation")
imputation_cell = nbf.v4.new_code_cell("""# Identify numeric and categorical columns
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove target variable from features
if 'SalePrice' in numeric_features:
    numeric_features.remove('SalePrice')

# Create imputers for numeric and categorical data
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Apply imputation
df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])
df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])

# Verify no missing values remain
print(f"Missing values after imputation: {df.isnull().sum().sum()}")""")

# Add preprocessing section
preprocessing_title = nbf.v4.new_markdown_cell("## 3. Data Preprocessing")
preprocessing_cell = nbf.v4.new_code_cell("""# Prepare the target variable
y = df['SalePrice']

# Log transform the target variable (common for house prices)
y = np.log1p(y)

# Prepare the feature matrix
X = df.drop('SalePrice', axis=1)

# Create a preprocessor for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])""")

# Add train-test split section
split_title = nbf.v4.new_markdown_cell("## 4. Train-Test Split")
split_cell = nbf.v4.new_code_cell("""# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")""")

# Add model training section
model_title = nbf.v4.new_markdown_cell("## 5. Model Training with 10-fold Cross-Validation")

# Ridge regression
ridge_title = nbf.v4.new_markdown_cell("### 5.1 Ridge Regression")
ridge_cell = nbf.v4.new_code_cell("""# Create Ridge Regression pipeline
ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=1.0))
])

# Set up 10-fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation
ridge_cv_scores = cross_val_score(ridge_pipeline, X_train, y_train, cv=kf, 
                                scoring='neg_mean_squared_error')

# Convert negative MSE to RMSE
ridge_rmse_scores = np.sqrt(-ridge_cv_scores)

print("Ridge Regression 10-fold CV Results:")
print(f"Mean RMSE: {ridge_rmse_scores.mean():.4f}")
print(f"Std RMSE: {ridge_rmse_scores.std():.4f}")

# Train on the full training set
ridge_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred_ridge = ridge_pipeline.predict(X_test)

# Calculate metrics on test set
ridge_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
ridge_test_r2 = r2_score(y_test, y_pred_ridge)

print(f"\\nTest RMSE: {ridge_test_rmse:.4f}")
print(f"Test R²: {ridge_test_r2:.4f}")""")

# Lasso regression
lasso_title = nbf.v4.new_markdown_cell("### 5.2 Lasso Regression")
lasso_cell = nbf.v4.new_code_cell("""# Create Lasso Regression pipeline
lasso_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Lasso(alpha=0.001))
])

# Perform cross-validation
lasso_cv_scores = cross_val_score(lasso_pipeline, X_train, y_train, cv=kf, 
                                 scoring='neg_mean_squared_error')

# Convert negative MSE to RMSE
lasso_rmse_scores = np.sqrt(-lasso_cv_scores)

print("Lasso Regression 10-fold CV Results:")
print(f"Mean RMSE: {lasso_rmse_scores.mean():.4f}")
print(f"Std RMSE: {lasso_rmse_scores.std():.4f}")

# Train on the full training set
lasso_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lasso = lasso_pipeline.predict(X_test)

# Calculate metrics on test set
lasso_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
lasso_test_r2 = r2_score(y_test, y_pred_lasso)

print(f"\\nTest RMSE: {lasso_test_rmse:.4f}")
print(f"Test R²: {lasso_test_r2:.4f}")""")

# Random Forest
rf_title = nbf.v4.new_markdown_cell("### 5.3 Random Forest")
rf_cell = nbf.v4.new_code_cell("""# Create Random Forest pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Perform cross-validation
rf_cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=kf, 
                              scoring='neg_mean_squared_error')

# Convert negative MSE to RMSE
rf_rmse_scores = np.sqrt(-rf_cv_scores)

print("Random Forest 10-fold CV Results:")
print(f"Mean RMSE: {rf_rmse_scores.mean():.4f}")
print(f"Std RMSE: {rf_rmse_scores.std():.4f}")

# Train on the full training set
rf_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_pipeline.predict(X_test)

# Calculate metrics on test set
rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_test_r2 = r2_score(y_test, y_pred_rf)

print(f"\\nTest RMSE: {rf_test_rmse:.4f}")
print(f"Test R²: {rf_test_r2:.4f}")""")

# PyTorch Neural Network
nn_title = nbf.v4.new_markdown_cell("### 5.4 PyTorch Neural Network")
nn_model_cell = nbf.v4.new_code_cell("""# First, we need to preprocess the data for PyTorch
# Apply preprocessing to the full dataset
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Define the Neural Network model
class HousingNN(nn.Module):
    def __init__(self, input_dim):
        super(HousingNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x""")

nn_train_cell = nbf.v4.new_code_cell("""# Function to train the neural network
def train_nn(X_train, y_train, input_dim, epochs=100, batch_size=32):
    # Convert to PyTorch tensors
    # Handle sparse matrix by converting to dense array first
    from scipy import sparse
    if sparse.issparse(X_train):
        X_train_tensor = torch.FloatTensor(X_train.toarray())
    else:
        X_train_tensor = torch.FloatTensor(X_train)
        
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = HousingNN(input_dim)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
    
    return model

# Function to evaluate the model
def evaluate_nn(model, X_test, y_test):
    # Handle sparse matrix
    from scipy import sparse
    if sparse.issparse(X_test):
        X_test_tensor = torch.FloatTensor(X_test.toarray())
    else:
        X_test_tensor = torch.FloatTensor(X_test)
        
    y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        mse = nn.MSELoss()(y_pred, y_test_tensor).item()
        rmse = np.sqrt(mse)
        
        # Convert to numpy for R² calculation
        y_pred_np = y_pred.numpy().flatten()
        y_test_np = y_test.values
        r2 = r2_score(y_test_np, y_pred_np)
        
    return rmse, r2""")

nn_cv_cell = nbf.v4.new_code_cell("""# Perform 10-fold cross-validation with the Neural Network
kf = KFold(n_splits=10, shuffle=True, random_state=42)
nn_rmse_scores = []

input_dim = X_train_preprocessed.shape[1]

print("PyTorch Neural Network 10-fold CV Results:")
for i, (train_idx, val_idx) in enumerate(kf.split(X_train_preprocessed)):
    # Split data
    X_fold_train, X_fold_val = X_train_preprocessed[train_idx], X_train_preprocessed[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train model
    model = train_nn(X_fold_train, y_fold_train, input_dim, epochs=50, batch_size=32)
    
    # Evaluate model
    from scipy import sparse
    if sparse.issparse(X_fold_val):
        X_fold_val_tensor = torch.FloatTensor(X_fold_val.toarray())
    else:
        X_fold_val_tensor = torch.FloatTensor(X_fold_val)
        
    y_fold_val_tensor = torch.FloatTensor(y_fold_val.values).reshape(-1, 1)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_fold_val_tensor)
        mse = nn.MSELoss()(y_pred, y_fold_val_tensor).item()
        fold_rmse = np.sqrt(mse)
    
    nn_rmse_scores.append(fold_rmse)
    print(f"Fold {i+1} RMSE: {fold_rmse:.4f}")

print(f"\\nMean RMSE: {np.mean(nn_rmse_scores):.4f}")
print(f"Std RMSE: {np.std(nn_rmse_scores):.4f}")

# Train on the full training set
final_nn_model = train_nn(X_train_preprocessed, y_train, input_dim, epochs=50, batch_size=32)

# Evaluate on test set
nn_test_rmse, nn_test_r2 = evaluate_nn(final_nn_model, X_test_preprocessed, y_test)

print(f"\\nTest RMSE: {nn_test_rmse:.4f}")
print(f"Test R²: {nn_test_r2:.4f}")""")

# Add model comparison section
comparison_title = nbf.v4.new_markdown_cell("## 6. Model Comparison")
comparison_cell = nbf.v4.new_code_cell("""# Create a comparison table of model performances
models = ['Ridge Regression', 'Lasso Regression', 'Random Forest', 'PyTorch Neural Network']
cv_rmse = [ridge_rmse_scores.mean(), lasso_rmse_scores.mean(), rf_rmse_scores.mean(), np.mean(nn_rmse_scores)]
test_rmse = [ridge_test_rmse, lasso_test_rmse, rf_test_rmse, nn_test_rmse]
test_r2 = [ridge_test_r2, lasso_test_r2, rf_test_r2, nn_test_r2]

comparison_df = pd.DataFrame({
    'Model': models,
    'CV RMSE': cv_rmse,
    'Test RMSE': test_rmse,
    'Test R²': test_r2
})

# Sort by test RMSE (lower is better)
comparison_df = comparison_df.sort_values('Test RMSE')
comparison_df""")

visualization_cell = nbf.v4.new_code_cell("""# Visualize model comparison
plt.figure(figsize=(12, 5))

# RMSE comparison
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Test RMSE', data=comparison_df)
plt.title('Model Comparison - Test RMSE')
plt.xticks(rotation=45)
plt.tight_layout()

# R² comparison
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='Test R²', data=comparison_df)
plt.title('Model Comparison - Test R²')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()""")

# Add conclusion section
conclusion_title = nbf.v4.new_markdown_cell("""## 7. Conclusion

In this notebook, we built and evaluated four different regression models for the Ames Housing dataset:

1. Ridge Regression
2. Lasso Regression
3. Random Forest Regression
4. PyTorch Neural Network

We performed proper data preprocessing, including missing value imputation, and evaluated each model using 10-fold cross-validation. The comparison table and visualization help identify the best performing model based on RMSE and R² metrics.

The best performing model can be used for predicting house prices on new data.""")

# Add cells to notebook
nb.cells.extend([
    title_cell, imports_cell, 
    data_loading_title, data_loading_cell, missing_values_cell,
    imputation_title, imputation_cell,
    preprocessing_title, preprocessing_cell,
    split_title, split_cell,
    model_title, 
    ridge_title, ridge_cell,
    lasso_title, lasso_cell,
    rf_title, rf_cell,
    nn_title, nn_model_cell, nn_train_cell, nn_cv_cell,
    comparison_title, comparison_cell, visualization_cell,
    conclusion_title
])

# Write the notebook to a file
with open('modeling.ipynb', 'w') as f:
    nbf.write(nb, f) 