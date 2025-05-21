import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy import sparse
import os

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Make sure directories exist
os.makedirs('report/figures', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Load the dataset
def load_data():
    try:
        df = pd.read_csv('data/ames_housing_train.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('ames_housing_train.csv')
        except FileNotFoundError:
            try:
                df = pd.read_csv('data/train.csv')
            except FileNotFoundError:
                try:
                    df = pd.read_csv('train.csv')
                except FileNotFoundError:
                    print("Please ensure the Ames Housing dataset is in the current directory or data/ folder")
                    return None
    
    print(f"Dataset loaded with shape: {df.shape}")
    return df

# Data preprocessing
def preprocess_data(df):
    # Variables to remove as specified
    vars_to_remove = ["PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu"]
    
    # Remove specified variables
    df = df.drop(columns=vars_to_remove, errors='ignore')
    
    # Identify numeric and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from features list if it exists
    if 'SalePrice' in numeric_features:
        numeric_features.remove('SalePrice')
    
    # Print information about the preprocessing
    print(f"Removed variables: {vars_to_remove}")
    print(f"Number of numerical features: {len(numeric_features)}")
    print(f"Number of categorical features: {len(categorical_features)}")
    
    # Return processed data and feature lists
    return df, numeric_features, categorical_features

# Create feature preprocessing pipeline
def create_preprocessor(numeric_features, categorical_features):
    # Create transformers for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

# Ridge Regression model with cross-validation and lambda tuning
def ridge_regression(X_train, y_train, X_test, y_test):
    # Lambda values to try
    alphas = np.logspace(-3, 6, 20)
    
    # Store results
    cv_rmse_scores = []
    test_rmse_scores = []
    
    # Create k-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Test each alpha value
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        
        # Cross-validation score
        cv_scores = -cross_val_score(ridge, X_train, y_train, cv=kf,
                                     scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(cv_scores).mean()
        cv_rmse_scores.append(cv_rmse)
        
        # Test set score
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_rmse_scores.append(test_rmse)
    
    # Find best alpha
    best_alpha_idx = np.argmin(cv_rmse_scores)
    best_alpha = alphas[best_alpha_idx]
    best_cv_rmse = cv_rmse_scores[best_alpha_idx]
    
    print(f"Ridge Regression Best Alpha: {best_alpha:.2f}")
    print(f"Ridge Regression Best CV RMSE: {best_cv_rmse:.2f}")
    
    # Plot RMSE vs lambda
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, cv_rmse_scores, '-o', label='Cross-validation RMSE')
    plt.semilogx(alphas, test_rmse_scores, '-o', label='Test RMSE')
    plt.axvline(best_alpha, color='red', linestyle='--', label=f'Best λ = {best_alpha:.2f}')
    plt.xlabel('λ (Alpha)')
    plt.ylabel('RMSE')
    plt.title('Ridge Regression: RMSE vs λ')
    plt.legend()
    plt.grid(True)
    plt.savefig('report/figures/ridge_lambda_vs_rmse.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/ridge_lambda_vs_rmse.png', dpi=300, bbox_inches='tight')
    
    # Train final model with best alpha
    final_ridge = Ridge(alpha=best_alpha)
    final_ridge.fit(X_train, y_train)
    
    # Get feature coefficients
    return final_ridge, best_alpha, best_cv_rmse

# Lasso Regression model with cross-validation and lambda tuning
def lasso_regression(X_train, y_train, X_test, y_test):
    # Lambda values to try
    alphas = np.logspace(-3, 2, 20)
    
    # Store results
    cv_rmse_scores = []
    test_rmse_scores = []
    
    # Create k-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Test each alpha value
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        
        # Cross-validation score
        cv_scores = -cross_val_score(lasso, X_train, y_train, cv=kf,
                                     scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(cv_scores).mean()
        cv_rmse_scores.append(cv_rmse)
        
        # Test set score
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_rmse_scores.append(test_rmse)
    
    # Find best alpha
    best_alpha_idx = np.argmin(cv_rmse_scores)
    best_alpha = alphas[best_alpha_idx]
    best_cv_rmse = cv_rmse_scores[best_alpha_idx]
    
    print(f"Lasso Regression Best Alpha: {best_alpha:.2f}")
    print(f"Lasso Regression Best CV RMSE: {best_cv_rmse:.2f}")
    
    # Plot RMSE vs lambda
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, cv_rmse_scores, '-o', label='Cross-validation RMSE')
    plt.semilogx(alphas, test_rmse_scores, '-o', label='Test RMSE')
    plt.axvline(best_alpha, color='red', linestyle='--', label=f'Best λ = {best_alpha:.2f}')
    plt.xlabel('λ (Alpha)')
    plt.ylabel('RMSE')
    plt.title('Lasso Regression: RMSE vs λ')
    plt.legend()
    plt.grid(True)
    plt.savefig('report/figures/lasso_lambda_vs_rmse.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/lasso_lambda_vs_rmse.png', dpi=300, bbox_inches='tight')
    
    # Train final model with best alpha
    final_lasso = Lasso(alpha=best_alpha, max_iter=10000)
    final_lasso.fit(X_train, y_train)
    
    return final_lasso, best_alpha, best_cv_rmse

# Random Forest model with cross-validation
def random_forest(X_train, y_train, X_test, y_test):
    # Create RF model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Create k-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_scores = -cross_val_score(rf, X_train, y_train, cv=kf,
                                scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(cv_scores).mean()
    
    print(f"Random Forest CV RMSE: {cv_rmse:.2f}")
    
    # Train final model
    rf.fit(X_train, y_train)
    
    # Test set performance
    y_pred = rf.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest Test RMSE: {test_rmse:.2f}")
    print(f"Random Forest Test R²: {test_r2:.4f}")
    
    return rf, cv_rmse, test_rmse, test_r2

# Neural Network model class
class HousingNN(nn.Module):
    def __init__(self, input_dim):
        super(HousingNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

# Function to train the neural network
def train_nn(X_train, y_train, input_dim, epochs=100, batch_size=32, learning_rate=0.0001):
    # Convert to PyTorch tensors
    # Handle sparse matrix
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    
    # To track training progress
    epoch_losses = []
    
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
            
        epoch_loss = running_loss/len(train_loader)
        epoch_losses.append(epoch_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
    
    # Plot the training progress
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Neural Network Training Progress')
    plt.grid(True)
    plt.savefig('report/figures/neural_network_training.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/neural_network_training.png', dpi=300, bbox_inches='tight')
    
    return model, epoch_losses

# Function to evaluate the neural network model
def evaluate_nn(model, X_test, y_test):
    # Handle sparse matrix
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
        
    return rmse, r2

# PyTorch Neural Network with cross-validation
def pytorch_neural_network(X_train_processed, y_train, X_test_processed, y_test):
    input_dim = X_train_processed.shape[1]
    
    # Create k-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Perform cross-validation
    nn_rmse_scores = []
    
    print("PyTorch Neural Network 10-fold CV:")
    for i, (train_idx, val_idx) in enumerate(kf.split(X_train_processed)):
        X_fold_train, X_fold_val = X_train_processed[train_idx], X_train_processed[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train model (using fewer epochs for CV)
        model, _ = train_nn(X_fold_train, y_fold_train, input_dim, epochs=50, batch_size=32, learning_rate=0.0001)
        
        # Evaluate model
        fold_rmse, _ = evaluate_nn(model, X_fold_val, y_fold_val)
        
        nn_rmse_scores.append(fold_rmse)
        print(f"Fold {i+1} RMSE: {fold_rmse:.2f}")
    
    cv_rmse = np.mean(nn_rmse_scores)
    print(f"Neural Network CV RMSE: {cv_rmse:.2f}")
    
    # Train final model on full training set
    final_model, _ = train_nn(X_train_processed, y_train, input_dim, epochs=100, batch_size=32, learning_rate=0.0001)
    
    # Evaluate on test set
    test_rmse, test_r2 = evaluate_nn(final_model, X_test_processed, y_test)
    print(f"Neural Network Test RMSE: {test_rmse:.2f}")
    print(f"Neural Network Test R²: {test_r2:.4f}")
    
    return final_model, cv_rmse, test_rmse, test_r2

# Plot feature importance for linear models
def plot_feature_importance(model, feature_names, model_name, top_n=10):
    # Get coefficients
    coefs = model.coef_
    
    # Check for feature name and coefficient length mismatch
    if len(feature_names) != len(coefs):
        print(f"Warning: Feature names ({len(feature_names)}) and coefficients ({len(coefs)}) have different lengths")
        # Create generic feature names if needed
        if len(feature_names) < len(coefs):
            feature_names = [f"Feature_{i}" for i in range(len(coefs))]
        else:
            feature_names = feature_names[:len(coefs)]
    
    # Create DataFrame for visualization
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
    
    # Sort by absolute coefficient values
    coef_df['AbsCoef'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('AbsCoef', ascending=False).head(top_n)
    
    # Plot the top features
    plt.figure(figsize=(12, 6))
    colors = ['blue' if c > 0 else 'red' for c in coef_df['Coefficient']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Features by Importance - {model_name}')
    plt.grid(axis='x')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'report/figures/{model_name.lower().replace(" ", "_")}_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'figures/{model_name.lower().replace(" ", "_")}_feature_importance.png', dpi=300, bbox_inches='tight')

# Create comparison figure for all models
def plot_model_comparison(models_data):
    # Extract metrics
    models = [data[0] for data in models_data]
    cv_rmse = [data[1] for data in models_data]
    test_rmse = [data[2] for data in models_data]
    test_r2 = [data[3] for data in models_data]
    
    # Create DataFrame for visualization
    comparison_df = pd.DataFrame({
        'Model': models,
        'CV RMSE': cv_rmse,
        'Test RMSE': test_rmse,
        'Test R²': test_r2
    })
    
    # Sort by test RMSE
    comparison_df = comparison_df.sort_values('Test RMSE')
    
    # Print the comparison table
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Plot RMSE comparison
    plt.figure(figsize=(12, 8))
    
    # RMSE subplot
    plt.subplot(2, 1, 1)
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, cv_rmse, width, label='CV RMSE')
    plt.bar(x + width/2, test_rmse, width, label='Test RMSE')
    plt.xticks(x, models)
    plt.ylabel('RMSE')
    plt.title('Model Comparison - RMSE (lower is better)')
    plt.legend()
    plt.grid(axis='y')
    
    # R² subplot
    plt.subplot(2, 1, 2)
    plt.bar(x, test_r2)
    plt.xticks(x, models)
    plt.ylabel('R² Score')
    plt.title('Model Comparison - R² (higher is better)')
    plt.grid(axis='y')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('report/figures/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
    
    return comparison_df

# Main function to run all analyses
def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Preprocess data
    df, numeric_features, categorical_features = preprocess_data(df)
    
    # Prepare target variable
    y = df['SalePrice']
    X = df.drop('SalePrice', axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # Process data for models
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print("Data preprocessing complete.")
    
    # Create approximately correct feature names (we'll fix any mismatches in the plot function)
    feature_names = numeric_features.copy()
    
    # Add generic names for categorical features after one-hot encoding
    for i in range(X_train_processed.shape[1] - len(numeric_features)):
        feature_names.append(f"Categorical_{i}")
    
    # Ridge Regression
    print("\n--- Ridge Regression ---")
    ridge_model, ridge_alpha, ridge_cv_rmse = ridge_regression(X_train_processed, y_train, X_test_processed, y_test)
    
    # Get predictions for test set
    y_pred_ridge = ridge_model.predict(X_test_processed)
    ridge_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    ridge_test_r2 = r2_score(y_test, y_pred_ridge)
    print(f"Ridge Test RMSE: {ridge_test_rmse:.2f}")
    print(f"Ridge Test R²: {ridge_test_r2:.4f}")
    
    # Feature importance for Ridge
    plot_feature_importance(ridge_model, feature_names, "Ridge Regression")
    
    # Lasso Regression
    print("\n--- Lasso Regression ---")
    lasso_model, lasso_alpha, lasso_cv_rmse = lasso_regression(X_train_processed, y_train, X_test_processed, y_test)
    
    # Get predictions for test set
    y_pred_lasso = lasso_model.predict(X_test_processed)
    lasso_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
    lasso_test_r2 = r2_score(y_test, y_pred_lasso)
    print(f"Lasso Test RMSE: {lasso_test_rmse:.2f}")
    print(f"Lasso Test R²: {lasso_test_r2:.4f}")
    
    # Feature importance for Lasso
    plot_feature_importance(lasso_model, feature_names, "Lasso Regression")
    
    # Random Forest
    print("\n--- Random Forest ---")
    rf_model, rf_cv_rmse, rf_test_rmse, rf_test_r2 = random_forest(X_train_processed, y_train, X_test_processed, y_test)
    
    # PyTorch Neural Network
    print("\n--- PyTorch Neural Network ---")
    nn_model, nn_cv_rmse, nn_test_rmse, nn_test_r2 = pytorch_neural_network(X_train_processed, y_train, X_test_processed, y_test)
    
    # Model comparison
    models_data = [
        ('Ridge Regression', ridge_cv_rmse, ridge_test_rmse, ridge_test_r2),
        ('Lasso Regression', lasso_cv_rmse, lasso_test_rmse, lasso_test_r2),
        ('Random Forest', rf_cv_rmse, rf_test_rmse, rf_test_r2),
        ('Neural Network', nn_cv_rmse, nn_test_rmse, nn_test_r2)
    ]
    
    comparison_df = plot_model_comparison(models_data)
    
    print("Analysis complete. All figures saved to report/figures/ and figures/ directories.")

if __name__ == "__main__":
    main() 