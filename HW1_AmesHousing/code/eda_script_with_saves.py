# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Create figures directory if it doesn't exist
os.makedirs('report/figures', exist_ok=True)

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Set style for visualizations
plt.style.use('seaborn')
sns.set_palette('husl')

# Load the training data
df = pd.read_csv('data/ames_housing_train.csv')

# 1. Missing Values Analysis
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
missing_percent = (missing_values / len(df)) * 100

if len(missing_values) > 0:
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(missing_values)), missing_percent)
    plt.xticks(range(len(missing_values)), missing_values.index, rotation=45, ha='right')
    plt.title('Percentage of Missing Values by Feature')
    plt.xlabel('Features')
    plt.ylabel('Percentage of Missing Values')
    plt.tight_layout()
    plt.savefig('report/figures/missing_values.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Sale Price Distribution Analysis
plt.figure(figsize=(15, 5))

# Distribution plot
plt.subplot(1, 3, 1)
sns.histplot(df['SalePrice'], kde=True)
plt.title('Distribution of Sale Price')

# Q-Q plot
plt.subplot(1, 3, 2)
stats.probplot(df['SalePrice'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Sale Price')

# Box plot
plt.subplot(1, 3, 3)
sns.boxplot(y=df['SalePrice'])
plt.title('Box Plot of Sale Price')

plt.tight_layout()
plt.savefig('report/figures/sale_price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Numerical Features Analysis
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlations = df[numerical_cols].corr()['SalePrice'].sort_values(ascending=False)

# Scatter plots for top correlated features
plt.figure(figsize=(15, 10))
top_6_features = correlations[1:7].index

for i, feature in enumerate(top_6_features, 1):
    plt.subplot(2, 3, i)
    plt.scatter(df[feature], df['SalePrice'], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(f'SalePrice vs {feature}')
plt.tight_layout()
plt.savefig('report/figures/feature_vs_price.png', dpi=300, bbox_inches='tight')
plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 8))
top_corr_features = correlations[:10].index
correlation_matrix = df[top_corr_features].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Top Features')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('report/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Categorical Features Analysis
important_cats = ['OverallQual', 'Neighborhood', 'ExterQual', 'KitchenQual', 'GarageType', 'MSZoning']
important_cats = [col for col in important_cats if col in df.columns]

plt.figure(figsize=(15, 10))
for i, col in enumerate(important_cats[:6], 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=df, x=col, y='SalePrice')
    plt.xticks(rotation=45)
    plt.title(f'SalePrice by {col}')
plt.tight_layout()
plt.savefig('report/figures/categorical_features_dist.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Numerical Features Distribution
num_features = ['GrLivArea', 'LotArea', 'TotalBsmtSF', 'GarageArea', '1stFlrSF', 'YearBuilt']
num_features = [col for col in num_features if col in df.columns]

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('report/figures/numerical_features_dist.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Outlier Analysis
important_features = ['SalePrice', 'GrLivArea', 'LotArea', 'TotalBsmtSF']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(important_features, 1):
    if feature in df.columns:
        plt.subplot(2, 2, i)
        sns.boxplot(x=df[feature])
        plt.title(f'Box Plot of {feature}')
plt.tight_layout()
plt.savefig('report/figures/outliers_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Feature Importance Analysis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Prepare data for feature importance
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
numerical_features = numerical_features.drop('SalePrice')

X = df[numerical_features]
y = df['SalePrice']

# Fill missing values with median
X = X.fillna(X.median())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

# Plot feature importance
importance = pd.DataFrame({
    'feature': numerical_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=importance.head(15), x='importance', y='feature')
plt.title('Top 15 Most Important Features')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.savefig('report/figures/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("All figures have been generated and saved in the report/figures directory.") 