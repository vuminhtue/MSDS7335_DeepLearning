# %% [markdown]
# # Exploratory Data Analysis - Ames Housing Dataset
# 
# This notebook performs a comprehensive exploratory data analysis on the Ames Housing training dataset.
# 
# ## Contents:
# 1. Data Loading and Initial Exploration
# 2. Missing Values Analysis
# 3. Target Variable Analysis
# 4. Numerical Features Analysis
# 5. Categorical Features Analysis
# 6. Feature Distributions
# 7. Outlier Analysis
# 8. Year-based Analysis

# %% [markdown]
# ## 1. Data Loading and Initial Exploration

# %%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Set style for visualizations
plt.style.use('seaborn')
sns.set_palette('husl')

# %%
# Load the training data
df = pd.read_csv('data/ames_housing_train.csv')
print('Dataset Shape:', df.shape)
df.head()

# %%
# Get basic information about the dataset
print("\nDataset Info:")
df.info()

# %%
# Get summary statistics
print("\nSummary Statistics:")
df.describe()

# %% [markdown]
# ## 2. Missing Values Analysis

# %%
# Calculate missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

# Calculate percentage of missing values
missing_percent = (missing_values / len(df)) * 100

# Create a DataFrame with missing value information
missing_info = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percent
})

print("Features with missing values:")
print(missing_info)

# Visualize missing values
if len(missing_values) > 0:
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(missing_values)), missing_percent)
    plt.xticks(range(len(missing_values)), missing_values.index, rotation=45, ha='right')
    plt.title('Percentage of Missing Values by Feature')
    plt.xlabel('Features')
    plt.ylabel('Percentage of Missing Values')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 3. Target Variable Analysis

# %%
# Analyze the target variable (SalePrice)
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
plt.show()

# Print summary statistics and skewness
print("\nSale Price Summary Statistics:")
print(df['SalePrice'].describe())
print(f"\nSkewness: {df['SalePrice'].skew():.2f}")
print(f"Kurtosis: {df['SalePrice'].kurtosis():.2f}")

# %% [markdown]
# ## 4. Numerical Features Analysis

# %%
# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Calculate correlations with SalePrice
correlations = df[numerical_cols].corr()['SalePrice'].sort_values(ascending=False)

print("Top 10 features correlated with SalePrice:")
print(correlations[:11])  # 11 because SalePrice will be included

# Create scatter plots for top 6 correlated numerical features
plt.figure(figsize=(15, 10))
top_6_features = correlations[1:7].index  # Exclude SalePrice itself

for i, feature in enumerate(top_6_features, 1):
    plt.subplot(2, 3, i)
    plt.scatter(df[feature], df['SalePrice'], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(f'SalePrice vs {feature}')
plt.tight_layout()
plt.show()

# %%
# Create correlation heatmap for top correlated features
plt.figure(figsize=(12, 8))
top_corr_features = correlations[:10].index
correlation_matrix = df[top_corr_features].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Top Features')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Categorical Features Analysis

# %%
# Select important categorical features
important_cats = ['OverallQual', 'Neighborhood', 'ExterQual', 'KitchenQual', 'GarageType', 'MSZoning']
important_cats = [col for col in important_cats if col in df.columns]

# Create box plots for categorical features vs SalePrice
plt.figure(figsize=(15, 10))
for i, col in enumerate(important_cats[:6], 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=df, x=col, y='SalePrice')
    plt.xticks(rotation=45)
    plt.title(f'SalePrice by {col}')
plt.tight_layout()
plt.show()

# Print value counts for categorical features
print("\nValue counts for important categorical features:")
for col in important_cats:
    print(f"\n{col}:")
    print(df[col].value_counts())

# %% [markdown]
# ## 6. Feature Distributions

# %%
# Select numerical features for distribution analysis
num_features = ['GrLivArea', 'LotArea', 'TotalBsmtSF', 'GarageArea', '1stFlrSF', 'YearBuilt']
num_features = [col for col in num_features if col in df.columns]

# Create distribution plots
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary statistics for key numerical features:")
print(df[num_features].describe())

# %% [markdown]
# ## 7. Outlier Analysis

# %%
def plot_boxplot_with_outliers(df, column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.show()
    
    # Calculate outlier boundaries
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    
    print(f"\nOutlier Analysis for {column}:")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Percentage of outliers: {(len(outliers)/len(df))*100:.2f}%")
    print(f"Outlier boundaries: [{lower_bound:.2f}, {upper_bound:.2f}]")
    if len(outliers) > 0:
        print("\nOutlier values:")
        print(outliers.sort_values(ascending=False))

# Analyze outliers for important numerical features
important_features = ['SalePrice', 'GrLivArea', 'LotArea', 'TotalBsmtSF']
for feature in important_features:
    if feature in df.columns:
        plot_boxplot_with_outliers(df, feature)

# %% [markdown]
# ## 8. Year-based Analysis

# %%
# Analyze price trends over years
plt.figure(figsize=(15, 5))

# Sale Price vs YearBuilt
plt.subplot(1, 2, 1)
plt.scatter(df['YearBuilt'], df['SalePrice'], alpha=0.5)
plt.xlabel('Year Built')
plt.ylabel('Sale Price')
plt.title('Sale Price vs Year Built')

# Average price by year built
avg_price_by_year = df.groupby('YearBuilt')['SalePrice'].mean()
plt.subplot(1, 2, 2)
avg_price_by_year.plot()
plt.xlabel('Year Built')
plt.ylabel('Average Sale Price')
plt.title('Average Sale Price by Year Built')

plt.tight_layout()
plt.show()

# Print summary statistics by decade
df['Decade'] = (df['YearBuilt'] // 10) * 10
decade_stats = df.groupby('Decade')['SalePrice'].agg(['count', 'mean', 'std', 'min', 'max'])
print("\nSale Price Statistics by Decade:")
print(decade_stats) 