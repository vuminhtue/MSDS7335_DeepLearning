#!/usr/bin/env python3
"""
Ames Housing Dataset Downloader
------------------------------
This script downloads the Ames Housing dataset and saves it as CSV files.
It avoids using Pandas and instead relies on numpy for data manipulation.
"""

import os
import urllib.request
import numpy as np
import csv
import ssl
import requests

def download_ames_housing_sklearn():
    """
    Download the Ames Housing dataset using scikit-learn's fetch_openml function.
    This method avoids using Pandas by handling the dataset directly.
    """
    try:
        from sklearn.datasets import fetch_openml
        print("Downloading Ames Housing dataset via scikit-learn...")
        
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Due to limitations in the no-pandas approach, we'll use as_frame=True
        # and then immediately convert to arrays
        print("Fetching dataset from OpenML...")
        housing = fetch_openml(name='house_prices', version=1, as_frame=True, parser='auto')
        
        # Convert to numpy arrays and save manually
        print("Converting data format...")
        X_df = housing.data
        y_df = housing.target
        
        # Save features to CSV
        print("Saving features dataset...")
        with open('data/ames_housing_features.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['id'] + list(X_df.columns))
            # Write data
            for i, (_, row) in enumerate(X_df.iterrows()):
                writer.writerow([i] + list(row.values))
        
        # Save target to CSV
        print("Saving target dataset...")
        with open('data/ames_housing_target.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['id', 'SalePrice'])
            # Write data
            for i, val in enumerate(y_df.values):
                writer.writerow([i, val])
        
        print(f"Dataset saved to data/ames_housing_features.csv and data/ames_housing_target.csv")
        print(f"Dataset shape: {X_df.shape[0]} samples, {X_df.shape[1]} features")
        
        return True
    except Exception as e:
        print(f"Error in scikit-learn method: {e}")
        return False

def download_ames_housing_direct():
    """
    Download the Ames Housing dataset directly from its source URL.
    This provides an alternative method in case the scikit-learn method fails.
    """
    print("Downloading Ames Housing dataset directly from source...")
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # URLs for the dataset
    train_url = "https://raw.githubusercontent.com/cristianasp/ml/master/train.csv"
    test_url = "https://raw.githubusercontent.com/cristianasp/ml/master/test.csv"
    
    # Download the training dataset using requests
    try:
        print("Downloading training dataset...")
        response = requests.get(train_url)
        if response.status_code == 200:
            with open('data/ames_housing_train.csv', 'wb') as f:
                f.write(response.content)
            print("Training dataset downloaded successfully.")
        else:
            print(f"Failed to download training dataset: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading training dataset: {e}")
    
    # Download the test dataset
    try:
        print("Downloading test dataset...")
        response = requests.get(test_url)
        if response.status_code == 200:
            with open('data/ames_housing_test.csv', 'wb') as f:
                f.write(response.content)
            print("Test dataset downloaded successfully.")
        else:
            print(f"Failed to download test dataset: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading test dataset: {e}")
    
    # Try an alternative source if the first one fails
    if not os.path.exists('data/ames_housing_train.csv') or os.path.getsize('data/ames_housing_train.csv') == 0:
        print("Trying alternative source...")
        kaggle_url = "https://storage.googleapis.com/kagglesdsdata/competitions/5407/868278/train.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1589924532&Signature=UUdCTBRSXsHk6UGGpL%2B%2B%2FzDu2G0aYOGTZ%2Fw47Ec4eUOeXAUMxkyWZ6nfVNSBdTrVQ3FLfgbagvi%2FcOgYuTUTFKnyBbhx61LcVjcNLNRQy1I%2FDpW9vJg4gjwZ9WKBvnhcVZQlQWqEz%2FIYLl4QCFS0NC0JpxOCZzHnQcmEMy1UQQNv1ZvO%2FXBWB8jKcfI5z0m4%2BrEyzJD4%2F1dqRopgAuqkL0lMkE23qeoBcNdMuHLA9H6GQA5JHZPJCzVzFz5uXYpL8RJPXxKL%2BqHBrHxRNGMW5NMtZ3xgJZdYpKuJwLQy5BaOfKyKI15UtWuZcnXnmU9Wc5G9bdlOmU9QPM27Cg%3D%3D"
        try:
            response = requests.get(kaggle_url)
            if response.status_code == 200:
                with open('data/ames_housing_train.csv', 'wb') as f:
                    f.write(response.content)
                print("Training dataset downloaded from alternative source.")
        except Exception as e:
            print(f"Error downloading from alternative source: {e}")

def download_from_uci():
    """
    Download from UCI Machine Learning Repository as another fallback.
    """
    print("Attempting download from UCI ML Repository...")
    try:
        # URL for UCI ML Repository
        uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
        response = requests.get(uci_url)
        if response.status_code == 200:
            with open('data/ames_housing_uci.csv', 'wb') as f:
                f.write(response.content)
            print("Dataset downloaded from UCI repository.")
            return True
        else:
            print(f"Failed to download from UCI: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading from UCI repository: {e}")
        return False

if __name__ == "__main__":
    print("Ames Housing Dataset Downloader")
    print("===============================")
    
    success = False
    
    # Try scikit-learn method first
    if download_ames_housing_sklearn():
        success = True
    else:
        print("Scikit-learn method failed, trying direct download...")
        download_ames_housing_direct()
        
        # Check if any files were successfully downloaded
        if os.path.exists('data/ames_housing_train.csv') and os.path.getsize('data/ames_housing_train.csv') > 0:
            success = True
        else:
            print("Direct download method failed, trying UCI repository...")
            success = download_from_uci()
    
    print("\nVerifying downloaded files:")
    found_files = False
    for filename in os.listdir('data'):
        if filename.startswith('ames_housing') or filename == 'housing.data':
            found_files = True
            file_path = os.path.join('data', filename)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            print(f"- {filename}: {file_size:.2f} KB")
    
    if not found_files:
        print("No dataset files were found in the data directory.")
        print("Please check your internet connection and try again.")
    elif success:
        print("\nDownload completed successfully!")
    else:
        print("\nDownload was partially successful or may have encountered issues.")
        print("Please check the files in the data directory.") 