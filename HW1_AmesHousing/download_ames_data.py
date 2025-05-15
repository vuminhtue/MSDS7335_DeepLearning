import pandas as pd
import os

def download_ames_housing_data():
    """
    Download the Ames Housing dataset and save it as CSV files.
    The data will be split into training and test sets.
    """
    # URLs for the Ames Housing dataset from Dean De Cock's original data
    train_url = "https://raw.githubusercontent.com/TITHI-KHAN/Predictive-Analysis-of-Housing-Prices/main/train.csv"
    test_url = "https://raw.githubusercontent.com/TITHI-KHAN/Predictive-Analysis-of-Housing-Prices/main/test.csv"
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    try:
        # Download training data
        print("Downloading training data...")
        train_data = pd.read_csv(train_url)
        train_data.to_csv('data/ames_housing_train.csv', index=False)
        print(f"Training data saved: {train_data.shape[0]} rows and {train_data.shape[1]} columns")
        
        # Download test data
        print("Downloading test data...")
        test_data = pd.read_csv(test_url)
        test_data.to_csv('data/ames_housing_test.csv', index=False)
        print(f"Test data saved: {test_data.shape[0]} rows and {test_data.shape[1]} columns")
        
        print("\nData successfully downloaded and saved in the 'data' directory!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    download_ames_housing_data() 