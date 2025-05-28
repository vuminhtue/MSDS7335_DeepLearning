# Ames Housing Dataset Downloader

## Overview
This repository contains a script to download the Ames Housing dataset, a comprehensive dataset for real estate analysis and predictive modeling. The dataset contains information on residential properties in Ames, Iowa, including features like lot size, building type, year built, and sale price.

## Implementation Details
The implementation deliberately avoids dependency on Pandas, instead utilizing NumPy and scikit-learn for data handling. The script provides two methodologies for dataset acquisition:

1. **Primary Method**: Utilizes scikit-learn's `fetch_openml` function to retrieve the dataset programmatically
2. **Fallback Method**: Direct download from GitHub repository source files

## Usage
Execute the downloader script:

```bash
python download_ames_housing.py
```

This will create a `data` directory containing:
- `ames_housing_features.csv`: Property features (independent variables)
- `ames_housing_target.csv`: Sale prices (dependent variable)

Or, if using the fallback method:
- `ames_housing_train.csv`: Training dataset
- `ames_housing_test.csv`: Test dataset

## Requirements
- Python 3.x
- NumPy
- scikit-learn

## Citation
De Cock, D. (2011). "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project," Journal of Statistics Education, Volume 19, Number 3.

## Methodological Considerations
The script implements robust error handling and verification procedures to ensure dataset integrity. The dual-method approach provides redundancy in case of API endpoint failures or network connectivity issues. 