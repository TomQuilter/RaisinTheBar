"""
Data preprocessing utilities for raisin classification.

Handles data loading, cleaning, encoding, splitting, and normalization.
"""

from config.local import RAISIN_DATA_PATH, RANDOM_SEED, TEST_SPLIT, VAL_SPLIT
from utils.train_test_split import train_val_test_split

import numpy as np
import pandas as pd


def preprocess_data(data):
    """
    Load and preprocess raisin dataset.
    
    Args:
        data: Path to CSV file containing raisin data
        
    Returns:
        x_train, x_val, x_test, y_train, y_val, y_test, train_min, train_range
        - Feature matrices and labels for train/val/test sets
        - train_min and train_range for normalization (to normalize new data)
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data is invalid or missing required columns
        RuntimeError: For unexpected errors during processing
    """

    #### Read in the data ####
    try:
        db = pd.read_csv(data)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Data file is empty: {data}")
    except Exception as e:
        raise RuntimeError(f"Error reading data file {data}: {e}") from e
    
    # Validate data has required column
    if 'Class' not in db.columns:
        raise ValueError(f"Required column 'Class' not found in data. Available columns: {list(db.columns)}")
    
    if db.empty:
        raise ValueError("Data file contains no rows")

    # Shuffle the dataset rows
    db = db.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    db = db.head(40) #TQ to remove - for quick testing 
        
    db["Class"] = [1 if each == "Kecimen" else 0 for each in db["Class"]]
   
    ## Split the output labels from the features

    y_db = db["Class"]

    ## Features_Database
    db.drop("Class" , axis = 1 , inplace = True)
    x_db = db

    ## Split the data into Train, Validation, and Test sets FIRST  (BEFORE normalization to avoid data leakage)

    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
        x_db, y_db, test_size=TEST_SPLIT, val_size=VAL_SPLIT
    )
    
    # NOW normalize using ONLY training set statistics to avoid data leakage
    # Normalised_Value = (Value - feature_min) / (FeatureMax - FeatureMin) = (val - min)/col width 
    # Compute min/max from training set only
    train_min = x_train.min()
    train_max = x_train.max()
    train_range = train_max - train_min
    
    # Avoid division by zero
    train_range = train_range.replace(0, 1)
    
    # Normalize all sets using training statistics
    x_train = (x_train - train_min) / train_range
    x_val = (x_val - train_min) / train_range
    x_test = (x_test - train_min) / train_range
    
    print("Xtrain1", x_train)
    # Reshape and transpose to match expected format
    y_train = y_train.values.reshape(-1,1)   
    y_test = y_test.values.reshape(-1,1)
    y_val = y_val.values.reshape(-1,1)
    
    x_train = x_train.T                      ## Transpose the data
    x_test = x_test.T 
    x_val = x_val.T
    
    y_train = y_train.T
    y_test = y_test.T
    y_val = y_val.T
    
    print("x_train shape : {}".format(x_train.shape)) 
    print("x_val shape : {}".format(x_val.shape))
    print("x_test shape : {}".format(x_test.shape))

    print("Nice!")

    return x_train, x_val, x_test, y_train, y_val, y_test, train_min, train_range 