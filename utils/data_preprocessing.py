from config.local import RAISIN_DATA_PATH, RANDOM_SEED, TEST_SPLIT, VAL_SPLIT
from sklearn.model_selection import train_test_split 

import numpy as np
import pandas as pd
 
def preprocess_data(data):

    # Read in the data

    db = pd.read_csv(data)
 
    print("DataPreShuffle = ", db)

    # Shuffle the dataset rows
    db = db.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
   
    print("DataPostShuffle = ", db)
 
    # db = db.head(1000) #TQ to remove 
       
    print ("data = " , db)
  
    # print(db["Class"])  

    db["Class"] = [1 if each == "Kecimen" else 0 for each in db["Class"]]
   
    # print ("data = " , db) 
    #print_raw_csv(data, num_rows=5)



    ## Split the output labels from the features

    ## Target_labels

    y_db = db["Class"]
 
    #print("target_Labels", y_db)

    ## Features_Database
    db.drop("Class" , axis = 1 , inplace = True)
    x_db = db

    ## Split the data into Train, Validation, and Test sets FIRST
    ## BEFORE normalization to avoid data leakage!
    ## First split: test vs (train+val)
    ## Second split: from (train+val), split into train and val

    # First split: test vs train+val
    x_temp, x_test, y_temp, y_test = train_test_split(
        x_db, y_db, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y_db
    )
    
    # Second split: train vs val (from the remaining data)
    # val_split is percentage of train+val data that becomes validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=y_temp
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