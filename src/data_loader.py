import numpy as np
 
def print_raw_csv(csv_path,num_rows=5):
    """   
    Loads the raisin dataset from a CSV file.
 
    Returns
    -------
    X : numpy array
        Feature matrix
    y : numpy array
        Target vector
    """

    raw_data = np.genfromtxt(
        csv_path,
        delimiter="\t",
        dtype=str
    )       

    print("Full shape:", raw_data.shape)

    for row in raw_data[:num_rows]: 
        print(row) 
    
