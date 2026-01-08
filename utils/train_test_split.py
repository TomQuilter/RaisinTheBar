def train_val_test_split(x_data, y_data, test_size: float, val_size: float):
    """
    
    Args:
        x_data: Feature data
        y_data: Target labels
        test_size: Proportion of data for testing (0.0 to 1.0)
        val_size: Proportion of data for validation (0.0 to 1.0)
     
    Returns:
        x_train, x_val, x_test, y_train, y_val, y_test
    """
    n_samples = len(x_data)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val
     
    # Simple split: train, then val, then test
    x_train = x_data[:n_train]
    x_val = x_data[n_train:n_train + n_val]
    x_test = x_data[n_train + n_val:]
    
    y_train = y_data[:n_train]
    y_val = y_data[n_train:n_train + n_val]
    y_test = y_data[n_train + n_val:]
    
    return x_train, x_val, x_test, y_train, y_val, y_test

