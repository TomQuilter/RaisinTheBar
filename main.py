from config.local import RAISIN_DATA_PATH, LEARNING_RATE, NUM_ITERATIONS
from utils.data_preprocessing import preprocess_data
from config.model.LogisticRegression import LogisticRegression

def main():
    """
    Main function to train the Logistic Regression model using OOP approach.
    """
    # Load configuration
    learning_rate = LEARNING_RATE
    num_of_iters = NUM_ITERATIONS

    # Load and preprocess data
    print("Loading and preprocessing data...")
    x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data(RAISIN_DATA_PATH)

    # Create model object (OOP way!)
    print("Creating Logistic Regression model...")
    TheLogisitcRegressionModel = LogisticRegression(
        learning_rate=learning_rate,
        max_iterations=num_of_iters
    )

    # Train the model (model stores weight/bias internally!)
    print("Training model...")
    history = TheLogisitcRegressionModel.fit(x_train, y_train, x_val, y_val, x_test, y_test)

    # Display results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Trained weight shape: {TheLogisitcRegressionModel.weight.shape}")
    print(f"Trained bias: {TheLogisitcRegressionModel.bias}")
    print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Final test accuracy: {history['test_acc'][-1]:.4f}")
    
    # Plot training curves (model has its own history!)
    print("\nPlotting training curves...")
    TheLogisitcRegressionModel.plot_training_curves(num_of_iters)   ## done on history NOT live 
  
    historyfromtheobject = TheLogisitcRegressionModel.TQPrintHistoy() 
    print("Call the val_nll_history", historyfromtheobject['epoch_indices']) 
     
if __name__ == "__main__":
    main() 
