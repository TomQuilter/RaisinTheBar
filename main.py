from config.local import RAISIN_DATA_PATH, LEARNING_RATE, NUM_ITERATIONS, RANDOM_SEED
from utils.data_preprocessing import preprocess_data
from models.LogisticRegression import LogisticRegression
import numpy as np
import mlflow
import logging
 
# Configure logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
 
def main():

    # MLfloww
    mlflow.set_tracking_uri("file:./mlruns")  

    # Load configuration
    learning_rate = LEARNING_RATE
    num_of_iters = NUM_ITERATIONS

    # draw in  the data
    logger.info("Loading and preprocessing data...")
    try:
        x_train, x_val, x_test, y_train, y_val, y_test, train_min, train_range = preprocess_data(RAISIN_DATA_PATH)
        logger.info(f"Data loaded successfully - Train: {x_train.shape[1]}, Val: {x_val.shape[1]}, Test: {x_test.shape[1]} samples")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return

    with mlflow.start_run():
        # Log parameters ... http://127.0.0.1:5000/ 
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_iterations", num_of_iters)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("n_features", x_train.shape[0])
        mlflow.log_param("n_train_samples", x_train.shape[1])
        mlflow.log_param("n_val_samples", x_val.shape[1])
        mlflow.log_param("n_test_samples", x_test.shape[1])

        # Create The model
        logger.info("Creating Logistic Regression model...")
        logger.info(f"Hyperparameters - Learning rate: {learning_rate}, Max iterations: {num_of_iters}")
        TheLogisitcRegressionModel = LogisticRegression(
            learning_rate=learning_rate,
            max_iterations=num_of_iters
        )  
    
        # ACTUALLy TRAIN the model 
        logger.info("Starting model training...")
        history = TheLogisitcRegressionModel.fit(x_train, y_train, x_val, y_val, x_test, y_test)
        logger.info("Training completed successfully")
  
        # Log metrics
        try:
            final_train_acc = history['train_acc'][-1]
            final_val_acc = history['val_acc'][-1]
            final_test_acc = history['test_acc'][-1]
            
            logger.info(f"Final accuracies - Train: {final_train_acc:.4f}, Val: {final_val_acc:.4f}, Test: {final_test_acc:.4f}")
            
            mlflow.log_metric("final_train_accuracy", final_train_acc)
            mlflow.log_metric("final_val_accuracy", final_val_acc)
            mlflow.log_metric("final_test_accuracy", final_test_acc)
             
            # Log FINAL loss
            if len(history['val_nll']) > 0:
                mlflow.log_metric("final_val_loss", history['val_nll'][-1])
            train_nll_array = history['train_nll']
            if len(train_nll_array) > 0:
                mlflow.log_metric("final_train_loss", train_nll_array[-1])
        except (KeyError, IndexError) as e:
            logger.warning(f"Could not log all metrics: {e}")
        except Exception as e:
            logger.warning(f"Error logging metrics to MLflow: {e}", exc_info=True)

        # Display results
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Trained bias: {TheLogisitcRegressionModel.bias}")
        print(f"Final train accuracy: {final_train_acc:.4f}")
        print(f"Final validation accuracy: {final_val_acc:.4f}")
        print(f"Final test accuracy: {final_test_acc:.4f}")
        print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
         
        # Plot training curves
        print("\nPlotting training curves...")
        TheLogisitcRegressionModel.plot_training_curves(num_of_iters)   
  
        # Just for fun - Predict on a new made-up raisin
        new_raisin = ((np.array([[90000, 450, 260, 0.82, 92000, 0.75, 1200]]) - train_min.values) / train_range.values).T
        print(f"\nPrediction on new raisin: {TheLogisitcRegressionModel.predict(new_raisin)[0,0]}")
       
        print(f"Final Model Weights - for later Explainability: {TheLogisitcRegressionModel.weight}")
  
        # Print the confusion matrix stats with context ...
        TheLogisitcRegressionModel.print_confusion_matrix_stats()
         
if __name__ == "__main__":
    main() 
