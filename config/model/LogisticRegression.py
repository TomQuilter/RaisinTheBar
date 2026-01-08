"""
Logistic Regression Model

"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

class LogisticRegression:
    
    def __init__(self, learning_rate: float = None, max_iterations: int = None, random_seed: int = None) -> None:
        """
        
        Args:
            learning_rate: How big steps to take when learning (defaults to JSON config)
            max_iterations: Maximum number of training iterations (defaults to JSON config)
            random_seed: For reproducible results (defaults to JSON config)
        """
        # Load config from JSON file
        config_dir = Path(__file__).resolve().parent.parent
        config_path = config_dir / "training_config.json"

        try:
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'training' not in config:
                raise KeyError("'training' key not found in configuration file")
            
            training_config = config['training']
            
            # Validate required keys exist
            required_keys = ['learning_rate', 'num_iterations', 'random_seed']
            missing_keys = [key for key in required_keys if key not in training_config]
            if missing_keys:
                raise KeyError(f"Missing required configuration keys: {missing_keys}")
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file error: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {config_path}: {e}") from e
        except KeyError as e:
            raise KeyError(f"Configuration error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading configuration: {e}") from e 

  
        # Set Attributes ...
        self.learning_rate = learning_rate if learning_rate is not None else training_config['learning_rate']
        self.max_iterations = max_iterations if max_iterations is not None else training_config['num_iterations']
        self.random_seed = random_seed if random_seed is not None else training_config['random_seed']
        
        self.weight = None  # learned weights
        self.bias = None     # learned bias
        self.history = {}    # training history
    
    # ============================================================
    # HELPER METHODS 
    # ========================================================
    @staticmethod   ###  
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
     
    def _initialise_parameters(self, n_features: int) -> None:
        np.random.seed(self.random_seed)
        self.weight = np.full((n_features, 1), 0.01)
        self.bias = 0.0
    
    def _forward_and_backprop(self, X_values_Train_Val_OrTest: np.ndarray, y_values_Train_Val_OrTest: np.ndarray) -> tuple:
        """ 
        Do one step: predict, calculate loss, calculate gradients.
        Returns the loss and how to update weights.
        """ 
        X = X_values_Train_Val_OrTest
        y = y_values_Train_Val_OrTest 
        # Forward: calculate predictions
        logits = np.dot(self.weight.T, X) + self.bias  
        probabilities = self._sigmoid(logits)
        
        # Calculate loss
        loss = -y * np.log(probabilities) - (1 - y) * np.log(1 - probabilities)
        nll = np.sum(loss) / X.shape[1]
        
        # Runtime CHECK - Make sure loss is finite! Common issue in training
        self._validate_loss_finite(nll)
        
        # Backward: calculate gradients (how to update)
        error = probabilities - y
        gradient_weight = np.dot(X, error.T) / X.shape[1]
        gradient_bias = np.sum(error) / X.shape[1]
        
        gradients = {
            "weight": gradient_weight,
            "bias": gradient_bias
        }
        
        return nll, gradients
     
    def _compute_nll(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate loss for validation/test (no gradients needed)."""
        logits = np.dot(self.weight.T, X) + self.bias
        probabilities = self._sigmoid(logits)

        loss = -y * np.log(probabilities) - (1 - y) * np.log(1 - probabilities)
        nll = np.sum(loss) / X.shape[1]
        
        # Runtime validation: ensure loss is finite
        self._validate_loss_finite(nll)
        
        return nll
    
    @staticmethod
    def _compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate how many predictions were correct."""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        return np.mean(y_pred_flat == y_true_flat)

    @staticmethod   
    def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix for binary classification.
        
        Returns a 2x2 matrix:
        [[TN, FP],
         [FN, TP]]
        where:
        - TN: True Negatives (pred=0, true=0)
        - FP: False Positives (pred=1, true=0)
        - FN: False Negatives (pred=0, true=1)
        - TP: True Positives (pred=1, true=1)
        """
        y_true_flat = y_true.flatten().astype(int)
        y_pred_flat = y_pred.flatten().astype(int)
        
        # Initialize 2x2 confusion matrix
        cm = np.zeros((2, 2), dtype=int)
        
        # Uses the 0 and 1s as co-ords of the 2 by 2 matrix, lovely stuff
        for i in range(len(y_true_flat)):
            true_val = y_true_flat[i]
            pred_val = y_pred_flat[i]
            cm[true_val, pred_val] += 1 
        
        return cm

    @staticmethod
    def _validate_loss_finite(loss_value: float) -> None:
        """Simple runtime validation: Check that loss is finite (not NaN, not infinity)."""
        assert np.isfinite(loss_value).all() if isinstance(loss_value, np.ndarray) else np.isfinite(loss_value), \
            "Loss is not finite (NaN or infinity)"

    # ============================================================
    # MAIN METHODS (these are what you call from outside)
    # ============================================================
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict:
 
        # initialise weights and bias
        self._initialise_parameters(x_train.shape[0])
        
        # Set up tracking arrays
        train_nll_history = np.zeros(self.max_iterations)
        val_nll_history = []
        test_nll_history = []
        train_acc_history = []
        val_acc_history = []
        test_acc_history = []
        epoch_indices = []
        
        # Training loop
        for epoch in range(self.max_iterations):
            # Calculate loss and gradients 
            train_nll, gradients = self._forward_and_backprop(x_train, y_train)
            # Runtime validation already performed in _forward_and_backprop
            train_nll_history[epoch] = train_nll   # just log values
            
            # Employ the gradient descent via the gradients calculated back  prop
            # Update weights and bias (stored in self!)
            self.weight = self.weight - gradients["weight"] * self.learning_rate
            self.bias = self.bias - gradients["bias"] * self.learning_rate
             
            # Every 100 epochs, check how we're doing
            if epoch % 100 == 0:
   
                # Make predictions just for accuracy purposes using current weights
                y_train_pred = self.predict(x_train)
                train_acc = self._compute_accuracy(y_train, y_train_pred)
                
                # Check validation set
                val_nll = self._compute_nll(x_val, y_val)
                # Runtime validation already performed in _compute_nll
                y_val_pred = self.predict(x_val)
                val_acc = self._compute_accuracy(y_val, y_val_pred)
                
                # Check test set
                test_nll = self._compute_nll(x_test, y_test)
                # Runtime validation already performed in _compute_nll
                y_test_pred = self.predict(x_test)
                test_acc = self._compute_accuracy(y_test, y_test_pred)
  
                # Store metrics
                epoch_indices.append(epoch)
                val_nll_history.append(val_nll)
                test_nll_history.append(test_nll)
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                test_acc_history.append(test_acc)
                
                # Print progress
                print(f'{epoch} train loss: {round(train_nll, 3)} train acc: {round(train_acc, 3)}; '
                      f'val loss: {round(val_nll, 3)} val acc: {round(val_acc, 3)}; '
                      f'test loss: {round(test_nll, 3)} test acc: {round(test_acc, 3)}')
        
        # Compute final predictions for confusion matrices
        y_train_pred_final = self.predict(x_train)
        y_val_pred_final = self.predict(x_val)
        y_test_pred_final = self.predict(x_test)
        
        # Store history in the object
        self.history = {
            'train_nll': train_nll_history,
            'val_nll': np.array(val_nll_history),
            'test_nll': np.array(test_nll_history),
            'train_acc': np.array(train_acc_history),
            'val_acc': np.array(val_acc_history),
            'test_acc': np.array(test_acc_history),
            'epoch_indices': np.array(epoch_indices),
            'train_cm': self._confusion_matrix(y_train, y_train_pred_final),
            'val_cm': self._confusion_matrix(y_val, y_val_pred_final),
            'test_cm': self._confusion_matrix(y_test, y_test_pred_final)
        }
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        """ 
        # Use self.weight and self.bias (the model remembers them!)
        logits = np.dot(self.weight.T, X) + self.bias
        probabilities = self._sigmoid(logits)
        predictions = (probabilities >= 0.5).astype(float)
        return predictions
    
    def plot_training_curves(self, num_of_iters: int) -> None:

        epoch_indices = self.history['epoch_indices']
        
        # Plot 1: NLL curves
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(num_of_iters), self.history['train_nll'], label='Train NLL')
        plt.plot(epoch_indices, self.history['val_nll'], label='Validation NLL', marker='o')
        plt.plot(epoch_indices, self.history['test_nll'], label='Test NLL', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log Likelihood')
        plt.title('NLL Curves')
        plt.legend()
        plt.grid(True)
         
        # Plot 2: Accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(epoch_indices, self.history['train_acc'], label='Train Accuracy', marker='o')
        plt.plot(epoch_indices, self.history['val_acc'], label='Validation Accuracy', marker='s')
        plt.plot(epoch_indices, self.history['test_acc'], label='Test Accuracy', marker='^')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True)
         
        plt.tight_layout()
        plt.show()
    
    def print_confusion_matrix_stats(self) -> None:
        """Print confusion matrix statistics for train, validation, and test sets."""
        print("\n" + "="*60)
        print("Confusion Matrix Statistics")
        print("="*60)
        
        # Class labels: 0 = Besni, 1 = Kecimen (based on encoding in data_preprocessing)
        class_names = {0: "Besni", 1: "Kecimen"}
          
        for dataset_name, cm in [("Train", self.history['train_cm']), 
                                 ("Validation", self.history['val_cm']), 
                                 ("Test", self.history['test_cm'])]:
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            total = tn + fp + fn + tp
            
            print(f"\n{dataset_name} Set:")
            print(f"  Confusion Matrix:")
            print(f"                    Predicted")
            print(f"                  {class_names[0]:8s} {class_names[1]:8s}")
            print(f"  True {class_names[0]:6s} [{tn:4d}      {fp:4d}     ]")
            print(f"       {class_names[1]:6s} [{fn:4d}      {tp:4d}     ]")
            print(f"\n  Breakdown:")
            print(f"    True Negatives (TN):  {tn:4d} ({tn/total*100:.2f}%) - Correctly predicted {class_names[0]}")
            print(f"    False Positives (FP): {fp:4d} ({fp/total*100:.2f}%) - Predicted {class_names[1]}, but was {class_names[0]}")
            print(f"    False Negatives (FN): {fn:4d} ({fn/total*100:.2f}%) - Predicted {class_names[0]}, but was {class_names[1]}")
            print(f"    True Positives (TP):  {tp:4d} ({tp/total*100:.2f}%) - Correctly predicted {class_names[1]}")
            print(f"    Total: {total}")

