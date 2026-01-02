"""
Logistic Regression Model - Class-Based Implementation

This converts all the separate functions into a single class.
Think of it like putting all your tools in one toolbox!
"""

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    """
    A class that represents a Logistic Regression model.
    
    Think of this like a "machine" that:
    1. Stores its own weights and bias (the "memory")
    2. Can train itself (the "learning" function)
    3. Can make predictions (the "prediction" function)
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=4000, random_seed=42):
        """
        This is called when you create a new model.
        It's like setting up a new machine with default settings.
        
        Args:
            learning_rate: How big steps to take when learning
            max_iterations: How many times to try learning
            random_seed: For reproducible results
        """
        # Store these settings in the object (so we remember them)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        
        # These will be set later when we train
        # They're like empty slots that get filled during training
        self.weight = None  # Will store the learned weights
        self.bias = None     # Will store the learned bias
        self.history = {}    # Will store training history
    
    # ============================================================
    # HELPER METHODS (internal functions, not called directly)
    # ============================================================
    
    @staticmethod
    def _sigmoid(z):
        """
        The sigmoid function - converts any number to 0-1 range.
        The @staticmethod means it doesn't need 'self' - it's just a utility.
        """
        return 1 / (1 + np.exp(-z))
    
    def _initialize_parameters(self, n_features):
        """
        Initialize weights and bias to starting values.
        Notice: it uses 'self' - so it can store things in the object!
        """
        np.random.seed(self.random_seed)
        # Store these in self.weight and self.bias (the object remembers them)
        self.weight = np.full((n_features, 1), 0.01)
        self.bias = 0.0
    
    def _forward_and_backprop(self, X_values_Train_Val_OrTest, y_values_Train_Val_OrTest):
        """ 

        Do one step: predict, calculate loss, calculate gradients.
        Returns the loss and how to update weights.
        """ 
        X = X_values_Train_Val_OrTest
        y = y_values_Train_Val_OrTest 
        # Forward: calculate predictions
        logits = np.dot(self.weight.T, X) + self.bias  # Notice: uses self.weight!
        probabilities = self._sigmoid(logits)
        
        # Calculate loss
        loss = -y * np.log(probabilities) - (1 - y) * np.log(1 - probabilities)
        nll = np.sum(loss) / X.shape[1]
        
        # Backward: calculate gradients (how to update)
        error = probabilities - y
        gradient_weight = np.dot(X, error.T) / X.shape[1]
        gradient_bias = np.sum(error) / X.shape[1]
        
        gradients = {
            "weight": gradient_weight,
            "bias": gradient_bias
        }
        
        return nll, gradients
    
    def _compute_nll(self, X, y):
        """Calculate loss for validation/test (no gradients needed)."""
        logits = np.dot(self.weight.T, X) + self.bias
        probabilities = self._sigmoid(logits)
        
        epsilon = 1e-15
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        
        loss = -y * np.log(probabilities) - (1 - y) * np.log(1 - probabilities)
        nll = np.sum(loss) / X.shape[1]
        return nll
    
    @staticmethod
    def _compute_accuracy(y_true, y_pred):
        """Calculate how many predictions were correct."""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        return np.mean(y_pred_flat == y_true_flat)

    def TQFn(self, x_train):

        print("the current weights are", self.weight)

        # print("the current x_train are", x_train)
    
        #print("the current y_true are", self.y_true)
 
        return  
    
    # ============================================================
    # MAIN METHODS (these are what you call from outside)
    # ============================================================
    
    def fit(self, x_train, y_train, x_val, y_val, x_test, y_test):
        """
        Takes in the data ...
        Train the model! This is the main function you call.
        
        BEFORE (functional style):
            weight, bias = initializing_weight_bias(...)
            trained_params, history = update_parameters(weight, bias, ...)
        
        NOW (OOP style):
            model = LogisticRegression()
            history = model.fit(...)  # weight and bias stored inside model!
        
        Notice: We don't need to pass weight/bias around anymore!
        The model remembers them in self.weight and self.bias.
        """
        # Step 1: Initialize weights and bias
        self._initialize_parameters(x_train.shape[0])
        
        # Step 2: Set up tracking arrays
        train_nll_history = np.zeros(self.max_iterations)
        val_nll_history = []
        test_nll_history = []
        train_acc_history = []
        val_acc_history = []
        test_acc_history = []
        epoch_indices = []
        
        # Step 3: Training loop
        for epoch in range(self.max_iterations):
            # Calculate loss and gradients 
            train_nll, gradients = self._forward_and_backprop(x_train, y_train)
            train_nll_history[epoch] = train_nll   # just log values
            
            # Employ the gradient descent via the gradients calculated back  prop
            # Update weights and bias (stored in self!)
            self.weight = self.weight - gradients["weight"] * self.learning_rate
            self.bias = self.bias - gradients["bias"] * self.learning_rate
            
            # Every 100 epochs, check how we're doing
            if epoch % 100 == 0:
  
                self.TQFn(x_train)
                # Make predictions just for accuracy purposes using current weights
                y_train_pred = self.predict(x_train)
                train_acc = self._compute_accuracy(y_train, y_train_pred)
                
                # Check validation set
                val_nll = self._compute_nll(x_val, y_val)
                y_val_pred = self.predict(x_val)
                val_acc = self._compute_accuracy(y_val, y_val_pred)
                
                # Check test set
                test_nll = self._compute_nll(x_test, y_test)
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
                print(f'{epoch} train: {round(train_nll, 3)} {round(train_acc, 3)}; '
                      f'val: {round(val_nll, 3)} {round(val_acc, 3)}; '
                      f'test: {round(test_nll, 3)} {round(test_acc, 3)}')
        
        # Store history in the object
        self.history = {
            'train_nll': train_nll_history,
            'val_nll': np.array(val_nll_history),
            'test_nll': np.array(test_nll_history),
            'train_acc': np.array(train_acc_history),
            'val_acc': np.array(val_acc_history),
            'test_acc': np.array(test_acc_history),
            'epoch_indices': np.array(epoch_indices)
        }
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        BEFORE (functional style):
            y_pred = make_predictions(weight, bias, X)  # Had to pass weight/bias
        
        NOW (OOP style):
            y_pred = model.predict(X)  # Model already knows its weight/bias!
        """
        # Use self.weight and self.bias (the model remembers them!)
        logits = np.dot(self.weight.T, X) + self.bias
        probabilities = self._sigmoid(logits)
        predictions = (probabilities >= 0.5).astype(float)
        return predictions

    def TQPrintHistoy(self): 

        return self.history 
    
    def plot_training_curves(self, num_of_iters):
        """
        Plot the training curves.
        
        BEFORE (functional style):
            plot_training_curves(history, num_of_iters)  # Had to pass history
        
        NOW (OOP style):
            model.plot_training_curves(num_of_iters)  # Model has its own history!
        """
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
