"""
Simple unit tests

HOW TO RUN:
-----------
python -m unittest tests.test_sigmoid
or
python tests/test_sigmoid.py
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path to import the model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.model.LogisticRegression import LogisticRegression
 
class Unit_Tests(unittest.TestCase):
    
    def test_sigmoid_zero_equals_half(self):
        """Test that sigmoid(0) equals 0.5"""   
        result = LogisticRegression._sigmoid(0)
        self.assertAlmostEqual(result, 0.5, places=10)
        print("TEST PASSED: test_sigmoid_zero_equals_half")
    
    def test_one_gradient_step_reduces_loss(self):
        """Test that one gradient step reduces loss on a tiny dataset"""
        # Create small dataset: 7 features, 3 samplese
        X = np.array([[1.0, 0.5, 2.0, 1.5, 0.8, 1.2, 0.7],  
                      [2.0, 1.0, 0.5, 2.5, 1.2, 0.9, 1.5],  
                      [0.5, 1.5, 1.0, 0.8, 1.8, 1.3, 1.1]]).T  # transpose to (7, 3)
        y = np.array([[1, 0, 1]])  
        
        # Create model 
        model = LogisticRegression(learning_rate=0.01, random_seed=42)
        
        # Initialize parameters
        model._initialize_parameters(X.shape[0])
        
        # Get initial loss
        initial_loss, _ = model._forward_and_backprop(X, y)
        
        # Do one gradient step
        loss, gradients = model._forward_and_backprop(X, y)
        model.weight = model.weight - gradients["weight"] * model.learning_rate
        model.bias = model.bias - gradients["bias"] * model.learning_rate
        
        # Get loss after gradient step
        new_loss, _ = model._forward_and_backprop(X, y)
        
        # Check that loss decreased
        self.assertLess(new_loss, initial_loss, 
                       f"Loss should decrease: {initial_loss} -> {new_loss}")
        print("TEST PASSED: test_one_gradient_step_reduces_loss")
 
if __name__ == '__main__':
    unittest.main()

