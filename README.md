# Hi Fuzzy Labs! :)
 
I started with more raw function based implementation to get a feel for the problem and validate the approach, I did this "manually" which was great fun ... I then restructured into oop code for better scalability and cleaner code ostly myself and using LLMs a bit, while making sure I drove and understood each part of the restructuring.  
 
Ive put in two unit tests, but havent had time to put lots of them in. I've put in one runtime test.
 
Ive used ML Flow and some logging.

I've just done the Main regression task so far, I'll do one of the extension tasks soon (was going to do it tonight but got sick).

Cheers, Tom 

# Raisin Classification with Logistic Regression
 
This is me building a logistic regression model from scratch to classify raisins. No  libraries - just  Python (was fun!)
 
Trained a binary classifier to tell the difference between two types of raisins (Kecimen and Besni) based on their features.  Just gradient descent and some lovely matrix maths.  

## Project structure

- `main.py` - The main script that runs everything
- `config/model/LogisticRegression.py` - The model class (all the math happens here)
- `utils/data_preprocessing.py` - Loads and preprocesses the data
- `utils/train_test_split.py` - Custom train/val/test splitter (no sklearn!)
- `config/training_config.json` - Hyperparameters live here
- `config/local.py` - Paths and config loading
- `training.log` - Log file (created automatically)

## The model

The `LogisticRegression` class does everything:
- Initializes weights and bias
- Forward pass (predictions)
- Backward pass (gradient calculation)
- Gradient descent updates
- Computes loss and accuracy
- Builds confusion matrices
- Plots training curves
 
It's all in one class, so you can create a model, train it, and use it to make predictions.
 

