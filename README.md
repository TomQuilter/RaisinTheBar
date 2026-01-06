# Raisin Regression

So this is a project where I'm training a logistic regression model to classify raisins. 

## What's this about?

Basically, I'm using a raisin dataset to train a logistic regression classifier from scratch. The model learns to classify raisins based on some features 

## How it works

The main script (`main.py`) does a few things:
1. Loads and preprocesses the raisin data
2. Splits it into train/validation/test sets
3. Creates a LogisticRegression model object
4. Trains it and tracks the training history
5. Shows you the final accuracy scores and plots some training curves

## Setup

You'll need Python and probably some standard ML libraries (numpy, pandas, matplotlib, etc.). I haven't made a requirements.txt yet, but you'll figure it out.

The data should be in `data/Raisin_Dataset.csv`. Make sure that's there before running.

## Running it

Just run:
```bash
python main.py
```

It'll print out some progress messages, show you the final accuracies, and generate some plots showing how the model learned over time.

## Project structure

- `main.py` - The main script that ties everything together
- `config/` - Configuration files for training parameters and data paths
- `data/` - Where the raisin dataset lives
- `utils/` - Helper functions for data preprocessing and splitting
- `config/model/` - The LogisticRegression class implementation

## Notes

The model stores its weights and bias internally, which is nice. You can also access the training history through the model object if you want to do more analysis later.

That's pretty much it. Nothing too fancy, just a clean implementation of logistic regression with some basic training visualization.

