"""
Data Analysis Script for Raisin Dataset

This script loads the raisin dataset and plots the distribution
of each of the 7 features, with different colors for each class.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config.local import RAISIN_DATA_PATH

def plot_feature_distributions():
    """Load data and plot distribution of each feature by class"""
    
    # Load the CSV data
    print("Loading data from:", RAISIN_DATA_PATH)
    df = pd.read_csv(RAISIN_DATA_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {df['Class'].unique()}")
    
    # Get the 7 feature columns (exclude Class)
    feature_columns = [col for col in df.columns if col != 'Class']
    
    # Create a figure with subplots for each feature
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Get unique classes for coloring
    classes = df['Class'].unique()
    colors = ['blue', 'orange']  # Two different colors for the two classes
    
    # Plot distribution for each feature
    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]
        
        # Plot histogram for each class
        for i, class_name in enumerate(classes):
            class_data = df[df['Class'] == class_name][feature]
            ax.hist(class_data, bins=30, alpha=0.6, label=class_name, 
                   color=colors[i], edgecolor='black')
        
        ax.set_title(f'{feature} Distribution', fontsize=10, fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        
        # Get the existing legend handles and labels
        handles, labels = ax.get_legend_handles_labels()
        
        # Add a custom entry for brown bars (overlap)
        from matplotlib.patches import Rectangle
        overlap_patch = Rectangle((0, 0), 1, 1, facecolor='brown', alpha=0.6, edgecolor='black')
        handles.append(overlap_patch)
        labels.append('Overlap')
         
        ax.legend(handles, labels)
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot (8th one, since we only have 7 features)
    axes[7].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Distribution of Raisin Features by Class', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.show()
    
    print("Plots generated successfully!")

if __name__ == '__main__':
    plot_feature_distributions()

