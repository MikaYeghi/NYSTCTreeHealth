"""
Script to create the main image of the repository.
"""

import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme("paper")

def main():
    # Load the data
    data_path = os.path.join("../data/raw/2015-street-tree-census-tree-data.csv")
    df = pd.read_csv(data_path)

    # Extract the data for the figure
    X = df[['x_sp', 'y_sp']]
    y = df['status']
    
    # Plot
    fig, ax = plt.subplots()
    for status in y.unique():
        ax.scatter(
            X[y == status]['x_sp'], 
            X[y == status]['y_sp'],
            s=0.1
        )
    
    # Remove grid, ticks, labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.set_frame_on(False)

    # Save the figure at 300 dpi
    fig.savefig("main.png", dpi=300, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    main()
