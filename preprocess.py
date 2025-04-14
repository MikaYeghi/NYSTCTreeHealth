"""
This script preprocesses raw data and saves the processed data to `data/processed`.
The processing logic replicates the one in `notebooks/EDA.ipynb`.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def remove_conflicting_records(df):
    input_cols = [col for col in df.columns if col not in ['tree_dbh', 'x_sp', 'y_sp', 'health_Good', 'health_Fair', 'health_Poor']]
    target_cols = ['health_Good', 'health_Fair', 'health_Poor']

    # Group by input columns and check how many unique one-hot target combinations exist per group
    label_variation = df.groupby(input_cols)[target_cols].nunique()

    # Identify input combinations where any label column has more than 1 unique value
    conflicting_input_combos = label_variation[(label_variation > 1).any(axis=1)].index

    # Create a mask to find all rows matching those conflicting input combinations
    input_tuples = df[input_cols].apply(tuple, axis=1)

    # Create a set of conflicting input tuples
    conflicting_input_set = set(conflicting_input_combos)

    # Build a boolean mask for rows that are not in the conflicting set
    non_conflicting_mask = ~input_tuples.isin(conflicting_input_set)

    # Apply the mask to keep only the non-conflicting rows
    df_cleaned = df[non_conflicting_mask].copy()

    print(f"Removed {round((len(df) - len(df_cleaned)) / len(df) * 100, 2)}% of the original data.")

    return df_cleaned

def main():
    # Some configs
    DATA_PATH = 'data/raw/2015-street-tree-census-tree-data.csv'    # raw data path
    SAVE_DIR = 'data/processed'                                     # directory to save the processed data
    RANDOM_SEED = 42                                                # random seed for reproducibility

    # Load the data
    print("Loading raw data")
    assert os.path.exists(DATA_PATH), f'Data file not found: {DATA_PATH}'
    df = pd.read_csv(DATA_PATH)

    # Remove columns
    columns_to_drop = [
        'tree_id', 'state', 'stump_diam', 'bin', 'bbl', 'problems',
        'longitude', 'latitude',
        'nta_name', 
        'spc_common',
        'block_id', 
        'address', 
        'postcode', 
        'community board', 
        'borocode', 
        'cncldist', 
        'council district', 
        'st_assem', 
        'st_senate', 
        'nta', 
        'census tract', 
        'boro_ct',
        'created_at',
        'zip_city', 
        'borough', 
        'zip_city', 
        'borough', 
        'spc_latin'
    ]
    print(f"Dropping {len(columns_to_drop)} columns")
    df.drop(columns=columns_to_drop, inplace=True)

    # Preserve only alive trees data
    print("Preserving only alive trees data")
    df = df[df['status'] == "Alive"]
    df.drop(columns=['status'], inplace=True)

    # Drop some rows & fill in null values for steward and guard fields
    print("Filling `steward` and `guard` fields with None")
    df['steward'] = df['steward'].fillna('None')
    df['guards'] = df['guards'].fillna('None')
    df.dropna(inplace=True)

    # Encode categorical variables
    print("Encoding categorical variables")
    cat_cols = df.columns[df.dtypes == 'object'].tolist()
    for cat_col in cat_cols:
        n_unique = df.nunique()[cat_col] # retrieve the number of unique categorical values

        # Encode
        if n_unique == 2:   # binary variable -> encode as (0, 1)
            df[cat_col] = df[cat_col].map(
                {
                    df[cat_col].unique()[0]: np.float64(0), 
                    df[cat_col].unique()[1]: np.float64(1)
                }
            )
        else:               # multi-class variable, use one-hot encoding
            df = pd.get_dummies(
                df, 
                columns=[cat_col], 
                prefix=cat_col,
                dtype=np.float64
            )
    
    # Convert the tree diameter into float
    df['tree_dbh'] = df['tree_dbh'].astype(float)
    
    # Remove duplicated rows. NOTE: uncomment if you want to drop repeating records
    # df = df.drop_duplicates()

    # Find conflicting records
    df = remove_conflicting_records(df)

    # Split the data into train/val/test. Currently the split is hardcoded.
    train_df, valtest_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED) 
    val_df, test_df = train_test_split(valtest_df, test_size=0.5, random_state=RANDOM_SEED)
    del valtest_df
    print(f"\nTrain data: {len(train_df)}\nValidation data: {len(val_df)}\nTest data: {len(test_df)}\n")

    # Save the split data
    print(f"Saving the data to {SAVE_DIR}")
    save_dir = os.path.join(SAVE_DIR)
    os.makedirs(save_dir, exist_ok=True)
    train_save_path = os.path.join(save_dir, "train.csv")
    val_save_path = os.path.join(save_dir, "val.csv")
    test_save_path = os.path.join(save_dir, "test.csv")
    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)

if __name__ == "__main__":
    main()
