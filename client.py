"""
Example script to POST to the FastAPI app.
"""

import os
import requests
import pandas as pd

def convert_to_fast_api_format(inputs: pd.DataFrame):
    inputs_list = []

    # Convert each record into a FastAPI-appropriate input format
    for idx in range(len(inputs)):
        inputs_list.append(
            {k.replace(" ", "_"): v for k, v in inputs.iloc[idx].to_dict().items()}
        )

    return inputs_list

def main():
    # Configs
    DATA_DIR = "data/processed"                     # directory where the processed data is stored
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")  # test data path
    N_SAMPLES = 5                                   # number of samples to query FastAPI

    # Load the test data
    test_data = pd.read_csv(TEST_PATH)

    # Randomly sample some points
    sample_data = test_data.sample(N_SAMPLES)

    # Compose the inputs and targets
    target_columns = ['health_Poor', 'health_Fair', 'health_Good']
    inputs = sample_data.drop(columns=target_columns)
    targets = sample_data[target_columns]
    
    # Convert the inputs into an appropriate format for FastAPI
    inputs = convert_to_fast_api_format(inputs)

    # Convert the targets to strings
    targets = [target.split("_")[-1] for target in targets.idxmax(axis=1).tolist()]

    # Query FastAPI
    url = "http://127.0.0.1:8000/predict"
    for input, target in zip(inputs, targets):
        response = requests.post(url, json=input) # query FastAPI
        print(f"Prediction: {response.json()['prediction']}. Ground truth: {target}")

if __name__ == "__main__":
    main()