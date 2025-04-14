"""
This script is used to run inference on a pre-trained model.
"""

import os
import torch
from pprint import pprint
from utils import load_model
from dataset import NYTreesDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

def run_checks(test_path: str, models_dir: str, model_name: str):
    # Check that the test data exists
    assert os.path.exists(test_path), f"Test data not found: {test_path}"

    # Check that the model checkpoint exists
    if not model_name:
        model_name = "last.pth"
    if model_name.split('.')[-1] != "pth":
        model_name += ".pth"
    ckpt_path = os.path.join(models_dir, model_name)
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

def main():
    # Configs
    DATA_DIR = "data/processed"                     # directory where the processed data is stored
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")  # test data path
    MODELS_DIR = "saved_models"                     # directory of pretrained models
    MODEL_NAME = None                               # either None, or model filename
    BATCH_SIZE = 128                                # batch size used during testing
    DEVICE = 'cuda:0'                               # one of ['cpu', 'cuda:0']
    VERBOSE = True                                  # one of [True, False]]

    # Run some checks
    run_checks(TEST_PATH, MODELS_DIR, MODEL_NAME)

    # Load the data
    test_data = NYTreesDataset(TEST_PATH, device=DEVICE, verbose=VERBOSE)

    # Initialize the test loader
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Load the model
    model = load_model(MODELS_DIR, MODEL_NAME, test_data[0][0].shape[0], device=DEVICE)

    # Initialize tensors to store the ground-truth and predictions
    preds_list = torch.empty((0, 3))
    gt_list = torch.empty((0, 3))

    # Run inference
    model.eval()
    with torch.no_grad():
        for inputs_batch, targets_batch in test_loader:
            # Predictions
            preds = model(inputs_batch)                                                     # produce prediction logits
            preds = torch.softmax(preds, dim=1)                                             # apply softmax to the logits [NOTE: in fact this is not necessary, but it's a good practice]
            preds = torch.nn.functional.one_hot(torch.argmax(preds, dim=1), num_classes=3)  # select the most likely prediction
            preds_list = torch.cat((preds_list, preds.cpu()), dim=0)                        # record the predictions

            # Ground-truth labels
            gt_list = torch.cat((gt_list, targets_batch.cpu()), dim=0)                      # record the ground-truth labels

    # Print the metrics
    pprint(classification_report(gt_list, preds_list))

    # Save the metrics
    # TODO

if __name__ == "__main__":
    main()