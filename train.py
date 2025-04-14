"""
This script is used to train a DL model.
"""

import os
import glob
import torch
import random
from tqdm import tqdm
from pprint import pprint
from model_zoo import MLP
from dataset import NYTreesDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

# Functions
def create_save_path(models_dir: str) -> str:
    """
    Function that creates a unique model name.
    TODO: create subfolders, save different epochs.
    """
    # Create the models directory if it doesn't exist yet
    os.makedirs(models_dir, exist_ok=True)

    # Get the list of models
    existing_models_list = glob.glob(os.path.join(models_dir, "*.pth"))

    # Generate a unique name
    is_unique = False
    model_name = "model-"
    while not is_unique:
        random_seq = str(random.randint(1000000, 9999999)) # generate a random 7-digit number
        proposed_name = model_name + random_seq + ".pth"
        if proposed_name not in existing_models_list:
            is_unique = True
    
    save_path = os.path.join(models_dir, proposed_name)

    return save_path

def save_checkpoint(model: torch.nn.Module, epoch: int, train_loss: float, val_loss: float, save_path: str):
    """
    Function for saving a model checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    }
    torch.save(checkpoint, save_path)

    # Also save to last.pth
    last_save_path = "/".join(save_path.split('/')[:-1] + ["last.pth"])
    torch.save(checkpoint, last_save_path)

def run_checks(train_path: str, val_path: str):
    """
    Function for running some basic checks before proceeding to training.
    """
    # Check that the train and validation files exist
    assert os.path.exists(train_path), f"Training data not found: {train_path}"
    assert os.path.exists(val_path), f"Validation data not found: {val_path}"

def main():
    # Configs
    DATA_DIR = "data/processed"                         # directory where the processed data is stored
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")    # path to the training data
    VAL_PATH = os.path.join(DATA_DIR, "val.csv")        # path to the validation data
    MODELS_DIR = "saved_models"                         # directory for storing trained models
    BATCH_SIZE = 128                                    # training/validation batch size
    N_EPOCHS = 500                                      # number of training epochs
    BASE_LR = 0.005                                     # base learning rate
    GAMMA_EPOCHS = 100                                  # gamma is reduced every GAMMA_EPOCHS epochs
    SCHEDULER_GAMMA = 0.5                               # learning rate is multiplied by this number every 100 epochs
    DEVICE = 'cuda:0'                                   # one of ['cpu', 'cuda:0']
    VERBOSE = True                                      # one of [True, False]

    # Run some checks
    run_checks(TRAIN_PATH, VAL_PATH)
    
    # Load the data
    train_data = NYTreesDataset(TRAIN_PATH, device=DEVICE, verbose=VERBOSE)
    val_data = NYTreesDataset(VAL_PATH, device=DEVICE, verbose=VERBOSE)

    # Initialize the dataloaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # Initialize the model
    print(f"Number of input features: {train_data[0][0].shape[0]}")
    model = MLP(input_dim=train_data[0][0].shape[0]).to(DEVICE)

    # Come up with a codename for the model. It will be used for identifying a save directory for the model
    save_path = create_save_path(MODELS_DIR)

    # Get weights for the loss function to handle data imbalance
    label_dist = train_data.label_dist()
    weights = {k: sum(label_dist.values()) / v for k, v in label_dist.items()}
    weights = {k: v / sum(weights.values()) for k, v in weights.items()} # normalize
    weights = torch.tensor([weights[k] for k in sorted(weights.keys())], device=DEVICE)

    # Initialize loss function, optimizer & LR scheduler
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, 
        gamma=SCHEDULER_GAMMA
    )

    # Optimize the weights
    best_val_loss = float('inf') # validation loss for identifying the best model
    for epoch in range(N_EPOCHS):
        print("=" * 20 + f"EPOCH {epoch + 1}/{N_EPOCHS}" + "=" * 20)
        
        # Training
        model.train()
        train_loss = 0
        for inputs_batch, targets_batch in tqdm(train_loader):
            preds = model(inputs_batch)
            loss = loss_fn(preds, targets_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        preds_list = torch.empty((0, 3))
        gt_list = torch.empty((0, 3))
        with torch.no_grad():
            for inputs_batch, targets_batch in tqdm(val_loader):
                # Compute the validation loss
                preds = model(inputs_batch)
                val_loss += loss_fn(preds, targets_batch).item()

                # Compute the validation metrics
                preds_one_hot = torch.nn.functional.one_hot(torch.argmax(torch.softmax(preds, dim=1), dim=1), num_classes=3) # NOTE: technically there is no need to apply a softmax here, but it is done for completeness
                preds_list = torch.cat((preds_list, preds_one_hot.cpu()))
                gt_list = torch.cat((gt_list, targets_batch.cpu()))
            
            print(f"F1 score: {round(f1_score(preds_list, gt_list, average='macro') * 100, 2)}%")
            pprint(classification_report(preds_list, gt_list))

        val_loss /= len(val_loader)

        # Print out the loss information
        if VERBOSE:
            print(f"Training loss: {round(train_loss / len(train_loader), 4)} | Validation loss: {round(val_loss, 4)} | LR: {optimizer.param_groups[0]['lr']}")

        # Save the model if it has improved compared to the previous checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, epoch, train_loss, val_loss, save_path)
            if VERBOSE:
                print(f"Best model has been updated at epoch {epoch + 1} | Validation loss: {round(val_loss, 4)}")

        # Reduce the learning rate
        if epoch % GAMMA_EPOCHS == 0:
            lr_scheduler.step()

if __name__ == "__main__":
    main()