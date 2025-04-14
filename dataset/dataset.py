import torch
import pandas as pd

def find_conflicting_inputs(X: pd.DataFrame, y: pd.DataFrame):
    # Columns with local features describe features local to each tree, and exclude external factors like location
    local_features_cols = ['curb_loc', 'sidewalk', 'root_stone', 'root_grate', 'root_other', 'trunk_wire', 'trnk_light', 'trnk_other', 'brch_light', 'brch_shoe', 'brch_other', 'steward_1or2', 'steward_3or4', 'steward_4orMore', 'steward_None', 'guards_Harmful', 'guards_Helpful', 'guards_None', 'guards_Unsure', 'user_type_NYC Parks Staff', 'user_type_TreesCount Staff', 'user_type_Volunteer']
    target_cols = ['health_Good', 'health_Fair', 'health_Poor']
    df = pd.concat([X, y], axis=1)

    # Group by input features and check how many unique one-hot vectors exist for each group
    grouped = df.groupby(local_features_cols)[target_cols].nunique()

    # Count how many groups have more than 1 unique one-hot pattern
    contradictions = grouped[(grouped > 1).any(axis=1)]

    print(f"Number of contradictory inputs: {len(contradictions)}")

class NYTreesDataset:
    def __init__(self, data_path: str, device: str = 'cpu', verbose: bool = False):
        self.data_path = data_path
        self.device = device
        self.verbose = verbose

        # Extract data
        self.X, self.y = self.extract_data(self.data_path)
        assert len(self.X) == len(self.y)
        if self.verbose:
            print(f"Extracted a dataset with {len(self)} records")

    def extract_data(self, data_path: str):
        # Load the CSV file
        data = pd.read_csv(data_path)

        # Normalize numerical features
        features_to_normalize = ['tree_dbh', 'x_sp', 'y_sp']
        for feature in features_to_normalize:
            data[feature] = (data[feature] - data[feature].mean()) / data[feature].std()

        # Separate the input and output variables
        target_features = [col_name for col_name in data.columns if "health" in col_name]
        X = data.drop(columns=target_features)
        y = data[target_features]

        # Lines below print out information about duplicated and conflicting records. NOTE: uncomment if you want to print the information
        # find_conflicting_inputs(X, y)
        # print(f"Percentage of duplicated columns: {round(X.drop(columns=['tree_dbh', 'x_sp', 'y_sp']).duplicated().sum() / len(X) * 100, 2)}%.")

        # Convert the data to a torch tensor
        X = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y = torch.tensor(y.values, dtype=torch.float32).to(self.device)

        return (X, y)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])
    
    def __len__(self):
        return len(self.X)
    
    def label_dist(self):
        """
        Return the label distribution.
        """
        labels, counts = self.y.unique(dim=0, return_counts=True)
        return {
            torch.argmax(label).item(): count.item()
            for label, count in zip(labels, counts)
        }
        

    def subsample(self, n):
        indices = torch.randperm(self.X.shape[0])[:n]
        self.X = self.X[indices]
        self.y = self.y[indices]