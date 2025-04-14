import torch
from fastapi import FastAPI
from utils import load_model
from pydantic import BaseModel

# Tree data class
class TreeData(BaseModel):
    # Numeric
    tree_dbh:   int
    x_sp:       float
    y_sp:       float

    # Binary categorical
    curb_loc:   int
    sidewalk:   int
    root_stone: int
    root_grate: int
    root_other: int
    trunk_wire: int
    trnk_light: int
    trnk_other: int
    brch_light: int
    brch_shoe:  int
    brch_other: int

    # One-hot categorical groups
    steward_1or2:       int
    steward_3or4:       int
    steward_4orMore:    int
    steward_None:       int

    guards_Harmful:     int
    guards_Helpful:     int
    guards_None:        int
    guards_Unsure:      int

    user_type_NYC_Parks_Staff:  int
    user_type_TreesCount_Staff: int
    user_type_Volunteer:        int

def convert_data(data: TreeData) -> torch.Tensor:
    input_tensor = torch.tensor([
        (data.tree_dbh - 12.94378698224852) / 8.033300860057345,    # normalize
        data.curb_loc,
        data.sidewalk,
        data.root_stone,
        data.root_grate,
        data.root_other,
        data.trunk_wire,
        data.trnk_light,
        data.trnk_other,
        data.brch_light,
        data.brch_shoe,
        data.brch_other,
        (data.x_sp - 1001652.7653807693) / 26698.80882726377,       # normalize
        (data.y_sp - 202086.60942810652) / 26698.80882726377,       # normalize
        data.steward_1or2,
        data.steward_3or4,
        data.steward_4orMore,
        data.steward_None,
        data.guards_Harmful,
        data.guards_Helpful,
        data.guards_None,
        data.guards_Unsure,
        data.user_type_NYC_Parks_Staff,
        data.user_type_TreesCount_Staff,
        data.user_type_Volunteer
    ]).unsqueeze(0)
    return input_tensor

# Initialize the app
app = FastAPI(title="Tree Health Classifier", version="1.0")

# Load the model
model = load_model("saved_models", None, 25, device='cuda:0')
model.eval()

@app.post("/predict")
def predict_health(data: TreeData):
    # Convert input to a valid input tensor
    input_tensor = convert_data(data)

    # Predict
    with torch.no_grad():
        output = model(input_tensor.to("cuda:0"))
        print(torch.softmax(output, dim=1))
        pred_class = torch.argmax(output, dim=1).item()

    label_map = {0: "Poor", 1: "Good", 2: "Fair"}
    return {"prediction": label_map[pred_class]}
