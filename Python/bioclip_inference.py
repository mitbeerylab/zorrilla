from pathlib import Path
import pandas as pd
import torch
from bioclip import CustomLabelsClassifier
from tqdm import tqdm

data_dir = Path(__file__).parent / ".." / "data"
raw_image_dir = Path("/data/vision/beery/scratch/data/occupancy_ml_uncertainty")

classes = [
    "human",
    "vehicle",
    "dog",
    "fox",
    "chamois",
    "wild boar",
    "badger",
    "roe deer",
    "cat",
    "lynx",
]
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
classifier = CustomLabelsClassifier(device=device)

df = []
for fn in tqdm(list(raw_image_dir.rglob("*.*"))):
    if fn.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue
    predictions = classifier.predict(str(fn), classes)
    row = {
        "filepath": str(fn.relative_to(raw_image_dir))
    }
    for prediction in predictions:
        row[prediction["classification"]] = prediction["score"]
    df.append(row)


pd.DataFrame(df).to_csv(data_dir / "bioclip_inference.csv")