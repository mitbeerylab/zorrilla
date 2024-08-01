from pathlib import Path
import pandas as pd
import torch
from bioclip import TreeOfLifeClassifier, Rank
from tqdm import tqdm

data_dir = Path(__file__).parent / ".." / "data"
raw_image_dir = Path("/data/vision/beery/scratch/data/occupancy_ml_uncertainty")

genus_to_classes = {
    "Homo": "human",
    "Canis": "dog",
    "Vulpes": "fox",
    "Rupicapra": "chamois",
    "Sus": "wild boar",
    "Meles": "badger",
    "Capreolus": "roe deer",
    "Felis": "cat",
    "Lynx": "lynx",
}
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
classifier = TreeOfLifeClassifier(device=device)

df = []
for fn in tqdm(list(raw_image_dir.rglob("*.*"))):
    
    # TODO: remove
    if not "ain" in str(fn).lower():
        continue

    if fn.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue
    predictions = classifier.predict(str(fn), Rank.GENUS, k=100)
    row = {
        "filepath": str(fn.relative_to(raw_image_dir))
    }
    for cls in genus_to_classes.values():
        row[cls] = 0
    for prediction in predictions:
        if not prediction["genus"] in genus_to_classes:
            continue
        cls = genus_to_classes[prediction["genus"]]
        row[cls] = prediction["score"]
    
    assert len(set(genus_to_classes.values()) - set(row.keys())) == 0, f"expected all genera to be present in row but missing: {set(genus_to_classes.values()) - set(row.keys())}"

    df.append(row)


pd.DataFrame(df).to_csv(data_dir / "bioclip_inference_genus.csv")