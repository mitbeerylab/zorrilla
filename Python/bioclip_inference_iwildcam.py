from pathlib import Path
import pandas as pd
import torch
from bioclip import CustomLabelsClassifier
from tqdm import tqdm
import json

data_dir = Path("/data/vision/beery/scratch/data/iwildcam_unzipped")

df = []
split = "train"
raw_image_dir = data_dir / split

with open(data_dir / "metadata" / f"iwildcam2022_{split}_annotations.json") as f:
    data = json.load(f)

with open(data_dir / "metadata" / f"gps_locations.json") as f:
    gps_data = json.load(f)

categories = {c["id"]: c["name"] for c in data["categories"]}
classes = [category.title() for category in categories.values()]
image_id_to_category_id = {image["id"]: [] for image in data["images"]}
for annotation in data["annotations"]:
    image_id_to_category_id[annotation["image_id"]] = sorted([*image_id_to_category_id[annotation["image_id"]], annotation["category_id"]])

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
classifier = CustomLabelsClassifier(classes, device=device)

def map_species_name_to_column(species_name: str):
    return species_name.lower().replace(" ", "_")

df = []
for image in tqdm(data["images"]):
    gt_category_ids = image_id_to_category_id[image["id"]]
    image_path = raw_image_dir / image["file_name"]
    predictions = classifier.predict(str(image_path))
    row = {
        "split": split,
        **image,
        **(gps_data[str(image["location"])] if str(image["location"]) in gps_data else {"latitude": None, "longitude": None}),
        **{f"gt_{map_species_name_to_column(v)}": k in gt_category_ids for k, v in categories.items()},
    }
    for prediction in predictions:
        row["pred_" + map_species_name_to_column(prediction["classification"])] = prediction["score"]
    df.append(row)

pd.DataFrame(df).to_csv(Path(__file__).parent / ".." / "data" / "iwildcam_2022_bioclip_inference.csv")