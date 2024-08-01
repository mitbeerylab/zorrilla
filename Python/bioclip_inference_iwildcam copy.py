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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

batch_size = 16
df = []
data["images"] = data["images"][:32]
progressbar = tqdm(total=len(data["images"]))
for images in chunks(data["images"], batch_size):
    image_paths = [str(raw_image_dir / image["file_name"]) for image in images]
    overall_predictions = classifier.predict(image_paths)

    rows = {image_path: {} for image_path in image_paths}

    for image, image_path in zip(images, image_paths):
        rows[image_path] = {
            "split": split,
            **image,
            **(gps_data[str(image["location"])] if str(image["location"]) in gps_data else {"latitude": None, "longitude": None}),
            **{f"gt_{map_species_name_to_column(v)}": k in image_id_to_category_id[image["id"]] for k, v in categories.items()},
        }

    for prediction in overall_predictions:
        rows[prediction["file_name"]]["pred_" + map_species_name_to_column(prediction["classification"])] = prediction["score"]
        
    for image_path in image_paths:
        df.append(rows[image_path])

    progressbar.update(len(images))

pd.DataFrame(df).to_csv(Path(__file__).parent / ".." / "data" / "iwildcam_2022_bioclip_inference.csv")