from pathlib import Path
import pandas as pd
import torch
from bioclip import CustomLabelsClassifier
from tqdm import tqdm
import json
import numpy as np
from PIL import Image
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.models import detection as pw_detection
import cv2
import os
import tempfile
import supervision as sv
detection_model = pw_detection.MegaDetectorV5(device="cuda", pretrained=True)


# test_path = '/data/vision/beery/scratch/data/iwildcam_unzipped/train/922ad620-21bc-11ea-a13a-137349068a90.jpg'
# pil_image = Image.open(str(test_path)).convert("RGB")
# img = np.array(pil_image)
# transform = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
#                                             stride=detection_model.STRIDE)
# results = detection_model.single_image_detection(transform(img), img.shape)
# print(results)
# quit()


data_dir = Path("/data/vision/beery/scratch/data/iwildcam_unzipped")

df = []
split = "train"
raw_image_dir = data_dir / split

with open(data_dir / "metadata" / f"iwildcam2022_{split}_annotations.json") as f:
    data = json.load(f)

with open(data_dir / "metadata" / f"gps_locations.json") as f:
    gps_data = json.load(f)

categories = {c["id"]: c["name"] for c in data["categories"]}
classes = {
    "tayassu_pecari": "Animalia Chordata Mammalia Artiodactyla Tayassuidae Tayassu pecari (White-lipped peccary)",
    "meleagris_ocellata": "Animalia Chordata Aves Galliformes Phasianidae Meleagris ocellata (Ocellated turkey)",
    "equus_quagga": "Animalia Chordata Mammalia Perissodactyla Equidae Equus quagga (Plains zebra)",
    "madoqua_guentheri": "Animalia Chordata Mammalia Artiodactyla Bovidae Antilopinae Madoqua guentheri (Günther's dik-dik)",
    "leopardus_pardalis": "Animalia Chordata Mammalia Carnivora Feliformia Felidae Leopardus pardalis (Ocelot)",
    "giraffa_camelopardalis": "Animalia Chordata Mammalia Artiodactyla Giraffidae Giraffa camelopardalis (Northern giraffe)",
    "sus_scrofa": "Animalia Chordata Mammalia Artiodactyla Suidae Sus scrofa (Wild boar)",
    "mazama_pandora": "Animalia Chordata Mammalia Artiodactyla Cervidae Odocoileus pandora (Yucatan brown brocket)",
}
classes_inv = {v: k for k, v in classes.items()}
image_id_to_category_id = {image["id"]: [] for image in data["images"]}
for annotation in data["annotations"]:
    image_id_to_category_id[annotation["image_id"]] = sorted([*image_id_to_category_id[annotation["image_id"]], annotation["category_id"]])

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
classifier = CustomLabelsClassifier(list(classes.values()), device=device)

def map_species_name_to_column(species_name: str):
    return classes_inv[species_name]

df = []
with tempfile.TemporaryDirectory() as tmpdir:
    for image in tqdm(data["images"]):
        gt_category_ids = image_id_to_category_id[image["id"]]
        image_path = raw_image_dir / image["file_name"]

        row = {
            "split": split,
            **image,
            **(gps_data[str(image["location"])] if str(image["location"]) in gps_data else {"latitude": None, "longitude": None}),
            **{f"gt_{v.lower().replace(' ', '_')}": k in gt_category_ids for k, v in categories.items()},
        }

        pil_image = Image.open(str(image_path)).convert("RGB")
        img = np.array(pil_image)
        transform = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                                    stride=detection_model.STRIDE)
        results = detection_model.single_image_detection(transform(img), img.shape)
        sorted_idx = np.argsort(results["detections"].confidence)[::-1]
        filtered_idx = [idx for idx in sorted_idx if results["detections"].class_id[idx] == 0]
        if len(filtered_idx) > 0:
            
        
            cropped_img = sv.crop_image(
                image=img, xyxy=results["detections"].xyxy[filtered_idx[0]]
            )
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
            suffix = ".png"
            
            cropped_img_path = os.path.join(tmpdir, os.path.splitext(os.path.basename(str(image_path)))[0] + ".png")
            cv2.imwrite(cropped_img_path, cropped_img)


            predictions = classifier.predict(cropped_img_path, return_logits=True)

            try:
                os.unlink(cropped_img_path)
            except:
                pass

            for prediction in predictions:
                row["pred_" + map_species_name_to_column(prediction["classification"])] = prediction["score"]
                row["logit_" + map_species_name_to_column(prediction["classification"])] = prediction["logit"]

        else:
            tqdm.write(f"No animal detections for image '{image_path}'")
        
        df.append(row)

    pd.DataFrame(df).to_csv(Path(__file__).parent / ".." / "data" / f"iwildcam_2022_crops_bioclip_inference_logits{'' if split == 'train' else ('_' + split)}_v3.csv")