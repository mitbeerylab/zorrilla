import json
import os
import pandas as pd
from tqdm import tqdm
import fire


def cameratrapai_to_csv(labels_path, gps_path, input_path_train, output_path, split="train"):
    with open(labels_path, 'r') as f:
        data = json.load(f)
    with open(gps_path, 'r') as f:
        gps_data = json.load(f)
    with open(input_path_train, 'r') as f:
        train_data = json.load(f)

    uuid_to_train_data_idx = {os.path.splitext(os.path.basename(prediction["filepath"]))[0]: idx for idx, prediction in enumerate(train_data["predictions"])}

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
    classes_cameratrapai = [
        # "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank",
        "575008c3-8e3b-4efe-8da6-47631d60d5b7;mammalia;cetartiodactyla;tayassuidae;tayassu;pecari;white-lipped peccary",
        "466b25f0-a916-432c-823e-394a69391328;aves;galliformes;phasianidae;meleagris;ocellata;ocellated turkey",
        "dd39bbd5-077c-482e-9d33-bd176116c870;mammalia;perissodactyla;equidae;equus;quagga;plains zebra",
        "c9b58a23-a776-4927-b737-68ed6c34fcba;mammalia;cetartiodactyla;bovidae;madoqua;guentheri;guenther's dik-dik",
        "22976d14-d424-4f18-a67a-d8e1689cefcc;mammalia;carnivora;felidae;leopardus;pardalis;ocelot",
        "2dca052b-dff5-4cc9-8072-1282c5713286;mammalia;cetartiodactyla;giraffidae;giraffa;camelopardalis;giraffe",
        "d372cda5-a8ca-4b7b-97ed-4e4fab9c9b4b;mammalia;cetartiodactyla;suidae;sus;scrofa;wild boar",
        "d5d634a9-f86f-493f-92f0-c2ae1ed1b5bd;mammalia;cetartiodactyla;cervidae;mazama;pandora;yucatan brown brocket",
        # "fa099712-2ca2-4645-afc2-367b2a585852;mammalia;cetartiodactyla;bovidae;litocranius;walleri;gerenuk",
        # "a0841346-8260-4450-96b8-1e4ded395805;aves;passeriformes;formicariidae;formicarius;analis;black-faced antthrush",
        # "6eb22182-155a-42f8-87ce-21e6053ca60e;mammalia;cetartiodactyla;tragulidae;tragulus;javanicus;javan chevrotain",
        # "3bbd196e-e3a2-4024-a6dc-f20153c9428f;reptilia;squamata;teiidae;tupinambis;teguixin;black tegu",
    ]
    classes_inv = {v: k for k, v in classes.items()}
    image_id_to_category_id = {image["id"]: [] for image in data["images"]}
    
    for annotation in data["annotations"]:
        image_id_to_category_id[annotation["image_id"]] = sorted([*image_id_to_category_id[annotation["image_id"]], annotation["category_id"]])

    df = []
    for image in tqdm(data["images"]):
        gt_category_ids = image_id_to_category_id[image["id"]]
        prediction = train_data["predictions"][uuid_to_train_data_idx[image["id"]]]

        df.append({
            "split": split,
            **image,
            **(gps_data[str(image["location"])] if str(image["location"]) in gps_data else {"latitude": None, "longitude": None}),
            **{f"gt_{v.lower().replace(' ', '_')}": k in gt_category_ids for k, v in categories.items()},
            **{f"logit_{k}": prediction["classifications"]["subset_logits"][i + 1] for i, k in enumerate(classes.keys())},
        })

    pd.DataFrame(df).to_csv(output_path)


if __name__ == "__main__":
    fire.Fire(cameratrapai_to_csv)