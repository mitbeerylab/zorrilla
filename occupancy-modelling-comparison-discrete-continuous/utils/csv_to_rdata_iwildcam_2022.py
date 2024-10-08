import rdata
import pyreadr
import pandas as pd
import numpy as np

dfo = rdata.read_rda("data/metadata_Ain.RData")["allfiles"]

dfn = pd.read_csv("data/iwildcam_2022_bioclip_inference.csv")

print(f"number of sites: {dfn['location'].nunique()}")

from collections import Counter
print(Counter(dfo["observed"].tolist()))


dfn["DateTimeOriginal"] = dfn["datetime"]

threshold = 1e-3
target_species = "meleagris_ocellata"
target_species_out = "lynx"


print("Removing", dfn["DateTimeOriginal"].isna().sum(), "missing datetime values")
dfn = dfn[~dfn["DateTimeOriginal"].isna()]

dfn["predicted"] = np.array([target_species_out if row[f"pred_{target_species}"] >= threshold else "other" for _, row in dfn.iterrows()])

# TODO: remove. Try to use some GT labels instead of predictions
np.random.seed(42)
gt_proportion = 0.0
use_gt_mask = np.random.rand(len(dfn)) < gt_proportion

if gt_proportion > 0:
    dfn["predicted"][use_gt_mask] = np.array([target_species_out if row[f"gt_{target_species}"] else "other" for _, row in dfn.iterrows()])[use_gt_mask]

print((dfn["predicted"] == target_species_out).sum(), f"positive observations with threshold {threshold}")


tp = ((dfn["predicted"] == target_species_out) &  dfn[f"gt_{target_species}"]).sum()
fp = ((dfn["predicted"] == target_species_out) & ~dfn[f"gt_{target_species}"]).sum()
fn = ((dfn["predicted"] != target_species_out) &  dfn[f"gt_{target_species}"]).sum()

recall = tp / (tp + fn)
precision = tp / (tp + fp)

print(f"{target_species} precision {precision:.2f} recall {recall:.2f}")

dfn["observed"] = dfn["predicted"]
dfn["pix"] = dfn["location"].astype(str) + "_" + dfn["file_name"]

pyreadr.write_rdata(f"data/metadata_iwildcam_2022_thres_{threshold}.RData", dfn, df_name="allfiles")