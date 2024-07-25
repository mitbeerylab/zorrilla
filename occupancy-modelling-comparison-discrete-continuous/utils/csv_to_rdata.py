import rdata
import pyreadr
import pandas as pd
import numpy as np

dfo = rdata.read_rda("data/metadata_Ain.RData")["allfiles"]

dfn = pd.read_csv("data/bioclip_inference_genus_plus_gt.csv")

print(f"number of sites: {dfn["n_point"].nunique()}")

from collections import Counter
print(Counter(dfo["observed"].tolist()))


dfn["DateTimeOriginal"] = dfn["datetime"]

threshold = 0.5
target_species = "roe deer"


print("Removing", dfn["DateTimeOriginal"].isna().sum(), "missing datetime values")
dfn = dfn[~dfn["DateTimeOriginal"].isna()]

dfn["predicted"] = np.array(["lynx" if row[target_species] >= threshold else "other" for _, row in dfn.iterrows()])

# TODO: remove. Try to use some GT labels instead of predictions
np.random.seed(42)
gt_proportion = 0.0
use_gt_mask = np.random.rand(len(dfn)) < gt_proportion

dfn["predicted"][use_gt_mask] = np.array(["lynx" if row["observed"] == target_species else "other" for _, row in dfn.iterrows()])[use_gt_mask]

print((dfn["predicted"] == "lynx").sum(), f"positive observations with threshold {threshold}")


tp = ((dfn["predicted"] == target_species) & (dfn["observed"] == target_species)).sum()
fp = ((dfn["predicted"] == target_species) & (dfn["observed"] != target_species)).sum()
fn = ((dfn["predicted"] != target_species) & (dfn["observed"] == target_species)).sum()

lynx_recall = tp / (tp + fn)
lynx_precision = tp / (tp + fp)

print(f"lynx precision {lynx_precision:.2f} recall {lynx_recall:.2f}")

dfn["observed"] = dfn["predicted"]



pyreadr.write_rdata(f"data/metadata_Ain_thres_{threshold}.RData", dfn, df_name="allfiles")