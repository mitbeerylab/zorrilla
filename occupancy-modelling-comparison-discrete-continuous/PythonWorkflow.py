# %%
import os
from collections import Counter
# import rdata
import pyreadr
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from astral import LocationInfo, sun
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()
from multiprocessing import Process, Queue, Event

# %%

from astropy.time import TimeDelta
x = list(range(48))
y = []
loc = coord.EarthLocation(lon=50 * u.deg, lat=5 * u.deg)
for i in x:
    time = Time('1999-01-01T00:00:00.123456789') + TimeDelta(i * 3600, format="sec")
    altaz = coord.AltAz(location=loc, obstime=time)
    sun = coord.get_sun(time)
    sun.transform_to(altaz).alt
    y.append(float(sun.transform_to(altaz).alt / u.deg))

import matplotlib.pyplot as plt
plt.plot(x, y)

# %%
def get_thresholds(target_species_scores, n_sweep_steps):
    valid_mask = np.isfinite(target_species_scores)
    return np.linspace(np.min(target_species_scores[valid_mask]), np.max(target_species_scores[valid_mask]), n_sweep_steps)
    # return np.percentile(target_species_scores[valid_mask], np.linspace(0, 100, n_sweep_steps))

# %%
target_species_out = "lynx"
output_table_path = os.path.join("data", "iwildcam_2022_results_v6.csv")

tf = TimezoneFinder()


os.chdir(os.path.dirname(__file__))  # TODO: remove

# dfo = rdata.read_rda(os.path.join("data", "metadata_Ain.RData"))["allfiles"]
dfn = pd.read_csv(os.path.join("..", "data", "iwildcam_2022_crops_bioclip_inference_logits_v3.csv"))

# parse datetimes and filter out rows missing datetimes
dfn["datetime"] = pd.to_datetime(dfn["datetime"])
print("Removing", dfn["datetime"].isna().sum(), "missing datetime values")
dfn = dfn[~dfn["datetime"].isna()]

df_dem = pd.read_csv(os.path.join("..", "data", "iwildcam_2022_dem.csv"))
df_landcover = pd.read_csv(os.path.join("..", "data", "iwildcam_2022_landcover.csv"))

print(f"number of sites before: {dfn['location'].nunique()}")

dfn = dfn.merge(df_dem, left_on="location", right_on="name", how="left", suffixes=("", "_dem")).merge(df_landcover, left_on="location", right_on="name", how="left", suffixes=("", "_landcover"))

# # compute time since sunrise/sunset
# for _, row in dfn:
#     loc = LocationInfo(latitude=row["latitude"], longitude=row["longitude"])
#     s = sun(loc.observer, date=row["datetime"])


# TODO: re-enable
# # compute sun altitude above horizon in degrees
# alt = []
# for _, row in dfn.iterrows():
# # for _, row in dfn.sample(frac=1).iterrows():  # TODO: remove
#     if np.isfinite([row["latitude"], row["longitude"]]).all():
#         loc = coord.EarthLocation(lon=row["longitude"] * u.deg, lat=row["latitude"] * u.deg)
#         local_time = row["datetime"].to_pydatetime().replace(tzinfo=ZoneInfo(tf.timezone_at(lng=row["longitude"], lat=row["latitude"])))
#         time = Time(local_time)
#         altaz = coord.AltAz(location=loc, obstime=time)
#         sun = coord.get_sun(time)
#         alt += [float(sun.transform_to(altaz).alt / u.deg)]
#         # filepath = f"/data/vision/beery/scratch/data/iwildcam_unzipped/train/{row['file_name']}"
#         # print(alt[-1], local_time, filepath)
#     else:
#         alt += [None]

# dfn["sun_alt"] = alt

print(f"number of sites: {dfn['location'].nunique()}")
# print(f"number of observations: {Counter(dfo['observed'].tolist())}")

pred_prefix = "logit_"  # "pred_" or "logit_"
n_sweep_steps = 11
calibration_min_samples = 10

# target_species_list = None
target_species_list = [e.replace(" ", "_") for e in [
    "tayassu pecari",
    "meleagris ocellata",
    "equus quagga",
    "madoqua guentheri",
    "leopardus pardalis",
    "giraffa camelopardalis",
    "sus scrofa",
    "mazama pandora",
    # "litocranius walleri",
    # "formicarius analis",
    # "tragulus javanicus",
    # "tupinambis teguixin",
]]

for target_species in target_species_list:
    scores = dfn[f"{pred_prefix}{target_species}"] = dfn[f"{pred_prefix}{target_species}"].fillna(value=-float("inf"))

best_threshold = {}

# split along sequence IDs
train_seq, test_seq = train_test_split(dfn["seq_id"].unique(), test_size=0.8, random_state=42)
df_train, df_test = dfn[dfn["seq_id"].isin(train_seq)], dfn[dfn["seq_id"].isin(test_seq)]

for target_species in target_species_list:
    gt = df_train[f"gt_{target_species}"]
    scores = df_train[f"{pred_prefix}{target_species}"]
    pos_scores = scores[ df_train[f"gt_{target_species}"]]
    neg_scores = scores[~df_train[f"gt_{target_species}"]]
    assert len(pos_scores) >= calibration_min_samples and len(neg_scores) > calibration_min_samples
    scores_unique = np.sort(np.unique(scores))
    f1 = []
    for s in scores_unique:
        tp = ((scores >= s) &  gt).sum()
        fp = ((scores >= s) & ~gt).sum()
        fn = ((scores <  s) &  gt).sum()
        tn = ((scores <  s) & ~gt).sum()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        if precision + recall > 0:
            f1_item = (2 * precision * recall) / (precision + recall)
        else:
            f1_item = 0
        f1 += [f1_item]
    f1 = np.array(f1)
    assert len(f1[np.isfinite(f1)]) > 0
    best_score = np.mean(scores_unique[f1 == f1[np.isfinite(f1)].max()])
    print(f"Best threshold for species '{target_species}' is {best_score:.2f} at f1 of {f1[np.isfinite(f1)].max():.2f}")
    best_threshold[target_species] = best_score

# use only test data to continue
dfn = df_test

output_table = []


available_species = [e.replace(pred_prefix, "") for e in dfn.columns if e.startswith(pred_prefix) and e != f"{pred_prefix}empty"]
if target_species_list is None:
    target_species_list = available_species

# for target_species in target_species_list:
#     fig, (ax1, ax2) = plt.subplots(1, 2, )
#     target_species_scores = dfn[f"{pred_prefix}{target_species}"]
#     valid_mask = ~np.isnan(target_species_scores)

#     thresholds = get_thresholds(target_species_scores, n_sweep_steps)
#     ax1.hist(target_species_scores[valid_mask &  dfn[f"gt_{target_species}"]], bins=20, alpha=0.5, label="True")
#     ax1.hist(target_species_scores[valid_mask & ~dfn[f"gt_{target_species}"]], bins=20, alpha=0.5, label="False")
#     ax1.plot(thresholds, [0] * len(thresholds), 'o', label="Thresholds")
#     ax1.set_yscale("log")
    
#     precision = []
#     recall = []
#     for threshold in thresholds:
#         predicted = np.array([target_species_out if row[f"{pred_prefix}{target_species}"] >= threshold else "other" for _, row in dfn.iterrows()])
#         tp = ((predicted == target_species_out) &  dfn[f"gt_{target_species}"]).sum()
#         fp = ((predicted == target_species_out) & ~dfn[f"gt_{target_species}"]).sum()
#         fn = ((predicted != target_species_out) &  dfn[f"gt_{target_species}"]).sum()
#         tn = ((predicted != target_species_out) & ~dfn[f"gt_{target_species}"]).sum()
#         r = tp / (tp + fn)
#         p = tp / (tp + fp)
#         if np.isfinite([r, p]).all():
#             recall += [r]
#             precision += [p]

#     ax2.plot(recall, precision)
#     ax1.set_xlabel("Logit")
#     ax1.set_ylabel("Number of Observations")
#     ax2.set_xlabel("Recall")
#     ax2.set_ylabel("Precision")
#     fig.suptitle(target_species)
#     os.makedirs("figures/scores", exist_ok=True)
#     plt.savefig(f"figures/scores/{target_species}.pdf", bbox_inches="tight", transparent=True)

# %%
for target_species in target_species_list:
    target_species_scores = dfn[f"{pred_prefix}{target_species}"]
    valid_mask = ~np.isnan(target_species_scores)
    if np.sum(valid_mask) < n_sweep_steps:
        print(f"Species '{target_species}' has to few samples, skipping...")
        continue
    thresholds = [*get_thresholds(target_species_scores, n_sweep_steps), best_threshold[target_species], float("NaN")]
    threshold_types = ["sampled"] * (len(thresholds) - 2) + ["calibrated"] + ["gt"]
    for threshold, threshold_type in zip(thresholds, threshold_types):
        if threshold_type != "gt":
            dfn["observed"] = dfn[f"{pred_prefix}{target_species}"] >= threshold
        else:
            dfn["observed"] = dfn[f"gt_{target_species}"]

        print(dfn["observed"].sum(), f"positive observations with threshold {threshold}")


        tp = ( dfn["observed"] &  dfn[f"gt_{target_species}"]).sum()
        fp = ( dfn["observed"] & ~dfn[f"gt_{target_species}"]).sum()
        fn = (~dfn["observed"] &  dfn[f"gt_{target_species}"]).sum()
        tn = (~dfn["observed"] & ~dfn[f"gt_{target_species}"]).sum()

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = (2 * precision * recall) / (precision + recall)

        print(f"{target_species} precision {precision:.2f} recall {recall:.2f} f1 {f1:.2f}")

        
        for aggregation, pd_freq in [("month", "ME"), ("week", "W"), ("day", "D")]:

            # TODO: implement better way to detect actual deployment times

            dfa = dfn[["datetime", "location", "observed"]].groupby([pd.Grouper(key="datetime", freq=pd_freq), "location"]).sum(numeric_only=True).reset_index()
            dfa = dfa.pivot(columns="datetime", index="location", values="observed").sort_values(by="location")
            
            n_sites = dfn["location"].nunique()
            if aggregation == "month":
                L = np.tile(dfa.columns.days_in_month.values, (n_sites, 1))
            elif aggregation == "week":
                L = np.full((n_sites, len(dfa.columns)), 7)
            elif aggregation == "day":
                L = np.full((n_sites, len(dfa.columns)), 1)
            else:
                raise ValueError()

            site_covs = df_landcover[dfa["location"]][["forest_type", "elevation"]]

            robjects.globalenv["dfa"] = robjects.conversion.get_conversion().py2rpy(dfa)
            robjects.globalenv["L"] = robjects.conversion.get_conversion().py2rpy(L)
            robjects.globalenv["site_covs"] = robjects.conversion.get_conversion().py2rpy(site_covs)

            r_scripts = dict(
                BP=r'''
                library(unmarked)
                library(tidyverse)
                library(lubridate)

                ModelComparisonDF <- data.frame()

                umf <- unmarkedFrameOccu(y = (as.matrix(dfa) > 1) * 1)

                (psi_init <- mean(rowSums(getY(umf), na.rm = TRUE) > 0))
                (p_init <- mean(
                    getY(umf)[rowSums(getY(umf), na.rm = TRUE) > 0,] > 0, 
                    na.rm = T
                ))

                siteCovs(umf) <- site_covs

                beforetime = Sys.time()
                OccuMod <- occu(formula =  ~ 1 ~ 1,
                    data = umf,
                    method = "Nelder-Mead",
                    starts = c(qlogis(psi_init), qlogis(p_init))
                )
                aftertime = Sys.time()

                backTransform(OccuMod, type = "state")
                backTransform(OccuMod, type = "det")

                plogis(confint(OccuMod, type = 'state', method = 'normal'))
                plogis(confint(OccuMod, type = 'det', method = 'normal'))

                plogis(confint(OccuMod, type = 'state', method = 'normal', level = 0.50))
                plogis(confint(OccuMod, type = 'det', method = 'normal', level = 0.50))

                ModelComparisonDF <- bind_rows(ModelComparisonDF, data.frame(
                "psi_TransformedPointEstimate" = unname(coef(OccuMod)["psi(Int)"]),
                "psi_TransformedSE" = unname(SE(OccuMod)["psi(Int)"]),
                "psi_PointEstimate" = backTransform(OccuMod, type = "state")@estimate,
                "psi_CI95lower" = plogis(confint(OccuMod, type = 'state', method = 'normal'))[1],
                "psi_CI95upper" = plogis(confint(OccuMod, type = 'state', method = 'normal'))[2],
                "psi_CI50lower" = plogis(confint(OccuMod, type = 'state', method = 'normal', level = 0.50))[1],
                "psi_CI50upper" = plogis(confint(OccuMod, type = 'state', method = 'normal', level = 0.50))[2],
                "p_TransformedPointEstimate" = unname(coef(OccuMod)["p(Int)"]),
                "p_TransformedSE" = unname(SE(OccuMod)["p(Int)"]),
                "p_PointEstimate" = backTransform(OccuMod, type = "det")@estimate,
                "p_CI95lower" = plogis(confint(OccuMod, type = 'det', method = 'normal'))[1],
                "p_CI95upper" = plogis(confint(OccuMod, type = 'det', method = 'normal'))[2],
                "p_CI50lower" = plogis(confint(OccuMod, type = 'det', method = 'normal', level = 0.50))[1],
                "p_CI50upper" = plogis(confint(OccuMod, type = 'det', method = 'normal', level = 0.50))[2]
                ))
                ''',
                BP_FP=r'''
                library(unmarked)
                library(tidyverse)
                library(lubridate)

                ModelComparisonDF <- data.frame()

                y <- (as.matrix(dfa) > 1) * 1
                umf <- unmarkedFrameOccuFP(y=y, type=c(0,dim(y)[2],0))

                siteCovs(umf) <- site_covs

                (psi_init <- mean(rowSums(getY(umf), na.rm = TRUE) > 0))
                (p_init <- mean(
                    getY(umf)[rowSums(getY(umf), na.rm = TRUE) > 0,] > 0, 
                    na.rm = T
                ))

                beforetime = Sys.time()
                OccuModFP <- occuFP(detformula =  ~ 1, stateformula=~1, FPformula=~1,
                    data = umf,
                    method = "Nelder-Mead",
                    starts = c(qlogis(psi_init), qlogis(p_init), 0.5)
                )
                aftertime = Sys.time()

                backTransform(OccuModFP, type = "state")
                backTransform(OccuModFP, type = "det")

                plogis(confint(OccuModFP, type = 'state', method = 'normal'))
                plogis(confint(OccuModFP, type = 'det', method = 'normal'))

                plogis(confint(OccuModFP, type = 'state', method = 'normal', level = 0.50))
                plogis(confint(OccuModFP, type = 'det', method = 'normal', level = 0.50))

                ModelComparisonDF <- bind_rows(ModelComparisonDF, data.frame(
                "psi_TransformedPointEstimate" = unname(coef(OccuModFP)["psi(Int)"]),
                "psi_TransformedSE" = unname(SE(OccuModFP)["psi(Int)"]),
                "psi_PointEstimate" = backTransform(OccuModFP, type = "state")@estimate,
                "psi_CI95lower" = plogis(confint(OccuModFP, type = 'state', method = 'normal'))[1],
                "psi_CI95upper" = plogis(confint(OccuModFP, type = 'state', method = 'normal'))[2],
                "psi_CI50lower" = plogis(confint(OccuModFP, type = 'state', method = 'normal', level = 0.50))[1],
                "psi_CI50upper" = plogis(confint(OccuModFP, type = 'state', method = 'normal', level = 0.50))[2],
                "p_TransformedPointEstimate" = unname(coef(OccuModFP)["p(Int)"]),
                "p_TransformedSE" = unname(SE(OccuModFP)["p(Int)"]),
                "p_PointEstimate" = backTransform(OccuModFP, type = "det")@estimate,
                "p_CI95lower" = plogis(confint(OccuModFP, type = 'det', method = 'normal'))[1],
                "p_CI95upper" = plogis(confint(OccuModFP, type = 'det', method = 'normal'))[2],
                "p_CI50lower" = plogis(confint(OccuModFP, type = 'det', method = 'normal', level = 0.50))[1],
                "p_CI50upper" = plogis(confint(OccuModFP, type = 'det', method = 'normal', level = 0.50))[2]
                ))
                ''',
                COP=r'''
                library(unmarked)
                library(tidyverse)
                library(lubridate)

                ModelComparisonDF <- data.frame()

                umf = unmarkedFrameOccuCOP(
                    y = as.matrix(dfa),
                    L = matrix(
                        data = L,
                        nrow = nrow(dfa),
                        ncol = ncol(dfa),
                        dimnames = dimnames(dfa)
                    )
                )

                (psi_init <- mean(rowSums(getY(umf), na.rm = TRUE) > 0))
                (lambda_init <- mean((getY(umf) / getL(umf))[rowSums(getY(umf), na.rm = TRUE) > 0, ], na.rm = T))

                beforetime = Sys.time()
                OccuCOPMod <- occuCOP(
                    data = umf,
                    psiformula =  ~ 1,
                    lambdaformula =  ~ 1,
                    method = "Nelder-Mead",
                    psistarts = qlogis(psi_init),
                    lambdastarts = log(lambda_init)
                )
                aftertime = Sys.time()

                backTransform(OccuCOPMod, type = "psi")
                backTransform(OccuCOPMod, type = "lambda")

                plogis(confint(OccuCOPMod, type = 'psi', method = 'normal'))
                plogis(confint(OccuCOPMod, type = 'lambda', method = 'normal'))

                plogis(confint(OccuCOPMod, type = 'psi', method = 'normal', level = 0.50))
                plogis(confint(OccuCOPMod, type = 'lambda', method = 'normal', level = 0.50))

                ModelComparisonDF <- bind_rows(ModelComparisonDF, data.frame(
                "psi_TransformedPointEstimate" = unname(coef(OccuCOPMod)["psi(Int)"]),
                "psi_TransformedSE" = unname(SE(OccuCOPMod)["psi(Int)"]),
                "psi_PointEstimate" = backTransform(OccuCOPMod, type = "psi")@estimate,
                "psi_CI95lower" = plogis(confint(OccuCOPMod, type = 'psi', method = 'normal'))[1],
                "psi_CI95upper" = plogis(confint(OccuCOPMod, type = 'psi', method = 'normal'))[2],
                "psi_CI50lower" = plogis(confint(OccuCOPMod, type = 'psi', method = 'normal', level = 0.50))[1],
                "psi_CI50upper" = plogis(confint(OccuCOPMod, type = 'psi', method = 'normal', level = 0.50))[2],
                "lambda_TransformedPointEstimate" = unname(coef(OccuCOPMod)["lambda(Int)"]),
                "lambda_TransformedSE" = unname(SE(OccuCOPMod)["lambda(Int)"]),
                "lambda_PointEstimate" = backTransform(OccuCOPMod, type = "lambda")@estimate,
                "lambda_CI95lower" = exp(confint(OccuCOPMod, type = 'lambda', method = 'normal'))[1],
                "lambda_CI95upper" = exp(confint(OccuCOPMod, type = 'lambda', method = 'normal'))[2],
                "lambda_CI50lower" = plogis(confint(OccuCOPMod, type = 'lambda', method = 'normal', level = 0.50))[1],
                "lambda_CI50upper" = plogis(confint(OccuCOPMod, type = 'lambda', method = 'normal', level = 0.50))[2]
                ))
                ''',
            )

            for model in ["BP", "BP_FP", "COP"]:
                model_comparison_df = pd.DataFrame([{}])
                fitting_time_elapsed = float("NaN")
                try:
                    robjects.r(r_scripts[model])
                    model_comparison_df = robjects.conversion.get_conversion().rpy2py(robjects.r("ModelComparisonDF"))
                    fitting_time_elapsed = robjects.r("aftertime - beforetime").item()
                except Exception as e:
                    print(f"Got exception: {e}")

                assert len(model_comparison_df) == 1

                model_comparison_df["Discretisation"] = aggregation.title()
                model_comparison_df["Model"] = model
                model_comparison_df["species"] = target_species
                model_comparison_df["threshold"] = threshold
                model_comparison_df["threshold_type"] = threshold_type
                model_comparison_df["tp"] = tp
                model_comparison_df["fp"] = fp
                model_comparison_df["fn"] = fn
                model_comparison_df["tn"] = tn
                model_comparison_df["precision"] = precision
                model_comparison_df["recall"] = recall
                model_comparison_df["f1"] = f1
                model_comparison_df["fitting_time_elapsed"] = fitting_time_elapsed

                output_table.append(model_comparison_df)
                pd.concat(output_table).to_csv(output_table_path)