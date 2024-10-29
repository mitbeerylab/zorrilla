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
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()
from multiprocessing import Process, Queue, Event
import json
from functools import cache


tf = TimezoneFinder()
timezone_at = cache(tf.timezone_at)


py2r = ro.conversion.get_conversion().py2rpy
r2py = ro.conversion.get_conversion().rpy2py


# from astropy.time import TimeDelta
# x = list(range(48))
# y = []
# loc = coord.EarthLocation(lon=50 * u.deg, lat=5 * u.deg)
# for i in x:
#     time = Time('1999-01-01T00:00:00.123456789') + TimeDelta(i * 3600, format="sec")
#     altaz = coord.AltAz(location=loc, obstime=time)
#     s = coord.get_sun(time)
#     s.transform_to(altaz).alt
#     y.append(float(s.transform_to(altaz).alt / u.deg))

# import matplotlib.pyplot as plt
# plt.plot(x, y)

def get_thresholds(target_species_scores, n_sweep_steps):
    valid_mask = np.isfinite(target_species_scores)
    return np.linspace(np.min(target_species_scores[valid_mask]), np.max(target_species_scores[valid_mask]), n_sweep_steps)
    # return np.percentile(target_species_scores[valid_mask], np.linspace(0, 100, n_sweep_steps))

target_species_out = "lynx"
output_table_path = os.path.join("data", "iwildcam_2022_results_v6.csv")



os.chdir(os.path.dirname(__file__))  # TODO: remove

# dfo = rdata.read_rda(os.path.join("data", "metadata_Ain.RData"))["allfiles"]
dfn = pd.read_csv(os.path.join("..", "data", "iwildcam_2022_crops_bioclip_inference_logits_v3.csv"))

# parse datetimes and filter out rows missing datetimes
dfn["datetime"] = pd.to_datetime(dfn["datetime"])
print("Removing", dfn["datetime"].isna().sum(), "missing datetime values")
dfn = dfn[~dfn["datetime"].isna()]

dfn = dfn[np.isfinite(dfn["latitude"]) & np.isfinite(dfn["longitude"])]  # TODO: keep

df_dem = pd.read_csv(os.path.join("..", "data", "iwildcam_2022_dem.csv"))
df_landcover = pd.read_csv(os.path.join("..", "data", "iwildcam_2022_landcover.csv"))
df_region_labels = pd.read_csv(os.path.join("..", "data", "iwildcam_2022_region_labels.csv"))
covs = df_dem.merge(df_landcover, on="name", suffixes=("_dem", "_landcover")).merge(df_region_labels, left_on="name", right_on="location", suffixes=("", "_region_labels"))

# dfn = dfn.merge(df_dem, left_on="location", right_on="name", how="left", suffixes=("", "_dem")).merge(df_landcover, left_on="location", right_on="name", how="left", suffixes=("", "_landcover"))

# localize datetime timezones
datetimes_new = []
for _, row in dfn.iterrows():
    tzinfo = ZoneInfo(timezone_at(lng=row["longitude"], lat=row["latitude"]))
    datetimes_new.append(row["datetime"].tz_localize(tzinfo))
dfn["datetime_local"] = datetimes_new

# compute time since sunrise/sunset
astral_times = []
for _, row in dfn.iterrows():
    loc = LocationInfo(latitude=row["latitude"], longitude=row["longitude"])
    tzinfo = ZoneInfo(timezone_at(lng=row["longitude"], lat=row["latitude"]))
    s = sun.sun(loc.observer, date=row["datetime_local"], tzinfo=tzinfo)
    astral_times.append(s)
for k in astral_times[0].keys():
    assert k not in dfn.columns
    dfn[k] = [e[k] for e in astral_times]


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

print(f"number of sites overall: {dfn['location'].nunique()}")
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

# split along sequence IDs
train_seq, test_seq = train_test_split(dfn["seq_id"].unique(), test_size=0.8, random_state=42)
df_train, df_test = dfn[dfn["seq_id"].isin(train_seq)], dfn[dfn["seq_id"].isin(test_seq)]

if os.path.exists("cache/optimal_thresholds.json"):
    with open("cache/optimal_thresholds.json") as f:
        best_threshold = json.load(f)
        assert set(best_threshold.keys()) == set(target_species_list)
else:
    best_threshold = {}

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
    
    os.makedirs("cache", exist_ok=True)
    with open("cache/optimal_thresholds.json", "w") as f:
        json.dump(best_threshold, f)

# TODO: re-enable
# for region_label in sorted(covs["region_label"].unique()):
for region_labels in [[0, 3, 4]]:

    # use only test data from region to continue
    dfn = df_test[df_test["location"].isin(covs[covs["region_label"].isin(region_labels)]["name"].unique())]

    print(f"number of sites in regions '{region_labels}': {dfn['location'].nunique()}")

    gt_only = False
    output_table = []


    if not os.path.exists("figures/scores"):
        for target_species in target_species_list:
            fig, (ax1, ax2) = plt.subplots(1, 2, )
            target_species_scores = dfn[f"{pred_prefix}{target_species}"]
            valid_mask = np.isfinite(target_species_scores)

            thresholds = get_thresholds(target_species_scores, n_sweep_steps)
            ax1.hist(target_species_scores[valid_mask &  dfn[f"gt_{target_species}"]], bins=20, alpha=0.5, label="True")
            ax1.hist(target_species_scores[valid_mask & ~dfn[f"gt_{target_species}"]], bins=20, alpha=0.5, label="False")
            ax1.plot(thresholds, [0] * len(thresholds), 'o', label="Thresholds")
            ax1.set_yscale("log")
            
            precision = []
            recall = []
            for threshold in thresholds:
                predicted = np.array([target_species_out if row[f"{pred_prefix}{target_species}"] >= threshold else "other" for _, row in dfn.iterrows()])
                tp = ((predicted == target_species_out) &  dfn[f"gt_{target_species}"]).sum()
                fp = ((predicted == target_species_out) & ~dfn[f"gt_{target_species}"]).sum()
                fn = ((predicted != target_species_out) &  dfn[f"gt_{target_species}"]).sum()
                tn = ((predicted != target_species_out) & ~dfn[f"gt_{target_species}"]).sum()
                r = tp / (tp + fn)
                p = tp / (tp + fp)
                if np.isfinite([r, p]).all():
                    recall += [r]
                    precision += [p]

            ax2.plot(recall, precision)
            ax1.set_xlabel("Logit")
            ax1.set_ylabel("Number of Observations")
            ax2.set_xlabel("Recall")
            ax2.set_ylabel("Precision")
            fig.suptitle(target_species)
            os.makedirs("figures/scores", exist_ok=True)
            plt.savefig(f"figures/scores/{target_species}.pdf", bbox_inches="tight", transparent=True)

    region_species = [species for species in target_species_list if dfn[f"gt_{species}"].sum() > 0]
    for target_species in region_species:
        target_species_scores = dfn[f"{pred_prefix}{target_species}"]
        valid_mask = ~np.isnan(target_species_scores)
        if np.sum(valid_mask) < n_sweep_steps:
            print(f"Species '{target_species}' has to few samples, skipping...")
            continue
        thresholds = [float("NaN")] + [best_threshold[target_species]] + [*get_thresholds(target_species_scores, n_sweep_steps)]
        threshold_types = ["gt"] + ["calibrated"] + ["sampled"] * (len(thresholds) - 2)
        if gt_only:
            thresholds, threshold_types = thresholds[0:1], threshold_types[0:1]
        gt_state_df = {}
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

                # # TODO: keep?
                # valid_locations = set([row["name"] for _, row in covs.iterrows() if np.isfinite(row["region_label"])])
                # dfn = dfn[dfn["location"].isin(valid_locations)]

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

                site_covs = []
                for location in dfa.index:
                    covs_filtered = covs[covs["name"] == location]
                    if len(covs_filtered) == 0:
                        raise ValueError()
                        print(f"Warning: missing covariates for site {location}")
                        site_covs.append({})  # TODO: this should not happen
                    else:
                        site_covs.append(covs_filtered.iloc[0].to_dict())
                site_covs = pd.DataFrame(site_covs)
                site_covs = site_covs.rename(columns=lambda x: x.replace("-", "_"))
                # site_covs = pd.concat([site_covs, pd.get_dummies(site_covs["forest_type"])], axis=1).drop(columns=["forest_type"])

                # inits = []
                # c(qlogis(psi_init), qlogis(p_init), 0.5)
                # TODO: estimate init for false positive rate

                ro.globalenv["dfa"] = py2r(dfa)
                ro.globalenv["L"] = py2r(L)
                ro.globalenv["site_covs"] = py2r(site_covs)

                r_prefix = r'''
                    library(unmarked)
                    library(tidyverse)
                    library(lubridate)
                '''
                r_scripts = dict(
                    BP=r'''
                    umf <- unmarkedFrameOccu(y = (as.matrix(dfa) > 1) * 1)
                    siteCovs(umf) <- site_covs
                    beforetime = Sys.time()
                    mod <- occu(
                        # formula =  ~ 1 ~ 1,
                        formula = ~ 1 ~ tree_coverfraction,
                        # formula =  ~ 1 ~ elevation + as.factor(forest_type),
                        data = umf,
                    )
                    aftertime = Sys.time()
                    ''',
                    BP_FP=r'''
                    site <- data.frame(tree_coverfraction = site_covs$tree_coverfraction)
                    y <- (as.matrix(dfa) > 1) * 1
                    umf <- unmarkedFrameOccuFP(y=y, site, type=c(0,dim(y)[2],0))
                    beforetime = Sys.time()
                    mod = tryCatch({
                        occuFP(
                            detformula = ~ 1, stateformula= ~ tree_coverfraction, FPformula= ~ 1,
                            data = umf,
                        )
                    }, warning = function(warning_condition) {
                        NULL
                    })
                    aftertime = Sys.time()
                    ''',
                    COP=r'''
                    umf = unmarkedFrameOccuCOP(
                        y = as.matrix(dfa),
                        L = matrix(
                            data = L,
                            nrow = nrow(dfa),
                            ncol = ncol(dfa),
                            dimnames = dimnames(dfa)
                        ),
                        siteCovs=site_covs,
                    )
                    beforetime = Sys.time()
                    mod <- occuCOP(
                        data = umf,
                        psiformula =  ~ tree_coverfraction,
                        lambdaformula =  ~ 1,
                        # method = "Nelder-Mead",
                        # psistarts = qlogis(psi_init),
                        # lambdastarts = log(lambda_init)
                    )
                    aftertime = Sys.time()
                    ''',
                )

                # TODO: re-enable
                for model in ["BP", "BP_FP", "COP", "NAIVE"]:
                    print(f"Fitting model '{model}'")
                    model_comparison_df = pd.DataFrame([{}] * n_sites)
                    fitting_time_elapsed = float("NaN")
                    if model != "NAIVE":
                        try:
                            ro.r(r_prefix)
                            ro.r(r_scripts[model])
                            fitting_time_elapsed = ro.r("aftertime - beforetime").item()
                            if not ro.r["is.null"](ro.r("mod"))[0]:
                                model_comparison_df = r2py(ro.r("predict(mod, 'state')"))
                                model_comparison_df.index = model_comparison_df.index.rename("Site")
                                model_comparison_df = model_comparison_df.reset_index()
                                assert len(model_comparison_df) == n_sites
                        except Exception as e:
                            print(f"Got exception: {e}")
                    else:
                        naive_occupancy = (dfa > 0).any(axis=1).sum() / len(dfa)
                        model_comparison_df = pd.DataFrame([{ "Predicted": naive_occupancy }] * n_sites, index=dfa.index)

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