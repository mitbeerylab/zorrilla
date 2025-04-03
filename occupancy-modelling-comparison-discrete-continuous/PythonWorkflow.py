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
from functools import cache, partial
from collections import defaultdict
from biolith.models import occu, occu_cop, occu_cs, occu_rn
from biolith.utils import fit
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from utils import uniform_sample_by_column, calibrate_threshold, time_limit
from joblib import Memory


memory = Memory("cache")


uniform_sample_by_column = memory.cache(uniform_sample_by_column)
calibrate_threshold = memory.cache(calibrate_threshold)


random_seed = 42
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



def mcmc_get_results(results, original_index, confidence_level=0.95):
    samples = results.samples
    PredictionKey = "psi" if "psi" in samples else ("abundance" if "abundance" in samples else None)
    try:
        num_divergences = results.mcmc.get_extra_fields()["diverging"].sum().item()
    except:
        num_divergences = float("nan")
    results_df = pd.DataFrame(dict(
        Predicted=samples[PredictionKey].mean(axis=0),
        DetectionProb=samples["prob_detection"].mean(axis=(0, 1)) if "prob_detection" in samples else None,
        DetectionRate=samples["rate_detection"].mean(axis=(0, 1)) if "rate_detection" in samples else None,
        Abundance=samples["abundance"].mean(axis=(0)) if "abundance" in samples else None,
        FPProb=samples["prob_fp_constant"].mean() if "prob_fp_constant" in samples else None,
        FPUnoccupiedProb=samples["prob_fp_unoccupied"].mean() if "prob_fp_unoccupied" in samples else None,

        PredictedHPDILower=hpdi(samples[PredictionKey], confidence_level)[0],
        PredictedHPDIUpper=hpdi(samples[PredictionKey], confidence_level)[1],

        DetectionProbHPDILower=hpdi(samples["prob_detection"].reshape(-1, samples["prob_detection"].shape[-1]), confidence_level)[0] if "prob_detection" in samples else None,
        DetectionProbHPDIUpper=hpdi(samples["prob_detection"].reshape(-1, samples["prob_detection"].shape[-1]), confidence_level)[1] if "prob_detection" in samples else None,

        DetectionRateHPDILower=hpdi(samples["rate_detection"].reshape(-1, samples["rate_detection"].shape[-1]), confidence_level)[0] if "rate_detection" in samples else None,
        DetectionRateHPDIUpper=hpdi(samples["rate_detection"].reshape(-1, samples["rate_detection"].shape[-1]), confidence_level)[1] if "rate_detection" in samples else None,

        AbundanceHPDILower=hpdi(samples["abundance"], confidence_level)[0] if "abundance" in samples else None,
        AbundanceHPDIUpper=hpdi(samples["abundance"], confidence_level)[1] if "abundance" in samples else None,

        FPProbHPDILower=hpdi(samples["prob_fp_constant"], confidence_level)[0] if "prob_fp_constant" in samples else None,
        FPProbHPDIUpper=hpdi(samples["prob_fp_constant"], confidence_level)[1] if "prob_fp_constant" in samples else None,

        FPUnoccupiedProbHPDILower=hpdi(samples["prob_fp_unoccupied"], confidence_level)[0] if "prob_fp_unoccupied" in samples else None,
        FPUnoccupiedProbHPDIUpper=hpdi(samples["prob_fp_unoccupied"], confidence_level)[1] if "prob_fp_unoccupied" in samples else None,

        **{f'{k}': samples[k].mean() for k in samples.keys() if k.startswith("cov_")},
        **{f'{k}_se': samples[k].std() for k in samples.keys() if k.startswith("cov_")},

        **{f'{k}_hdpi_lower': hpdi(samples[k], confidence_level)[0] for k in samples.keys() if k.startswith("cov_")},
        **{f'{k}_hdpi_upper': hpdi(samples[k], confidence_level)[1] for k in samples.keys() if k.startswith("cov_")},

        num_divergences=num_divergences,
    ), index=original_index)

    return results_df



os.chdir(os.path.dirname(__file__))  # TODO: remove


output_table = []
output_table_path = os.path.join("data", f"iwildcam_2022_results_v22_combined.csv")
for classifier in ["cameratrapai", "bioclip"]:

    # dfo = rdata.read_rda(os.path.join("data", "metadata_Ain.RData"))["allfiles"]
    dfn_path = dict(
        bioclip=os.path.join("..", "data", "iwildcam_2022_crops_bioclip_inference_logits_v3.csv"),
        cameratrapai=os.path.join("..", "data", "iwildcam_2022_cameratrapai_pt_train.csv"),
    )
    # dfn = pd.read_csv()
    dfn = pd.read_csv(dfn_path[classifier])

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

    dfn["hours_since_sunrise"] = ((dfn["datetime_local"] - dfn["sunrise"]) / np.timedelta64(1, "h")) % 24
    dfn["hours_since_sunset"] = ((dfn["datetime_local"] - dfn["sunset"]) / np.timedelta64(1, "h")) % 24
    dfn["hours_since_sunrise"] = (dfn["hours_since_sunrise"] - dfn["hours_since_sunrise"].mean()) / dfn["hours_since_sunrise"].std()
    dfn["hours_since_sunset"] = (dfn["hours_since_sunset"] - dfn["hours_since_sunset"].mean()) / dfn["hours_since_sunset"].std()

    # TODO: make configurable
    covs["tree-coverfraction"] = (covs["tree-coverfraction"] - covs["tree-coverfraction"].mean()) / covs["tree-coverfraction"].std()
    covs["elevation"] = (covs["elevation"] - covs["elevation"].mean()) / covs["elevation"].std()

    # dfn["hours_since_sunrise"] = np.random.randn(len(dfn))
    # dfn["hours_since_sunset"] = np.random.randn(len(dfn))


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
    cal_prefix = "cal_"
    n_sweep_steps = 11
    calibration_min_samples = 10
    filter_topn = True
    filter_topn_rank = 3
    n_calibration_samples_per_species = 100

    # target_species_list = None
    target_species_list = [e.replace(" ", "_") for e in [
        "tayassu pecari",
        "meleagris ocellata",
        # "equus quagga",
        # "madoqua guentheri",
        "leopardus pardalis",
        # "giraffa camelopardalis",
        # "sus scrofa",
        "mazama pandora",
        # "litocranius walleri",
        # "formicarius analis",
        # "tragulus javanicus",
        # "tupinambis teguixin",
    ]]

    for target_species in target_species_list:
        scores = dfn[f"{pred_prefix}{target_species}"] = dfn[f"{pred_prefix}{target_species}"].fillna(value=-float("inf"))

    # # split along sequence IDs
    # train_seq, test_seq = train_test_split(dfn["seq_id"].unique(), test_size=0.8, random_state=42)
    # df_train, df_test = dfn[dfn["seq_id"].isin(train_seq)].copy(), dfn[dfn["seq_id"].isin(test_seq)].copy()

    # from sklearn.linear_model import LogisticRegression
    # from sklearn.calibration import CalibratedClassifierCV

    # for target_species in target_species_list:
    #     calibrated_classifier = CalibratedClassifierCV(LogisticRegression(random_state=random_seed))
    #     train_finite_idx = np.isfinite(df_train[f"{pred_prefix}{target_species}"])
    #     calibrated_classifier.fit(df_train[f"{pred_prefix}{target_species}"][train_finite_idx].to_numpy()[:, None], df_train[f"gt_{target_species}"][train_finite_idx].to_numpy())
    #     df_test[f"{cal_prefix}{target_species}"] = calibrated_classifier.predict_proba(np.nan_to_num(df_test[f"{pred_prefix}{target_species}"].to_numpy(), neginf=-1e6)[:, None])[:, 1]

    # TODO: re-enable
    # for region_label in sorted(covs["region_label"].unique()):
    for region_labels in [[0, 3, 4]]:

        # use only test data from region to continue
        dfn = dfn[dfn["location"].isin(covs[covs["region_label"].isin(region_labels)]["name"].unique())]

        print(f"number of sites in regions '{region_labels}': {dfn['location'].nunique()}")

        best_threshold = {}
        threshold_fp_calibration = {}
        score_distributions = {}
        for target_species in target_species_list:
            df_labeled = uniform_sample_by_column(dfn, f"{pred_prefix}{target_species}", n_calibration_samples_per_species)
            calibrated_threshold, thresholds, false_positive_rates, false_positive_rates_std, mu0_mean, mu0_std, mu1_mean, mu1_std, sigma0_mean, sigma0_std, sigma1_mean, sigma1_std = calibrate_threshold(dfn, df_labeled, f"{pred_prefix}{target_species}", f"gt_{target_species}")
            best_threshold[target_species] = calibrated_threshold
            threshold_fp_calibration[target_species] = dict(threshold=thresholds, fpr=false_positive_rates, fpr_std=false_positive_rates_std)
            score_distributions[target_species] = dict(mu0_mean=mu0_mean, mu0_std=mu0_std, mu1_mean=mu1_mean, mu1_std=mu1_std, sigma0_mean=sigma0_mean, sigma0_std=sigma0_std, sigma1_mean=sigma1_mean, sigma1_std=sigma1_std)

        gt_only = False


        if not os.path.exists(f"figures/scores_{classifier}"):
            for target_species in target_species_list:
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
                target_species_scores = dfn[f"{pred_prefix}{target_species}"]
                valid_mask = np.isfinite(target_species_scores)

                thresholds = get_thresholds(target_species_scores, n_sweep_steps)
                ax1.hist(target_species_scores[valid_mask &  dfn[f"gt_{target_species}"]], bins=20, alpha=0.5, label="True")
                ax1.hist(target_species_scores[valid_mask & ~dfn[f"gt_{target_species}"]], bins=20, alpha=0.5, label="False")
                ax1.plot(thresholds, [0] * len(thresholds), 'o', label="Thresholds")
                ax1.set_yscale("log")
                
                precision = []
                recall = []
                threshold_valid = []
                for threshold in thresholds:
                    predicted = dfn[f"{pred_prefix}{target_species}"] >= threshold
                    tp = (( predicted) &  dfn[f"gt_{target_species}"]).sum()
                    fp = (( predicted) & ~dfn[f"gt_{target_species}"]).sum()
                    fn = ((~predicted) &  dfn[f"gt_{target_species}"]).sum()
                    tn = ((~predicted) & ~dfn[f"gt_{target_species}"]).sum()
                    r = tp / (tp + fn)
                    p = tp / (tp + fp)
                    if np.isfinite([r, p]).all():
                        recall += [r]
                        precision += [p]
                        threshold_valid += [threshold]

                ax2.plot(recall, precision)
                ax3.plot(threshold_valid, precision)
                ax4.plot(threshold_valid, recall)
                ax1.set_xlabel("Logit")
                ax1.set_ylabel("Number of Observations")
                ax2.set_xlabel("Recall")
                ax2.set_ylabel("Precision")
                ax3.set_xlabel("Logit")
                ax3.set_ylabel("Precision")
                ax4.set_xlabel("Logit")
                ax4.set_ylabel("Recall")
                fig.suptitle(target_species)
                os.makedirs(f"figures/scores_{classifier}", exist_ok=True)
                plt.savefig(f"figures/scores_{classifier}/{target_species}.pdf", bbox_inches="tight", transparent=True)

        region_species = [species for species in target_species_list if dfn[f"gt_{species}"].sum() > 0]
        for target_species in region_species:
            target_species_gt_col = f"gt_{target_species}"
            target_species_score_col = f"{pred_prefix}{target_species}"
            target_species_prob_col = f"{cal_prefix}{target_species}"
            target_species_scores = dfn[target_species_score_col]
            valid_mask = ~np.isnan(target_species_scores)
            if np.sum(valid_mask) < n_sweep_steps:
                print(f"Species '{target_species}' has to few samples, skipping...")
                continue
            thresholds = [float("NaN")] + [float("NaN")] + [best_threshold[target_species]] + [*get_thresholds(target_species_scores, n_sweep_steps)]
            threshold_types = ["gt"] + ["topn"] + ["calibrated"] + ["sampled"] * (len(thresholds) - 3)  # TODO: re-add "probabilistic"?
            if gt_only:
                thresholds, threshold_types = thresholds[0:1], threshold_types[0:1]
            gt_state_df = {}
            assert len(thresholds) == len(threshold_types)
            for threshold, threshold_type in zip(thresholds, threshold_types):
                if threshold_type == "gt":
                    dfn["observed"] = dfn[target_species_gt_col]
                elif threshold_type == "probabilistic":
                    dfn["observed"] = dfn[target_species_prob_col]
                else:
                    dfn["observed"] = dfn[target_species_score_col] >= threshold                

                print(dfn["observed"].sum(), f"positive observations with threshold {threshold}")


                tp = ( (dfn["observed"] >= 0.5) &  dfn[target_species_gt_col]).sum()
                fp = ( (dfn["observed"] >= 0.5) & ~dfn[target_species_gt_col]).sum()
                fn = (~(dfn["observed"] >= 0.5) &  dfn[target_species_gt_col]).sum()
                tn = (~(dfn["observed"] >= 0.5) & ~dfn[target_species_gt_col]).sum()

                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                fpr = fp / (fp + tn)
                f1 = (2 * precision * recall) / (precision + recall)

                print(f"{target_species} precision {precision:.2f} recall {recall:.2f} f1 {f1:.2f}")

                for site_covs_list, obs_covs_list in [
                    (["elevation", "tree_coverfraction"], ["hours_since_sunrise", "hours_since_sunset"]),
                    ([], []),
                ]:
                    for enable_naive_fp_prior in [True, False]:
                        for aggregation, pd_freq in [("month", "ME"), ("week", "W"), ("day", "D")]:

                            # # TODO: keep?
                            # valid_locations = set([row["name"] for _, row in covs.iterrows() if np.isfinite(row["region_label"])])
                            # dfn = dfn[dfn["location"].isin(valid_locations)]

                            # TODO: implement better way to detect actual deployment times
                            if threshold_type == "topn":
                                dfa = dfn[["datetime", "location", target_species_score_col]].groupby([pd.Grouper(key="datetime", freq=pd_freq), "location"]).apply(lambda x: np.sort(x[target_species_score_col])).reset_index()
                                dfa = dfa.pivot(columns="datetime", index="location").sort_values(by="location")
                                dfa.columns = dfa.columns.get_level_values(1)
                                topn = dfn[["datetime", "location"] + [e for e in dfn.columns if e.startswith(pred_prefix)]].melt(id_vars=["datetime", "location"], value_name="score").groupby([pd.Grouper(key="datetime", freq=pd_freq), "location"]).agg({"score": lambda x: (np.sort(x)[-filter_topn_rank] if (len(x) >= filter_topn_rank) else np.sort(x)[0])}).reset_index()
                                topn = topn.pivot(columns="datetime", index="location", values="score").sort_values(by="location")
                                assert dfa.shape == topn.shape
                                for i, idx in enumerate(dfa.index):
                                    for j, col in enumerate(dfa.columns):
                                        try:
                                            thres = topn.iloc[i, j]
                                        except IndexError:
                                            thres = float("inf")
                                        dfa.iloc[i, j] = (np.array(dfa.iloc[i, j]) >= thres).sum().item()
                            elif threshold_type == "probabilistic":
                                dfa = dfn[["datetime", "location", "observed"]].groupby([pd.Grouper(key="datetime", freq=pd_freq), "location"]).max(numeric_only=True).reset_index()
                                dfa = dfa.pivot(columns="datetime", index="location", values="observed").sort_values(by="location")
                            else:
                                dfa = dfn[["datetime", "location", "observed"]].groupby([pd.Grouper(key="datetime", freq=pd_freq), "location"]).sum(numeric_only=True).reset_index()
                                dfa = dfa.pivot(columns="datetime", index="location", values="observed").sort_values(by="location")
                            
                            # compute score data for continuous score model
                            dfs = dfn[["datetime", "location", target_species_score_col]].groupby([pd.Grouper(key="datetime", freq=pd_freq), "location"]).apply(lambda x: np.max(x[target_species_score_col])).reset_index()
                            dfs = dfs.pivot(columns="datetime", index="location").sort_values(by="location")
                            dfs.columns = dfs.columns.get_level_values(1)

                            y = ((dfa >= 1) * 1).where(dfa.notna(), np.nan)
                            
                            n_sites = dfn["location"].nunique()
                            if aggregation == "month":
                                L = np.tile(dfa.columns.days_in_month.values, (n_sites, 1))
                            elif aggregation == "week":
                                L = np.full((n_sites, len(dfa.columns)), 7)
                            elif aggregation == "day":
                                L = np.full((n_sites, len(dfa.columns)), 1)
                            else:
                                raise ValueError()
                            
                            # get number of non-na entries in dfa
                            average_images_per_aggregation_window = len(dfn) / dfa.notna().sum().sum()  # TODO: is this correct?
                            average_images_per_day = average_images_per_aggregation_window / np.mean(L)

                            # TODO: keep?
                            # # rename columns because rpy2 does not handle datetimes well
                            # dfa.columns = range(len(dfa.columns))

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

                            obs_covs = dfn[["datetime", "location"] + obs_covs_list].groupby([pd.Grouper(key="datetime", freq=pd_freq), "location"]).median().reset_index()
                            obs_covs = obs_covs.pivot(columns="datetime", index="location", values=obs_covs_list).sort_values(by="location")
                            obs_covs_dict = {}
                            for cov_name in set(obs_covs.columns.get_level_values(0)):
                                obs_covs_dict[cov_name] = obs_covs[cov_name]

                            # inits = []
                            # c(qlogis(psi_init), qlogis(p_init), 0.5)
                            # TODO: estimate init for false positive rate

                            for implementation in ["R", "NumPyro"]:

                                r_prefix = r'''
                                    library(unmarked)
                                    library(tidyverse)
                                    library(lubridate)
                                '''
                                r_scripts = dict(
                                    BP=f'''
                                    umf <- unmarkedFrameOccu(y = as.matrix(y), siteCovs=site_covs, obsCovs=obs_covs)
                                    beforetime = Sys.time()
                                    mod = occu(
                                        data = umf,
                                        formula = ~ {" + ".join(['1'] + obs_covs_list)} ~ {" + ".join(['1'] + site_covs_list)},
                                        # starts = c(qlogis(psi_init), qlogis(p_init), 0, 0),
                                        method="Nelder-Mead"
                                    )
                                    aftertime = Sys.time()
                                    ''',
                                    BP_FP=f'''
                                    umf <- unmarkedFrameOccuFP(y=as.matrix(y), site_covs, obs_covs, type=c(0,dim(y)[2],0))
                                    beforetime = Sys.time()
                                    mod = occuFP(
                                        data = umf,
                                        detformula = ~ {" + ".join(['1'] + obs_covs_list)}, stateformula= ~ {" + ".join(['1'] + site_covs_list)}, FPformula= ~ 1,
                                        # starts = c(qlogis(psi_init), qlogis(p_init), fpr_init, 0, 0, 0, 0, 0, 0),
                                        method="Nelder-Mead"
                                    )
                                    aftertime = Sys.time()
                                    ''',
                                    COP=f'''
                                    umf = unmarkedFrameOccuCOP(
                                        y = as.matrix(dfa),
                                        L = matrix(
                                            data = L,
                                            nrow = nrow(dfa),
                                            ncol = ncol(dfa),
                                            dimnames = dimnames(dfa)
                                        ),
                                        siteCovs=site_covs,
                                        obsCovs=obs_covs
                                    )
                                    beforetime = Sys.time()
                                    mod = occuCOP(
                                        data = umf,
                                        psiformula = ~ {" + ".join(['1'] + site_covs_list)}, lambdaformula = ~ {" + ".join(['1'] + obs_covs_list)},
                                        # psistarts = qlogis(psi_init), lambdastarts = log(mean(as.matrix(dfa)[rowSums(as.matrix(dfa)) > 0, ])),
                                        method="Nelder-Mead",
                                        L1=TRUE,
                                    )
                                    aftertime = Sys.time()
                                    ''',
                                    RN=f'''
                                    umf <- unmarkedFrameOccu(y = as.matrix(y), siteCovs=site_covs, obsCovs=obs_covs)
                                    beforetime = Sys.time()
                                    mod = occuRN(
                                        data = umf,
                                        formula = ~ {" + ".join(['1'] + obs_covs_list)} ~ {" + ".join(['1'] + site_covs_list)},
                                        # starts = c(qlogis(psi_init), qlogis(p_init), 0, 0),
                                        method="Nelder-Mead",
                                        K=25,
                                    )
                                    aftertime = Sys.time()
                                    ''',
                                )

                                naive_occupancy = float("NaN")
                                naive_detection_probability = float("NaN")
                                naive_false_positive_prob = float("NaN")
                                naive_false_positive_prob_std = float("NaN")
                                naive_false_positive_rate = float("NaN")
                                naive_false_positive_rate_std = float("NaN")
                            
                                for model in ["NAIVE", "BP", "BP_FP", "COP", "CS", "RN"]:
                                    print(f"Fitting model '{model}'")
                                    parameter_names = {
                                        "state": defaultdict(lambda: "state", { "COP": "psi" })[model],
                                        "det": defaultdict(lambda: "det", { "COP": "lambda" })[model],
                                    }
                                    model_comparison_df = pd.DataFrame([{}] * n_sites, index=dfa.index)
                                    fitting_time_elapsed = float("NaN")
                                    if model != "NAIVE":
                                        if implementation == "R":
                                            if model == "CS":
                                                continue
                                            try:
                                                with robjects.local_context() as rctx:
                                                    rctx["dfa"] = py2r(dfa)
                                                    rctx["y"] = py2r(y)
                                                    rctx["L"] = py2r(L)
                                                    if len(site_covs_list) > 0:
                                                        rctx["site_covs"] = py2r(site_covs)
                                                    else:
                                                        ro.r("site_covs <- NULL")
                                                    if len(obs_covs_list) > 0:
                                                        ro.r("obs_covs <- list()")
                                                        ro.r("obs_covs_names <- list()")
                                                        for obs_covs_name, obs_covs_df in obs_covs_dict.items():
                                                            rctx[obs_covs_name] = py2r(obs_covs_df)
                                                        obs_covs_list_str = ",".join([f'"{obs_covs_name}" = as.matrix({obs_covs_name})' for obs_covs_name in obs_covs_dict.keys()])
                                                        ro.r(f"obs_covs <- list({obs_covs_list_str})")
                                                    else:
                                                        ro.r("obs_covs <- NULL")
                                                        ro.r("obs_covs_names <- list()")
                                                    rctx["psi_init"] = naive_occupancy
                                                    rctx["p_init"] = naive_detection_probability
                                                    rctx["fpr_init"] = naive_false_positive_prob
                                                    ro.r(r_prefix)
                                                    ro.r(r_scripts[model])
                                                    fitting_time_elapsed = ro.r("aftertime - beforetime").item()
                                                    success = not ro.r["is.null"](ro.r("mod"))[0]
                                                    if success:
                                                        model_comparison_df = r2py(ro.r(f"predict(mod, '{parameter_names['state']}')"))
                                                        model_comparison_df.index = dfa.index
                                                        try:
                                                            model_comparison_df["FPProb"] = r2py(ro.r(f"predict(mod, type = 'fp')"))["Predicted"]
                                                        except:
                                                            pass
                                                        for cov_type in ["state", "det"]:
                                                            for cov_name, cov_value, cov_se in zip(r2py(ro.r(f"names(mod@estimates['{parameter_names[cov_type]}']@estimates)")), r2py(ro.r(f"mod@estimates['{parameter_names[cov_type]}']@estimates")), r2py(ro.r(f"SE(mod@estimates['{parameter_names[cov_type]}'])"))):
                                                                model_comparison_df[f"cov_{cov_type}_{cov_name.replace('(Intercept)', 'intercept')}"] = cov_value
                                                                model_comparison_df[f"cov_{cov_type}_{cov_name.replace('(Intercept)', 'intercept')}_se"] = cov_value
                                                        assert len(model_comparison_df) == n_sites
                                                        model_comparison_df["sucess"] = True
                                            except Exception as e:
                                                print(f"Got exception: {e}")
                                        elif implementation == "NumPyro":
                                            if threshold_type in {"sampled", "calibrated"}:
                                                assert np.isfinite(threshold)
                                                assert np.isfinite(naive_false_positive_prob)
                                                assert np.isfinite(naive_false_positive_prob_std)
                                            extra_args = dict(
                                                BP=dict(),
                                                BP_FP=dict(prior_prob_fp_constant=dist.TruncatedDistribution(dist.Normal(naive_false_positive_prob, naive_false_positive_prob_std), low=0, high=1)),
                                                COP=dict(prior_rate_fp_constant=dist.TruncatedDistribution(dist.Normal(naive_false_positive_rate, naive_false_positive_rate_std), low=0)),  # TODO: implement naive false positive rate?
                                                CS=dict(prior_mu=(dist.Normal(score_distributions[target_species]["mu0_mean"], score_distributions[target_species]["mu0_std"]), dist.Normal(score_distributions[target_species]["mu1_mean"], score_distributions[target_species]["mu1_std"])), prior_sigma=(dist.TruncatedNormal(score_distributions[target_species]["sigma0_mean"], score_distributions[target_species]["sigma0_std"], low=0), dist.TruncatedNormal(score_distributions[target_species]["sigma1_mean"], score_distributions[target_species]["sigma1_std"], low=0))),
                                                RN=dict(prior_prob_fp_constant=dist.TruncatedDistribution(dist.Normal(naive_false_positive_prob, naive_false_positive_prob_std), low=0, high=1)),
                                            ) if (enable_naive_fp_prior and np.isfinite(threshold)) else dict()

                                            model_fn = dict(
                                                BP=partial(occu, false_positives_constant=False, **extra_args.get(model, {})),
                                                BP_FP=partial(occu, false_positives_constant=True, **extra_args.get(model, {})),
                                                COP=partial(occu_cop, false_positives_constant=True, session_duration=L, **extra_args.get(model, {})),
                                                CS=partial(occu_cs, **extra_args.get(model, {})),
                                                RN=partial(occu_rn, false_positives_constant=True, max_abundance=25, **extra_args.get(model, {})),
                                            )[model]
                                            obs = dict(
                                                BP=y,
                                                BP_FP=y,
                                                COP=dfa,
                                                CS=dfs,
                                                RN=y,
                                            )[model]
                                            try:
                                                with time_limit(300):  # 5 minutes
                                                    results = fit(model_fn, site_covs[site_covs_list], obs_covs[obs_covs_list], obs=obs, num_samples=500, num_warmup=500, num_chains=3)
                                                model_comparison_df = mcmc_get_results(results, obs.index)
                                                model_comparison_df["sucess"] = True
                                            except Exception as e:
                                                print(f"Got exception: {e}")
                                                pass
                                    else:
                                        naive_occupancy = (dfa > 0).any(axis=1).mean()

                                        # TODO: the R version taken from the original code differs from the pure Python re-implementation below. Check which one is correct.
                                        with robjects.local_context() as rctx:
                                            ro.r(r_prefix)
                                            rctx["dfa"] = py2r(dfa)
                                            ro.r("umf <- unmarkedFrameOccu(y = (as.matrix(dfa) > 1) * 1)")
                                            naive_detection_probability = ro.r("mean(getY(umf)[rowSums(getY(umf), na.rm = TRUE) > 0,] > 0, na.rm = TRUE)").item()
                                        # naive_detection_probability = (dfa[(dfa > 0).any(axis=1)] > 0).mean(axis=1).mean()
                                        
                                        # TODO: is this correct?
                                        if np.isfinite(threshold):
                                            naive_false_positive_prob_per_image = np.interp(threshold, threshold_fp_calibration[target_species]["threshold"], threshold_fp_calibration[target_species]["fpr"])
                                            naive_false_positive_prob_per_image_std = np.interp(threshold, threshold_fp_calibration[target_species]["threshold"], threshold_fp_calibration[target_species]["fpr_std"])

                                            rng = np.random.default_rng(seed=42)
                                            naive_false_positive_prob_per_image_samples = rng.normal(naive_false_positive_prob_per_image, naive_false_positive_prob_per_image_std, 100).clip(0, 1)
                                            
                                            naive_false_positive_prob_samples = 1 - (1 - naive_false_positive_prob_per_image_samples) ** int(round(average_images_per_aggregation_window))
                                            naive_false_positive_prob = np.mean(naive_false_positive_prob_samples)
                                            naive_false_positive_prob_std = np.std(naive_false_positive_prob_samples)
                                            
                                            naive_false_positive_rate_samples = naive_false_positive_prob_per_image_samples * average_images_per_day
                                            naive_false_positive_rate = np.mean(naive_false_positive_rate_samples)
                                            naive_false_positive_rate_std = np.std(naive_false_positive_rate_samples)

                                            assert 0 <= naive_false_positive_prob <= 1
                                            assert 0 <= naive_false_positive_rate

                                            assert np.isfinite(naive_false_positive_prob)
                                            assert np.isfinite(naive_false_positive_prob_std)
                                            assert np.isfinite(naive_false_positive_rate)
                                            assert np.isfinite(naive_false_positive_rate_std)

                                        model_comparison_df = pd.DataFrame([{ "Predicted": naive_occupancy }] * n_sites, index=dfa.index)
                                        model_comparison_df["sucess"] = True

                                    model_comparison_df["classifier"] = classifier
                                    model_comparison_df["site_covs"] = ",".join(site_covs_list)
                                    model_comparison_df["obs_covs"] = ",".join(obs_covs_list)
                                    model_comparison_df["implementation"] = implementation
                                    model_comparison_df["naive_fp_prior"] = enable_naive_fp_prior
                                    model_comparison_df["naive_false_positive_prob"] = naive_false_positive_prob
                                    model_comparison_df["naive_false_positive_prob_std"] = naive_false_positive_prob_std
                                    model_comparison_df["naive_false_positive_rate"] = naive_false_positive_rate
                                    model_comparison_df["naive_false_positive_rate_std"] = naive_false_positive_rate_std
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
                                    model_comparison_df["fpr"] = fpr
                                    model_comparison_df["f1"] = f1
                                    model_comparison_df["fitting_time_elapsed"] = fitting_time_elapsed
                                    model_comparison_df["sucess"] = "sucess" in model_comparison_df and model_comparison_df["sucess"]
                                    
                                    output_table.append(model_comparison_df)
                                pd.concat(output_table).to_csv(output_table_path)