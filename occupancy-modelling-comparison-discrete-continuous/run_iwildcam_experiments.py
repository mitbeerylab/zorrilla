import os
from collections import Counter
# import rdata
import pyreadr
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

from multiprocessing import Process, Queue, Event


def evaluate_r_occupancy(rscript, queue, finished_flag):
    try:
        model_comparison_df = None
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri

        for i, block in enumerate(re.split(r"#[# ]*-+.*\n", rscript)):
            print(f"Processing block #{i}")
            # for block in rscript.split("## ----class.source = 'fold-show'-----------------------------------------------"):
            try:
                robjects.r(block)
            except Exception as e:
                print(f"Exception during R script evaluation at block {i}:", e)
                print("R code block below")
                print(block)

            try:
                with (robjects.default_converter + pandas2ri.converter).context():
                    model_comparison_df = robjects.conversion.get_conversion().rpy2py(robjects.r('ModelComparisonDF')).copy()
                queue.put(model_comparison_df)
            except:
                pass
    finally:
        finished_flag.set()


def get_thresholds(target_species_scores, n_sweep_steps):
    valid_mask = ~np.isnan(target_species_scores)
    return np.linspace(np.min(target_species_scores[valid_mask]), np.max(target_species_scores[valid_mask]), n_sweep_steps)
    # return np.percentile(target_species_scores[valid_mask], np.linspace(0, 100, n_sweep_steps))


def main():
    original_cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))

    target_species_out = "lynx"
    output_path = os.path.join(os.path.dirname(__file__), "data", "metadata_iwildcam_2022_tmp_v3.RData")
    output_table_path = os.path.join(os.path.dirname(__file__), "data", "iwildcam_2022_results_v4.csv")
    # assert not os.path.exists(output_path)


    # dfo = rdata.read_rda(os.path.join(os.path.dirname(__file__), "data", "metadata_Ain.RData"))["allfiles"]
    dfn = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "iwildcam_2022_crops_bioclip_inference_logits_v3.csv"))

    print(f"number of sites: {dfn['location'].nunique()}")
    # print(f"number of observations: {Counter(dfo['observed'].tolist())}")

    with open(os.path.join(os.path.dirname(__file__), "Ain_lynx_occupancy.R"), "rt") as f:
        rscript = f.read()

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

    best_threshold = {}

    df_train, df_test = train_test_split(dfn, test_size=0.8, random_state=42)

    for target_species in target_species_list:
        gt = df_train[f"gt_{target_species}"]
        scores = df_train[f"{pred_prefix}{target_species}"]
        pos_scores = scores[ df_train[f"gt_{target_species}"]]
        neg_scores = scores[~df_train[f"gt_{target_species}"]]
        assert len(pos_scores) >= calibration_min_samples and len(neg_scores) > calibration_min_samples
        f1 = []
        for s in scores:
            tp = ((scores >= s) &  gt).sum()
            fp = ((scores >= s) & ~gt).sum()
            fn = ((scores <  s) &  gt).sum()
            tn = ((scores <  s) & ~gt).sum()
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 += [(2 * precision * recall) / (precision + recall)]
        f1 = np.array(f1)
        assert len(f1[np.isfinite(f1)]) > 0
        best_score = np.mean(scores[f1 == f1[np.isfinite(f1)].max()])
        print(f"Best threshold for species '{target_species}' is {best_score:.2f} at f1 of {f1[np.isfinite(f1)].max():.2f}")
        best_threshold[target_species] = best_score

    # use only test data to continue
    dfn = df_test

    output_table = []
    try:
        available_species = [e.replace(pred_prefix, "") for e in dfn.columns if e.startswith(pred_prefix) and e != f"{pred_prefix}empty"]
        if target_species_list is None:
            target_species_list = available_species

        import matplotlib.pyplot as plt
        for target_species in target_species_list:
            fig, (ax1, ax2) = plt.subplots(1, 2, )
            target_species_scores = dfn[f"{pred_prefix}{target_species}"]
            valid_mask = ~np.isnan(target_species_scores)

            thresholds = get_thresholds(target_species_scores, n_sweep_steps)
            ax1.hist(target_species_scores[valid_mask &  dfn[f"gt_{target_species}"]], bins=20, alpha=0.5, label="True")
            ax1.hist(target_species_scores[valid_mask & ~dfn[f"gt_{target_species}"]], bins=20, alpha=0.5, label="False")
            ax1.plot(thresholds, [0] * len(thresholds), 'o', label="Thresholds")
            
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
            plt.savefig(f"figures/scores/{target_species}.png", bbox_inches="tight")

        for target_species in target_species_list:
            # for threshold in np.linspace(dfn[f"{pred_prefix}{target_species}"].min(), dfn[f"{pred_prefix}{target_species}"].max(), 10)[::-1]:
            target_species_scores = dfn[f"{pred_prefix}{target_species}"]
            valid_mask = ~np.isnan(target_species_scores)
            if np.sum(valid_mask) < n_sweep_steps:
                print(f"Species '{target_species}' has to few samples, skipping...")
                continue
            thresholds = [*get_thresholds(target_species_scores, n_sweep_steps), best_threshold[target_species], float("NaN")]
            threshold_types = ["sampled"] * (len(thresholds) - 2) + ["calibrated"] + ["gt"]
            for threshold, threshold_type in zip(thresholds, threshold_types):
                dfn["DateTimeOriginal"] = dfn["datetime"]
                print("Removing", dfn["DateTimeOriginal"].isna().sum(), "missing datetime values")
                dfn = dfn[~dfn["DateTimeOriginal"].isna()]

                if threshold_type != "gt":
                    dfn["predicted"] = np.array([target_species_out if row[f"{pred_prefix}{target_species}"] >= threshold else "other" for _, row in dfn.iterrows()])
                else:
                    dfn["predicted"] = np.array([target_species_out if row[f"gt_{target_species}"] else "other" for _, row in dfn.iterrows()])

                # # TODO: remove. Try to use some GT labels instead of predictions
                # np.random.seed(42)
                # gt_proportion = 0.0
                # use_gt_mask = np.random.rand(len(dfn)) < gt_proportion

                # if gt_proportion > 0:
                #     dfn["predicted"][use_gt_mask] = np.array([target_species_out if row[f"gt_{target_species}"] else "other" for _, row in dfn.iterrows()])[use_gt_mask]

                print((dfn["predicted"] == target_species_out).sum(), f"positive observations with threshold {threshold}")


                tp = ((dfn["predicted"] == target_species_out) &  dfn[f"gt_{target_species}"]).sum()
                fp = ((dfn["predicted"] == target_species_out) & ~dfn[f"gt_{target_species}"]).sum()
                fn = ((dfn["predicted"] != target_species_out) &  dfn[f"gt_{target_species}"]).sum()
                tn = ((dfn["predicted"] != target_species_out) & ~dfn[f"gt_{target_species}"]).sum()

                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                f1 = (2 * precision * recall) / (precision + recall)

                print(f"{target_species} precision {precision:.2f} recall {recall:.2f} f1 {f1:.2f}")

                dfn["observed"] = dfn["predicted"]
                dfn["pix"] = dfn["location"].astype(str) + "_" + dfn["file_name"]


                pyreadr.write_rdata(output_path, dfn, df_name="allfiles")

                model_comparison_df = None
                queue = Queue()
                finished_flag = Event()
                p = Process(target=evaluate_r_occupancy, args=(rscript, queue, finished_flag))
                p.start()
                while p.is_alive() and not finished_flag.is_set():
                    p.join(timeout=1)
                while not queue.empty():
                    model_comparison_df = queue.get()

                if model_comparison_df is not None:
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

                    print("Successfully got model comparison df")
                    print(model_comparison_df)

                    output_table.append(model_comparison_df)
                    pd.concat(output_table).to_csv(output_table_path)
                
                else:
                    print("Failed to get model comparison df")

                # for model in [
                #     "MonthOccuMod",
                #     "WeekOccuMod",
                #     "DayOccuMod",
                # ]:
                #     if success:
                #         try:
                #             aic = robjects.r(f"attr({model}, 'AIC')")[0]
                #         except:
                #             aic = None
                #         try:
                #             occ_est = robjects.r(f"attr({model}, 'opt')")[0][0]
                #         except:
                #             occ_est = None
                #         try:
                #             det_est = robjects.r(f"attr({model}, 'opt')")[0][1]
                #         except:
                #             det_est = None
                #         try:
                #             fpo_est = robjects.r(f"attr({model}, 'opt')")[0][2]
                #         except:
                #             fpo_est = None
                #     else:
                #         aic = None
                #         occ_est = None
                #         det_est = None
                #         fpo_est = None

                #     output_table.append({
                #         "species": target_species,
                #         "threshold": threshold,
                #         "precision": precision,
                #         "recall": recall,
                #         "f1": f1,
                #         "model": model,
                #         "aic": aic,
                #         "occ_est": occ_est,
                #         "det_est": det_est,
                #         "fpo_est": fpo_est,
                #     })

    finally:
        # os.unlink(output_path)
        os.chdir(original_cwd)
        if len(output_table) > 0:
            pd.concat(output_table).to_csv(output_table_path)

if __name__ == "__main__":
    main()