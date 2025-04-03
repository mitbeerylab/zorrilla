import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import signal
from contextlib import contextmanager
from numpyro.diagnostics import hpdi


# let's define a function to sample images uniformly across the target species logits
def uniform_sample_by_column(df, column, n_samples, n_bins=10, random_state=42):

    # filter to rows that have a value in the column
    valid_df = df[df[column].notna() & np.isfinite(df[column])]

    if valid_df.empty:
        return pd.DataFrame()

    # create bins across the range of values
    bins = np.linspace(valid_df[column].min(), valid_df[column].max(), n_bins+1)
    valid_df['bin'] = pd.cut(valid_df[column], bins=bins)

    # sample equally from each bin
    samples_per_bin = max(1, int(n_samples / n_bins))

    sampled_rows = []
    for _, bin_group in valid_df.groupby('bin'):
        if len(bin_group) > 0:
            n_to_sample = min(samples_per_bin, len(bin_group))
            sampled_rows.append(bin_group.sample(n_to_sample, random_state=random_state))

    # combine and trim to requested sample size
    result = pd.concat(sampled_rows).drop('bin', axis=1)
    if len(result) > n_samples:
        result = result.sample(n_samples, random_state=random_state)

    return result


def calibrate_threshold(df_unlabeled, df_labeled, score_column, gt_column, k=5):
    # get a sorted list of all possible thresholds
    possible_thresholds = sorted(df_labeled[score_column].unique())

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    selected_thresholds = []
    selected_false_positive_rates = []
    mu0s = []
    sigma0s = []
    mu1s = []
    sigma1s = []
    for train_index, test_index in kf.split(np.arange(len(df_labeled))):

        df_train = df_labeled.iloc[train_index]
        df_test = df_labeled.iloc[test_index]
                
        false_examples = df_test[~df_test[gt_column]][score_column]
        true_examples = df_test[df_test[gt_column]][score_column]
        
        mu0s.append(false_examples.mean())
        sigma0s.append(false_examples.std())
        mu1s.append(true_examples.mean())
        sigma1s.append(true_examples.std())

        # initialize variables to keep track of the best threshold and F1 score
        best_f1 = float("-inf")
        best_thresholds = []
        false_positive_rates = []

        # loop through all possible thresholds
        for candidate_threshold in possible_thresholds:

            # see which labeled images would be predicted as positives and negatives given the candidate threshold
            predicted_positives = df_train[score_column] >= candidate_threshold

            # see which labeled images are actually positives and negatives
            gt_positives = df_train[gt_column]
            gt_negatives = ~gt_positives

            # calculate precision and recall
            precision = (predicted_positives & gt_positives).sum() / (predicted_positives.sum() if predicted_positives.sum() > 0 else 1)
            recall = (predicted_positives & gt_positives).sum() / (gt_positives.sum() if gt_positives.sum() > 0 else 1)

            # calculate false positive rate on test set
            predicted_positives_test = df_test[score_column] >= candidate_threshold
            gt_positives_test = df_test[gt_column]
            gt_negatives_test = ~gt_positives_test
            false_positive_rates.append((gt_negatives_test & predicted_positives_test).sum() / (gt_negatives_test.sum() if gt_negatives_test.sum() > 0 else 1))

            # calculate F1 score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # update the best threshold and F1 score if the current threshold is better
            if f1 == best_f1:
                best_thresholds.append(candidate_threshold)
            elif f1 > best_f1:
                best_f1 = f1
                best_thresholds = [candidate_threshold]

        # average over the best thresholds to get a single threshold
        best_threshold = np.mean(best_thresholds)
        selected_thresholds.append(best_threshold)
        selected_false_positive_rates.append(false_positive_rates)

    false_positive_rates = np.array(selected_false_positive_rates).mean(axis=0)
    false_positive_rates_std = np.array(selected_false_positive_rates).std(axis=0)

    mu0_mean = np.mean(mu0s)
    mu0_std = np.std(mu0s)
    mu1_mean = np.mean(mu1s)
    mu1_std = np.std(mu1s)
    sigma0_mean = np.mean(sigma0s)
    sigma0_std = np.std(sigma0s)
    sigma1_mean = np.mean(sigma1s)
    sigma1_std = np.std(sigma1s)


    return best_threshold, possible_thresholds, false_positive_rates, false_positive_rates_std, mu0_mean, mu0_std, mu1_mean, mu1_std, sigma0_mean, sigma0_std, sigma1_mean, sigma1_std




class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


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