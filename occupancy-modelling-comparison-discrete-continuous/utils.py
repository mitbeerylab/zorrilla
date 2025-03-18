import numpy as np
import pandas as pd


# let's define a function to sample images uniformly across the target species logits
def uniform_sample_by_column(df, column, n_samples, n_bins=10, random_state=42):

    # filter to rows that have a value in the column
    valid_df = df[df[column].notna()]

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