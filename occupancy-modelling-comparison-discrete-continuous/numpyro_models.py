import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist
from numpyro.examples.datasets import LYNXHARE, load_dataset
from numpyro.infer import MCMC, DiscreteHMCGibbs, NUTS, Predictive
from numpyro.diagnostics import summary, hpdi, effective_sample_size, split_gelman_rubin

matplotlib.use("Agg")  # noqa: E402


def occu(site_covs, obs_covs, false_positives: bool = False, obs=None):

    assert obs is None or len(obs.shape) == 2, "obs must be None or of shape (n_sites, time_periods)"
    assert len(site_covs.shape) == 2, "site_covs must be of shape (n_sites, n_site_covs)"
    assert len(obs_covs.shape) == 3, "obs_covs must be of shape (n_sites, time_periods, n_obs_covs)"

    n_sites = site_covs.shape[0]
    time_periods = obs_covs.shape[1]
    n_site_covs = site_covs.shape[1]
    n_obs_covs = obs_covs.shape[2]

    assert n_sites == site_covs.shape[0] == obs_covs.shape[0], "site_covs and obs_covs must have the same number of sites"
    assert time_periods == obs_covs.shape[1], "obs_covs must have the same number of time periods as obs"
    if obs is not None:
        assert n_sites == obs.shape[0], "obs must have n_sites rows"
        assert time_periods == obs.shape[1], "obs must have time_periods columns"

    # Priors
    if false_positives:
        prob_fp = numpyro.sample('prob_fp', dist.Beta(2, 5))
    else:
        prob_fp = 0
    beta = jnp.array([numpyro.sample(f'beta_{i}', dist.Normal(0, 2)) for i in range(n_site_covs + 1)])
    alpha = jnp.array([numpyro.sample(f'alpha_{i}', dist.Normal(0, 2)) for i in range(n_obs_covs + 1)])

    with numpyro.plate('site', n_sites, dim=-2):

        # Occupancy process
        psi = numpyro.deterministic('psi', jax.nn.sigmoid(beta[0] + jnp.sum(jnp.array([beta[i + 1] * site_covs[..., i] for i in range(n_site_covs)]), axis=0)))
        z = numpyro.sample('z', dist.Bernoulli(psi[:, None]))

        with numpyro.plate('time_periods', time_periods, dim=-1):

            # Detection process
            prob_detection = numpyro.deterministic(f'prob_detection', jax.nn.sigmoid(alpha[0] + jnp.sum(jnp.array([alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)]), axis=0)))
            p_det = z * prob_detection + prob_fp
            y = numpyro.sample(f'y', dist.Bernoulli(jnp.nan_to_num(p_det)).mask(jnp.isfinite(p_det)), obs=jnp.nan_to_num(obs), obs_mask=jnp.isfinite(obs) & jnp.isfinite(p_det))

    # Estimate proportion of occupied sites
    NOcc = numpyro.deterministic('NOcc', jnp.sum(z))
    PropOcc = numpyro.deterministic('PropOcc', NOcc / n_sites)


def run(model_fn, site_covs, obs_covs, obs, num_samples=1000, num_warmup=500, random_seed=0, prob=0.9, **kwargs):
    nuts_kernel = NUTS(model_fn)
    gibbs_kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)
    mcmc = MCMC(gibbs_kernel, num_samples=num_samples, num_warmup=num_warmup)

    # convert dataframes to numpy arrays
    original_index = None
    site_covs_names = None
    obs_covs_names = None
    if isinstance(site_covs, pd.DataFrame):
        site_covs_names = site_covs.columns
        site_covs = site_covs.sort_index().to_numpy()
    if isinstance(obs_covs, pd.DataFrame):
        if not isinstance(obs_covs.columns, pd.MultiIndex):
            obs_covs = obs_covs.sort_index().to_numpy()
        else:
            assert len(obs_covs.columns.levels) == 2, "obs_covs with MultiIndex columns must have columns of exactly two levels"
            obs_covs_names = obs_covs.columns.levels[0]
            obs_covs = obs_covs.sort_index().to_numpy().reshape(obs_covs.shape[0], len(obs_covs.columns.levels[0]), len(obs_covs.columns.levels[1])).transpose(0, 2, 1)
    if isinstance(obs, pd.DataFrame):
        original_index = obs.index
        obs = obs.sort_index().to_numpy()

    if site_covs_names is None:
        site_covs_names = [str(i) for i in range(site_covs.shape[1])]
    if obs_covs_names is None:
        obs_covs_names = [str(i) for i in range(obs_covs.shape[2])]

    # TODO: remove
    obs = jnp.nan_to_num(obs)
    obs_covs = jnp.nan_to_num(obs_covs)
    site_covs = jnp.nan_to_num(site_covs)
    
    mcmc.run(jax.random.PRNGKey(random_seed), site_covs, obs_covs, obs=obs, **kwargs)
    # mcmc.print_summary()
    samples = mcmc.get_samples()
    
    results_df = pd.DataFrame(dict(
        Predicted=samples["psi"].mean(axis=0),
        DetectionProb=samples["prob_detection"].mean(axis=(0, -1)),
        FPProb=samples["prob_fp"].mean() if "prob_fp" in samples else None,

        cov_state_intercept = samples["beta_0"].mean(),
        cov_state_intercept_se = samples["beta_0"].std(),
        cov_det_intercept = samples["alpha_0"].mean(),
        cov_det_intercept_se = samples["alpha_0"].std(),
        **{f'cov_state_{site_covs_names[i]}': samples[f'beta_{i + 1}'].mean() for i in range(site_covs.shape[1])},
        **{f'cov_state_{site_covs_names[i]}_se': samples[f'beta_{i + 1}'].std() for i in range(site_covs.shape[1])},
        **{f'cov_det_{obs_covs_names[i]}': samples[f'alpha_{i + 1}'].mean() for i in range(obs_covs.shape[2])},
        **{f'cov_det_{obs_covs_names[i]}_se': samples[f'alpha_{i + 1}'].std() for i in range(obs_covs.shape[2])},
    ), index=original_index)

    return results_df

if __name__ == "__main__":

    # Initialize random number generator
    random_seed = 0
    rng = np.random.default_rng(random_seed)

    # Generate occupancy and site-level covariates
    n_sites = 100  # number of sites
    n_site_covs = 4
    site_covs = rng.normal(size=(n_sites, n_site_covs))
    beta = [-2, -0.5, 0.5, 0, 0]  # intercept and slopes for occupancy logistic regression
    psi_cov = 1 / (1 + np.exp(-(beta[0] + np.sum([beta[i + 1] * site_covs[..., i] for i in range(n_site_covs)]))))
    z = rng.binomial(n=1, p=psi_cov, size=n_sites)  # vector of latent occupancy status for each site

    # Generate detection data
    deployment_days_per_site = 365
    
    prob_fp = 0.01  # probability of a false positive for a given time point
    session_duration = 7  # 1, 7, or 30
    time_periods = round(deployment_days_per_site / session_duration)

    # Create matrix of detection covariates
    n_obs_covs = 3
    obs_covs = rng.normal(size=(n_sites, time_periods, n_obs_covs))
    alpha = [-2, 0.1, -0.1, 0]  # intercept and slopes for detection logistic regression
    obs_reg = 1 / (1 + np.exp(-(alpha[0] + np.sum([alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)], axis=0))))

    # Create matrix of detections
    dfa = np.zeros((n_sites, time_periods))

    for i in range(n_sites):
        # According to the Royle model in unmarked, false positives are generated only if the site is unoccupied
        # Note this is different than how we think about false positives being a random occurrence per image.
        # For now, this is generating positive/negative per time period, which is different than per image.
        dfa[i, :] = rng.binomial(n=1, p=(obs_reg[i, :] * z[i] + prob_fp * (1 - z[i])), size=time_periods)

    # Convert counts into observed occupancy
    obs = (dfa >= 1) * 1

    # # convert numpy arrays into dataframes
    # obs = pd.DataFrame(obs, columns=[f'time_{i}' for i in range(time_periods)])
    # site_covs = pd.DataFrame(site_covs, columns=[f'cov_{i}' for i in range(n_site_covs)])
    # obs_covs = pd.DataFrame(obs_covs.reshape(obs_covs.shape[0], -1), columns=[[f'time_{i}', f'cov_{j}' for j in range(n_obs_covs)] for i in time_periods])

    print(f"True occupancy: {np.mean(z):.4f}")

    run(occu, site_covs, obs_covs, false_positives=True, obs=obs, random_seed=random_seed)