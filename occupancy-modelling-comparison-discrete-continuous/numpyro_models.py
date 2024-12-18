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

matplotlib.use("Agg")  # noqa: E402


def occuFP(site_covs, obs_covs, n_sites, time_periods, obs=None):

    n_site_covs = site_covs.shape[-1]
    n_obs_covs = obs_covs.shape[-1]

    # Priors
    prob_fp = numpyro.sample('prob_fp', dist.Beta(2, 5))
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
            y = numpyro.sample(f'y', dist.Bernoulli(p_det), obs=obs)

    # Estimate proportion of occupied sites
    NOcc = numpyro.deterministic('NOcc', jnp.sum(z))
    PropOcc = numpyro.deterministic('PropOcc', NOcc / n_sites)


def run(model_fn, site_covs, obs_covs, n_sites, time_periods, obs=None, num_samples=1000, num_warmup=500, random_seed=0):
    nuts_kernel = NUTS(model_fn)
    gibbs_kernel = DiscreteHMCGibbs(nuts_kernel, modified=True)
    mcmc = MCMC(gibbs_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(jax.random.PRNGKey(random_seed), site_covs, obs_covs, n_sites, time_periods, obs=obs)
    mcmc.print_summary(exclude_deterministic=False)


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

    print(f"True occupancy: {np.mean(z):.4f}")

    run(occuFP, site_covs, obs_covs, n_sites, time_periods, obs=obs, random_seed=random_seed)