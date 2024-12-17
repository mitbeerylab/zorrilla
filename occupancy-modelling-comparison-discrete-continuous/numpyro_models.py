import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist
from numpyro.examples.datasets import LYNXHARE, load_dataset
from numpyro.infer import MCMC, NUTS, Predictive

matplotlib.use("Agg")  # noqa: E402


def occuFP(site_covs, obs_covs, n_sites, time_periods, obs=None):

    # Priors
    prob_fp = numpyro.sample('prob_fp', dist.Beta(    2, 5))
    beta0   = numpyro.sample('beta0',   dist.Uniform(-5, 5))
    beta1   = numpyro.sample('beta1',   dist.Uniform(-5, 5))
    alpha0  = numpyro.sample('alpha0',  dist.Normal(  0, 2))
    alpha1  = numpyro.sample('alpha1',  dist.Normal(  0, 2))

    with numpyro.plate('site', n_sites, dim=-2):

        # Occupancy process
        psi = numpyro.deterministic('psi', jax.nn.sigmoid(beta0 + beta1 * site_covs))
        z = numpyro.sample('z', dist.Bernoulli(psi[:, None]), infer={'enumerate': 'parallel'})

        with numpyro.plate('time_periods', time_periods, dim=-1):

            # Detection process
            prob_detection = numpyro.deterministic(f'prob_detection', jax.nn.sigmoid(alpha0 + alpha1 * obs_covs))
            p_det = z * prob_detection + prob_fp
            y = numpyro.sample(f'y', dist.Bernoulli(p_det), obs=obs)

    # Estimate proportion of occupied sites
    NOcc = numpyro.deterministic('NOcc', jnp.sum(z))
    PropOcc = numpyro.deterministic('PropOcc', NOcc / n_sites)


def run(model_fn, site_covs, obs_covs, n_sites, time_periods, obs=None, num_samples=1000, num_warmup=500, random_seed=0):
    nuts_kernel = NUTS(model_fn)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(jax.random.PRNGKey(random_seed), site_covs, obs_covs, n_sites, time_periods, obs=obs)
    mcmc.print_summary()


if __name__ == "__main__":
    # Generate occupancy and site-level covariates
    n_sites = 100  # number of sites
    site_covs = np.random.normal(size=n_sites)
    beta1 = 1  # intercept for occupancy logistic regression
    beta0 = -1  # slope for occupancy logistic regression
    psi_cov = 1 / (1 + np.exp(-(beta0 + beta1 * site_covs)))
    z = np.random.binomial(n=1, p=psi_cov, size=n_sites)  # vector of latent occupancy status for each site

    # Generate detection data
    deployment_days_per_site = 120
    
    prob_fp = 0.01  # probability of a false positive for a given time point
    session_duration = 7  # 1, 7, or 30
    time_periods = round(deployment_days_per_site / session_duration)

    # Create matrix of detection covariates
    obs_covs = np.random.randn(n_sites, time_periods)
    alpha0 = -2  # intercept for detection logistic regression
    alpha1 = 0.1  # slope for detection logistic regression
    obs_reg = 1 / (1 + np.exp(-(alpha0 + alpha1 * obs_covs)))

    # Create matrix of detections
    dfa = np.zeros((n_sites, time_periods))

    for i in range(n_sites):
        # According to the Royle model in unmarked, false positives are generated only if the site is unoccupied
        # Note this is different than how we think about false positives being a random occurrence per image.
        # For now, this is generating positive/negative per time period, which is different than per image.
        dfa[i, :] = np.random.binomial(n=1, p=(obs_reg[i, :] * z[i] + prob_fp * (1 - z[i])), size=time_periods)

    # Convert counts into observed occupancy
    obs = (dfa >= 1) * 1

    run(occuFP, site_covs, obs_covs, n_sites, time_periods, obs=obs)