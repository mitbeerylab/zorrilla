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


def occupancy_model(site_covar, n_sites, time_periods, y=None):
    # Priors
    prob_detection = numpyro.sample('prob_detection', dist.Beta(2, 2))
    prob_fp = numpyro.sample('prob_fp', dist.Beta(2, 5))  # TODO: double check parameter order
    beta0 = numpyro.sample('beta0', dist.Uniform(-5, 5))
    beta1 = numpyro.sample('beta1', dist.Uniform(-5, 5))

    with numpyro.plate('site', n_sites):

      # Occupancy process
      psi = numpyro.deterministic('psi', jax.nn.sigmoid(beta0 + beta1 * site_covar))
      z = numpyro.sample('z', dist.Bernoulli(psi))

      with numpyro.plate('time_periods', time_periods):

        # Detection process
        # prob_detection = numpyro.deterministic('prob_detection', jax.nn.sigmoid(alpha0 + alpha1 * det_covar))   # TODO
        p_det = z * prob_detection + prob_fp
        y = numpyro.sample('y', dist.Bernoulli(p_det), obs=y.T)

    # Estimate proportion of occupied sites
    NOcc = numpyro.deterministic('NOcc', jnp.sum(z))
    PropOcc = numpyro.deterministic('PropOcc', NOcc / n_sites)


# Generate occupancy and site-level covariates
n_sites = 100  # number of sites
site_cov = np.random.normal(size=n_sites)
beta1 = 1  # intercept for occupancy logistic regression
beta0 = -1  # slope for occupancy logistic regression
psi_cov = 1 / (1 + np.exp(-(beta0 + beta1 * site_cov)))
z = np.random.binomial(n=1, p=psi_cov, size=n_sites)  # vector of latent occupancy status for each site

# Generate detection data
deployment_days_per_site = 120
prob_detection = 0.3  # probability of detecting species of interest
prob_fp = 0.01  # probability of a false positive for a given time point
session_duration = 7  # 1, 7, or 30
time_periods = round(deployment_days_per_site / session_duration)

# Create matrix of detections
dfa = np.zeros((n_sites, time_periods))

for i in range(n_sites):
    # According to the Royle model in unmarked, false positives are generated only if the site is unoccupied
    # Note this is different than how we think about false positives being a random occurrence per image.
    # For now, this is generating positive/negative per time period, which is different than per image.
    dfa[i, :] = np.random.binomial(n=1, p=(prob_detection * z[i] + prob_fp * (1 - z[i])), size=time_periods)

# Convert counts into observed occupancy
y = (dfa >= 1) * 1

nuts_kernel = NUTS(occupancy_model)
mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=500)
mcmc.run(jax.random.PRNGKey(0), site_cov, n_sites, time_periods, y=y)
mcmc.print_summary()
pass