from typing import Optional
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def occu_cs(site_covs: np.ndarray, obs_covs: np.ndarray, obs: Optional[np.ndarray] = None):

    # Check input data
    assert obs is None or obs.ndim == 2, "obs must be None or of shape (n_sites, time_periods)"
    assert site_covs.ndim == 2, "site_covs must be of shape (n_sites, n_site_covs)"
    assert obs_covs.ndim == 3, "obs_covs must be of shape (n_sites, time_periods, n_obs_covs)"

    n_sites = site_covs.shape[0]
    time_periods = obs_covs.shape[1]
    n_site_covs = site_covs.shape[1]
    n_obs_covs = obs_covs.shape[2]

    assert n_sites == site_covs.shape[0] == obs_covs.shape[0], "site_covs and obs_covs must have the same number of sites"
    assert time_periods == obs_covs.shape[1], "obs_covs must have the same number of time periods as obs"
    if obs is not None:
        assert n_sites == obs.shape[0], "obs must have n_sites rows"
        assert time_periods == obs.shape[1], "obs must have time_periods columns"

    # Mask observations where covariates are missing
    obs_mask = jnp.isnan(obs_covs).any(axis=-1) | jnp.tile(jnp.isnan(site_covs).any(axis=-1)[:, None], (1, time_periods))
    obs = jnp.where(obs_mask, jnp.nan, obs)
    obs_covs = jnp.nan_to_num(obs_covs)
    site_covs = jnp.nan_to_num(site_covs)
    
    # Occupancy and detection covariates
    beta = jnp.array([numpyro.sample(f'beta_{i}', dist.Normal()) for i in range(n_site_covs + 1)])
    alpha = jnp.array([numpyro.sample(f'alpha_{i}', dist.Normal()) for i in range(n_obs_covs + 1)])

    # Continuous score parameters
    mu0 = numpyro.sample('mu0', dist.Uniform(-100, 100))
    mu1 = numpyro.sample('mu1', dist.Uniform(-100, 100))
    numpyro.factor("mu_constraint", -jax.nn.softplus(mu0 - mu1))  # constrain such that mu0 < mu1
    sigma0 = numpyro.sample('sigma0', dist.HalfNormal())
    sigma1 = numpyro.sample('sigma1', dist.HalfNormal())

    # Transpose in order to fit NumPyro's plate structure
    site_covs = site_covs.transpose((1, 0))
    obs_covs = obs_covs.transpose((2, 1, 0))
    obs = obs.transpose((1, 0))

    with numpyro.plate('site', n_sites, dim=-1):

        # Occupancy process
        psi = numpyro.deterministic('psi', jax.nn.sigmoid(jnp.tile(beta[0], (n_sites,)) + jnp.sum(jnp.array([beta[i + 1] * site_covs[i, ...] for i in range(n_site_covs)]), axis=0)))
        z = numpyro.sample('z', dist.Bernoulli(probs=psi), infer={'enumerate': 'parallel'})

        with numpyro.plate('time_periods', time_periods, dim=-2):

            # Detection process
            f = numpyro.sample('f', dist.Bernoulli(z * jax.nn.sigmoid(jnp.tile(alpha[0], (time_periods, n_sites)) + jnp.sum(jnp.array([alpha[i + 1] * obs_covs[i, ...] for i in range(n_obs_covs)]), axis=0))), infer={'enumerate': 'parallel'})

            with numpyro.handlers.mask(mask=jnp.isfinite(obs)):
                s = numpyro.sample('s', dist.Normal(jnp.where(f == 0, mu0, mu1), jnp.where(f == 0, sigma0, sigma1)), obs=jnp.nan_to_num(obs))


    # Estimate proportion of occupied sites
    NOcc = numpyro.deterministic('NOcc', jnp.sum(z))
    PropOcc = numpyro.deterministic('PropOcc', NOcc / n_sites)


def run(model_fn, site_covs, obs_covs, obs, session_duration=None, num_samples=1000, num_warmup=1000, random_seed=0, num_chains=1, **kwargs):
    nuts_kernel = NUTS(model_fn)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains, chain_method='parallel' if num_chains <= jax.local_device_count() else 'sequential')

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
    if isinstance(session_duration, pd.DataFrame):
        session_duration = session_duration.sort_index().to_numpy()
    if isinstance(obs, pd.DataFrame):
        original_index = obs.index
        obs = obs.sort_index().to_numpy()

    if site_covs_names is None:
        site_covs_names = [str(i) for i in range(site_covs.shape[1])]
    if obs_covs_names is None:
        obs_covs_names = [str(i) for i in range(obs_covs.shape[2])]
    
    mcmc.run(jax.random.PRNGKey(random_seed), site_covs, obs_covs, obs=obs, **kwargs)
    samples = mcmc.get_samples()
    
    results_df = pd.DataFrame(dict(
        Predicted=samples["psi"].mean(axis=0),
        DetectionProb=samples["prob_detection"].mean(axis=(0, 1)) if "prob_detection" in samples else None,
        DetectionRate=samples["rate_detection"].mean(axis=(0, 1)) if "rate_detection" in samples else None,
        FPProb=samples["prob_fp_constant"].mean() if "prob_fp_constant" in samples else None,
        FPUnoccupiedProb=samples["prob_fp_unoccupied"].mean() if "prob_fp_unoccupied" in samples else None,

        cov_state_intercept = samples["beta_0"].mean(),
        cov_state_intercept_se = samples["beta_0"].std(),
        cov_det_intercept = samples["alpha_0"].mean(),
        cov_det_intercept_se = samples["alpha_0"].std(),
        **{f'cov_state_{site_covs_names[i]}': samples[f'beta_{i + 1}'].mean() for i in range(site_covs.shape[1])},
        **{f'cov_state_{site_covs_names[i]}_se': samples[f'beta_{i + 1}'].std() for i in range(site_covs.shape[1])},
        **{f'cov_det_{obs_covs_names[i]}': samples[f'alpha_{i + 1}'].mean() for i in range(obs_covs.shape[2])},
        **{f'cov_det_{obs_covs_names[i]}_se': samples[f'alpha_{i + 1}'].std() for i in range(obs_covs.shape[2])},
    ), index=original_index)

    return results_df, mcmc


def test_occu_cs():
    # Initialize random number generator
    random_seed = 0
    rng = np.random.default_rng(random_seed)

    # Generate occupancy and site-level covariates
    n_sites = 100  # number of sites
    n_site_covs = 4
    site_covs = rng.normal(size=(n_sites, n_site_covs))
    beta = [1, -0.05, 0.02, 0.01, -0.02]  # intercept and slopes for occupancy logistic regression
    psi_cov = 1 / (1 + np.exp(-(beta[0] + np.sum([beta[i + 1] * site_covs[..., i] for i in range(n_site_covs)]))))
    z = rng.binomial(n=1, p=psi_cov, size=n_sites)  # vector of latent occupancy status for each site

    # Generate detection data
    deployment_days_per_site = 365
    detections_per_day = 10
    
    time_periods = round(deployment_days_per_site * detections_per_day)

    # Create matrix of detection covariates
    n_obs_covs = 3
    obs_covs = rng.normal(size=(n_sites, time_periods, n_obs_covs))
    alpha = [-0.5, 0.5, -0.5, 0]  # intercept and slopes for detection logistic regression
    obs_reg = 1 / (1 + np.exp(-(alpha[0] + np.sum([alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)], axis=0))))

    # Generate score distributions
    mu0 = 0
    sigma0 = 10
    mu1 = 10
    sigma1 = 5

    # Create matrix of detections
    dfa = np.zeros((n_sites, time_periods))

    for i in range(n_sites):
        # According to the Royle model in unmarked, false positives are generated only if the site is unoccupied
        # Note this is different than how we think about false positives being a random occurrence per image.
        # For now, this is generating positive/negative per time period, which is different than per image.
        f = rng.binomial(n=1, p=(obs_reg[i, :] * z[i]), size=time_periods)
        for t in range(time_periods):
            dfa[i, t] = rng.normal(mu0 if f[t] == 0 else mu1, sigma0 if f[t] == 0 else sigma1)

    obs = dfa

    # Simulate missing data:
    obs[np.random.choice([True, False], size=obs.shape, p=[0.2, 0.8])] = np.nan
    obs_covs[np.random.choice([True, False], size=obs_covs.shape, p=[0.05, 0.95])] = np.nan
    site_covs[np.random.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])] = np.nan

    print(f"True occupancy: {np.mean(z):.4f}")
    results_df, mcmc = run(occu_cs, site_covs, obs_covs, obs=obs, random_seed=random_seed)

    mcmc.print_summary()
    print(results_df)


if __name__ == "__main__":
    test_occu_cs()