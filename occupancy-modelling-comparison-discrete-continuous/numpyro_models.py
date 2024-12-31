from typing import Optional
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, DiscreteHMCGibbs, NUTS


def occu(site_covs: np.ndarray, obs_covs: np.ndarray, session_duration: Optional[np.ndarray] = None, false_positives_constant: bool = False, false_positives_unoccupied: bool = False, counting_occurences: bool = False, obs: Optional[np.ndarray] = None):

    # Check input data
    assert obs is None or obs.ndim == 2, "obs must be None or of shape (n_sites, time_periods)"
    assert site_covs.ndim == 2, "site_covs must be of shape (n_sites, n_site_covs)"
    assert obs_covs.ndim == 3, "obs_covs must be of shape (n_sites, time_periods, n_obs_covs)"
    assert session_duration is None or session_duration.ndim == 2, "session_duration must be None or of shape (n_sites, time_periods)"
    assert (obs[np.isfinite(obs)] >= 0).all(), "observations must be non-negative"
    assert counting_occurences or (obs[np.isfinite(obs)] <= 1).all(), "observations must be binary when counting_occurences is False"
    assert not (false_positives_constant and false_positives_unoccupied), "false_positives_constant and false_positives_unoccupied cannot both be True"

    n_sites = site_covs.shape[0]
    time_periods = obs_covs.shape[1]
    n_site_covs = site_covs.shape[1]
    n_obs_covs = obs_covs.shape[2]

    assert n_sites == site_covs.shape[0] == obs_covs.shape[0], "site_covs and obs_covs must have the same number of sites"
    assert time_periods == obs_covs.shape[1], "obs_covs must have the same number of time periods as obs"
    if obs is not None:
        assert n_sites == obs.shape[0], "obs must have n_sites rows"
        assert time_periods == obs.shape[1], "obs must have time_periods columns"
    assert not counting_occurences or session_duration is not None, "session_duration must be provided when counting_occurences is True"
    if session_duration is not None:
        assert n_sites == session_duration.shape[0], "session_duration must have n_sites rows"
        assert time_periods == session_duration.shape[1], "session_duration must have time_periods columns"

    # Mask observations where covariates are missing
    obs_mask = jnp.isnan(obs_covs).any(axis=-1) | jnp.tile(jnp.isnan(site_covs).any(axis=-1)[:, None], (1, time_periods))
    obs = jnp.where(obs_mask, jnp.nan, obs)
    obs_covs = jnp.nan_to_num(obs_covs)
    site_covs = jnp.nan_to_num(site_covs)

    # Priors
    if not counting_occurences:
        # Model false positive rate for both occupied and unoccupied sites
        prob_fp_constant = numpyro.sample('prob_fp_constant', dist.Beta(2, 5)) if false_positives_constant else 0
        
        # Model false positive rate only for occupied sites
        prob_fp_unoccupied = numpyro.sample('prob_fp_unoccupied', dist.Beta(2, 5)) if false_positives_unoccupied else 0
    else:
        # get maximum observation rate
        obs_max = jnp.where(jnp.isfinite(obs), obs, 0).max()

        # Model false positive rate for both occupied and unoccupied sites
        rate_fp_constant = numpyro.sample('rate_fp_constant', dist.Uniform(0, obs_max)) if false_positives_constant else 0
        
        # Model false positive rate only for occupied sites
        rate_fp_unoccupied = numpyro.sample('rate_fp_unoccupied', dist.Uniform(0, obs_max)) if false_positives_unoccupied else 0
    
    # Occupancy and detection covariates
    beta = jnp.array([numpyro.sample(f'beta_{i}', dist.Normal()) for i in range(n_site_covs + 1)])
    alpha = jnp.array([numpyro.sample(f'alpha_{i}', dist.Normal()) for i in range(n_obs_covs + 1)])

    with numpyro.plate('site', n_sites, dim=-2):

        # Occupancy process
        psi = numpyro.deterministic('psi', jax.nn.sigmoid(beta[0] + jnp.sum(jnp.array([beta[i + 1] * site_covs[..., i] for i in range(n_site_covs)]), axis=0)))
        z = numpyro.sample('z', dist.Bernoulli(psi[:, None]))

        with numpyro.plate('time_periods', time_periods, dim=-1):

            if not counting_occurences:

                # Detection process
                prob_detection = numpyro.deterministic(f'prob_detection', jax.nn.sigmoid(alpha[0] + jnp.sum(jnp.array([alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)]), axis=0)))
                p_det = z * prob_detection + (1 - z) * prob_fp_unoccupied + prob_fp_constant
                p_det = jnp.clip(p_det, min=0, max=1)

                with numpyro.handlers.mask(mask=jnp.isfinite(obs)):
                    y = numpyro.sample(f'y', dist.Bernoulli(p_det), obs=jnp.nan_to_num(obs))
            else:

                # Detection process
                rate_detection = numpyro.deterministic(f'rate_detection', jax.nn.relu(alpha[0] + jnp.sum(jnp.array([alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)]), axis=0)))
                l_det = z * rate_detection + (1 - z) * rate_fp_unoccupied + rate_fp_constant
                l_det = jnp.clip(l_det, min=0)

                with numpyro.handlers.mask(mask=jnp.isfinite(obs)):
                    y = numpyro.sample(f'y', dist.Poisson(l_det), obs=jnp.nan_to_num(obs))


    # Estimate proportion of occupied sites
    NOcc = numpyro.deterministic('NOcc', jnp.sum(z))
    PropOcc = numpyro.deterministic('PropOcc', NOcc / n_sites)


def run(model_fn, site_covs, obs_covs, obs, session_duration=None, num_samples=1000, num_warmup=500, random_seed=0, **kwargs):
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
    if isinstance(session_duration, pd.DataFrame):
        session_duration = session_duration.sort_index().to_numpy()
    if isinstance(obs, pd.DataFrame):
        original_index = obs.index
        obs = obs.sort_index().to_numpy()

    if site_covs_names is None:
        site_covs_names = [str(i) for i in range(site_covs.shape[1])]
    if obs_covs_names is None:
        obs_covs_names = [str(i) for i in range(obs_covs.shape[2])]
    
    mcmc.run(jax.random.PRNGKey(random_seed), site_covs, obs_covs, session_duration=session_duration, obs=obs, **kwargs)
    samples = mcmc.get_samples()
    
    results_df = pd.DataFrame(dict(
        Predicted=samples["psi"].mean(axis=0),
        DetectionProb=samples["prob_detection"].mean(axis=(0, -1)) if "prob_detection" in samples else None,
        DetectionRate=samples["rate_detection"].mean(axis=(0, -1)) if "rate_detection" in samples else None,
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


def test_occu_fp():
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
    
    prob_fp = 0.01  # probability of a false positive for a given time point
    session_duration = 7  # 1, 7, or 30
    time_periods = round(deployment_days_per_site / session_duration)

    # Create matrix of detection covariates
    n_obs_covs = 3
    obs_covs = rng.normal(size=(n_sites, time_periods, n_obs_covs))
    alpha = [-0.5, 0.5, -0.5, 0]  # intercept and slopes for detection logistic regression
    obs_reg = 1 / (1 + np.exp(-(alpha[0] + np.sum([alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)], axis=0))))

    # Create matrix of detections
    dfa = np.zeros((n_sites, time_periods))

    for i in range(n_sites):
        # According to the Royle model in unmarked, false positives are generated only if the site is unoccupied
        # Note this is different than how we think about false positives being a random occurrence per image.
        # For now, this is generating positive/negative per time period, which is different than per image.
        dfa[i, :] = rng.binomial(n=1, p=(obs_reg[i, :] * z[i] + prob_fp * (1 - z[i])), size=time_periods)

    # Convert counts into observed occupancy
    obs = (dfa >= 1) * 1.

    # Simulate missing data:
    obs[np.random.choice([True, False], size=obs.shape, p=[0.2, 0.8])] = np.nan
    obs_covs[np.random.choice([True, False], size=obs_covs.shape, p=[0.05, 0.95])] = np.nan
    site_covs[np.random.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])] = np.nan

    print(f"True occupancy: {np.mean(z):.4f}")
    print(f"Proportion of timesteps with observation: {np.mean(obs[np.isfinite(obs)]):.4f}")

    # TODO: really, false_positives_constant should be False and false_positives_unoccupied should be True according to
    # the simulation process, but the model for some reason fails to converge in that case
    results_df, mcmc = run(occu, site_covs, obs_covs, false_positives_constant=True, obs=obs, random_seed=random_seed)

    mcmc.print_summary()
    print(results_df)


def test_occu_cop():
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
    
    rate_fp = 0.01  # probability of a false positive for a given time point
    session_duration = 7  # 1, 7, or 30
    time_periods = round(deployment_days_per_site / session_duration)

    # Create matrix of detection covariates
    n_obs_covs = 3
    obs_covs = rng.normal(size=(n_sites, time_periods, n_obs_covs))
    alpha = [0.5, 0.1, -0.1, 0]  # intercept and slopes for detection logistic regression
    obs_reg = np.clip(alpha[0] + np.sum([alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)], axis=0), a_min=0, a_max=np.inf)

    # Create matrix of detections
    dfa = np.zeros((n_sites, time_periods))

    for i in range(n_sites):
        # According to the Royle model in unmarked, false positives are generated only if the site is unoccupied
        # Note this is different than how we think about false positives being a random occurrence per image.
        # For now, this is generating positive/negative per time period, which is different than per image.
        dfa[i, :] = rng.poisson(lam=(obs_reg[i, :] * z[i] + rate_fp * (1 - z[i])), size=time_periods)

    obs = dfa

    # Simulate missing data
    obs[np.random.choice([True, False], size=obs.shape, p=[0.2, 0.8])] = np.nan
    obs_covs[np.random.choice([True, False], size=obs_covs.shape, p=[0.05, 0.95])] = np.nan
    site_covs[np.random.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])] = np.nan

    print(f"True occupancy: {np.mean(z):.4f}")
    print(f"Mean rate: {np.mean(dfa[np.isfinite(dfa)]):.4f}")

    session_duration_arr = np.full((n_sites, time_periods), session_duration)

    # TODO: really, false_positives_constant should be False and false_positives_unoccupied should be True according to
    # the simulation process, but the model for some reason fails to converge in that case
    results_df, mcmc = run(occu, site_covs, obs_covs, session_duration=session_duration_arr, false_positives_constant=True, counting_occurences=True, obs=obs, random_seed=random_seed)

    mcmc.print_summary()
    print(results_df)


if __name__ == "__main__":
    test_occu_fp()
    test_occu_cop()