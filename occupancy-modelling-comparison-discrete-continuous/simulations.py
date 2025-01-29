import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import rpy2
import rpy2.robjects as robjects
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()

from numpyro_models import occu, run


py2r = ro.conversion.get_conversion().py2rpy
r2py = ro.conversion.get_conversion().rpy2py


def simulate_occu_fp(
        n_site_covs=1,
        n_obs_covs=1,
        n_sites=100,  # number of sites
        deployment_days_per_site=365,  # number of days each site is monitored
        session_duration=7,  # 1, 7, or 30 days
        prob_fp=0.01,  # probability of a false positive for a given time point
        random_seed=0,
):
    
    # Initialize random number generator
    rng = np.random.default_rng(random_seed)

    # Make sure true occupancy is not too close to 0 or 1
    # TODO: find a better way
    z = None
    while z is None or z.mean() < 0.25 or z.mean() > 0.75 or np.mean(obs[np.isfinite(obs)]) < 0.1 or np.mean(obs[np.isfinite(obs)]) > 0.5:

        # Generate intercept and slopes
        beta = rng.normal(size=n_site_covs + 1)  # intercept and slopes for occupancy logistic regression
        alpha = rng.normal(size=n_obs_covs + 1)  # intercept and slopes for detection logistic regression

        # Generate occupancy and site-level covariates
        site_covs = rng.normal(size=(n_sites, n_site_covs))
        psi_cov = 1 / (1 + np.exp(-(beta[0].repeat(n_sites) + np.sum([beta[i + 1] * site_covs[..., i] for i in range(n_site_covs)]))))
        z = rng.binomial(n=1, p=psi_cov, size=n_sites)  # vector of latent occupancy status for each site

        # Generate detection data
        time_periods = round(deployment_days_per_site / session_duration)

        # Create matrix of detection covariates
        obs_covs = rng.normal(size=(n_sites, time_periods, n_obs_covs))
        obs_reg = 1 / (1 + np.exp(-(alpha[0].repeat(n_sites)[:, None] + np.sum([alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)], axis=0))))

        # Create matrix of detections
        dfa = np.zeros((n_sites, time_periods))

        for i in range(n_sites):
            # According to the Royle model in unmarked, false positives are generated only if the site is unoccupied
            # Note this is different than how we think about false positives being a random occurrence per image.
            # For now, this is generating positive/negative per time period, which is different than per image.
            dfa[i, :] = rng.binomial(n=1, p=(obs_reg[i, :] * z[i] + prob_fp * (1 - z[i])), size=time_periods)

        # Convert counts into observed occupancy
        obs = (dfa >= 1) * 1.

        # # Simulate missing data:
        # obs[np.random.choice([True, False], size=obs.shape, p=[0.2, 0.8])] = np.nan
        # obs_covs[np.random.choice([True, False], size=obs_covs.shape, p=[0.05, 0.95])] = np.nan
        # site_covs[np.random.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])] = np.nan

    print(f"True occupancy: {np.mean(z):.4f}")
    print(f"Proportion of timesteps with observation: {np.mean(obs[np.isfinite(obs)]):.4f}")

    return obs, site_covs, obs_covs, beta, alpha, z


def fit_r(obs, site_covs, obs_covs, random_seed=0):

    n_site_covs = site_covs.shape[1]
    n_obs_covs = obs_covs.shape[2]

    obs = pd.DataFrame(obs)
    site_covs = pd.DataFrame(site_covs, columns=[f"site_cov_{i + 1}" for i in range(n_site_covs)]) if site_covs.size > 0 else None
    obs_covs = pd.DataFrame(obs_covs.reshape(-1, n_obs_covs), columns=[f"obs_cov_{i + 1}" for i in range(n_obs_covs)]) if obs_covs.size > 0 else None

    with robjects.local_context() as rctx:
        rctx["y"] = py2r(obs)
        if site_covs is not None:
            rctx["site_covs"] = py2r(site_covs)
        else:
            rctx["site_covs"] = ro.r('NULL')
        if obs_covs is not None:
            rctx["obs_covs"] = py2r(obs_covs)
        else:
            rctx["obs_covs"] = ro.r('NULL')

        ro.r(f'''
            set.seed({random_seed})
            library(unmarked)
            umf <- unmarkedFrameOccu(y = as.matrix(y), siteCovs=site_covs, obsCovs=obs_covs)
            mod = occu(
                data = umf,
                formula = ~ 1{(' + ' if n_obs_covs > 0  else '') + ' + '.join(['obs_cov_' + str(i + 1) for i in range(n_obs_covs)])} ~ 1{(' + ' if n_site_covs > 0  else '') + ' + '.join(['site_cov_' + str(i + 1) for i in range(n_site_covs)])},
                method="Nelder-Mead"
            )
            summary(mod)

            # compute likelihood
            print(logLik(mod))
        ''')

        site_covs_estimated = np.array(
            [ro.r(f"mod@estimates@estimates['state']$state@estimates['(Intercept)']").item()] +
            [ro.r(f"mod@estimates@estimates['state']$state@estimates['site_cov_{i + 1}']").item() for i in range(n_site_covs)]
        )

        obs_covs_estimated = np.array(
            [ro.r(f"mod@estimates@estimates['det']$det@estimates['(Intercept)']").item()] +
            [ro.r(f"mod@estimates@estimates['det']$det@estimates['obs_cov_{i + 1}']").item() for i in range(n_obs_covs)]
        )

        psi_estimated = r2py(ro.r("predict(mod, type = 'state')")).Predicted.to_numpy()

        return site_covs_estimated, obs_covs_estimated, psi_estimated


n_simulations = 5  # TODO: set to ~100

covs_rmse_unmarked = []
covs_rmse_numpyro = []
psi_rmse_unmarked = []
psi_rmse_numpyro = []
n_sites_list = [10, 25, 100, 500, 1000]
for n_sites in n_sites_list:

    for simulation_seed in range(n_simulations):
    
        obs, site_covs, obs_covs, beta, alpha, z = simulate_occu_fp(n_sites=n_sites, n_obs_covs=2, n_site_covs=2, prob_fp=0, random_seed=simulation_seed)

        train_sites, test_sites = train_test_split(np.arange(n_sites), test_size=0.2, random_state=simulation_seed)

        site_covs_estimated, obs_covs_estimated, psi_estimated = fit_r(obs[train_sites], site_covs[train_sites], obs_covs[train_sites], random_seed=simulation_seed)
        covs_rmse_unmarked += [(np.mean((site_covs_estimated - beta) ** 2) + np.mean((obs_covs_estimated - alpha) ** 2)) ** 0.5]

        psi_rmse_unmarked += [np.mean((psi_estimated - z[train_sites]) ** 2) ** 0.5]

        model_comparison_df, mcmc = run(occu, site_covs[train_sites], obs_covs[train_sites], obs=obs[train_sites])

        site_covs_estimated = np.array(
            [model_comparison_df[f'cov_state_intercept'].mean()] +
            [model_comparison_df[f'cov_state_{i}'].mean() for i in range(site_covs.shape[1])]
        )
        obs_covs_estimated = np.array(
            [model_comparison_df[f'cov_det_intercept'].mean()] +
            [model_comparison_df[f'cov_det_{i}'].mean() for i in range(obs_covs.shape[2])]
        )
        covs_rmse_numpyro += [(np.mean((site_covs_estimated - beta) ** 2) + np.mean((obs_covs_estimated - alpha) ** 2)) ** 0.5]

        psi_estimated = model_comparison_df.Predicted.to_numpy()
        psi_rmse_numpyro += [np.mean((psi_estimated - z[train_sites]) ** 2) ** 0.5]

        model_comparison_df_test, mcmc = run(occu, site_covs[test_sites], obs_covs[test_sites], obs=obs[test_sites])



covs_rmse_unmarked = np.array(covs_rmse_unmarked).reshape(-1, n_simulations).mean(axis=1)
covs_rmse_numpyro = np.array(covs_rmse_numpyro).reshape(-1, n_simulations).mean(axis=1)
psi_rmse_unmarked = np.array(psi_rmse_unmarked).reshape(-1, n_simulations).mean(axis=1)
psi_rmse_numpyro = np.array(psi_rmse_numpyro).reshape(-1, n_simulations).mean(axis=1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(n_sites_list, covs_rmse_unmarked, label='Unmarked')
ax1.plot(n_sites_list, covs_rmse_numpyro, label='NumPyro')
ax1.set_xlabel('Number of sites')
ax1.set_ylabel('Covariate RMSE')
ax2.plot(n_sites_list, psi_rmse_unmarked, label='Unmarked')
ax2.plot(n_sites_list, psi_rmse_numpyro, label='NumPyro')
ax2.set_xlabel('Number of sites')
ax2.set_ylabel('Psi RMSE')
ax2.legend()
os.makedirs(os.path.join(os.path.dirname(__file__), 'figures'), exist_ok=True)
fig.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'simulations_nsites.png'), bbox_inches='tight')




covs_rmse_unmarked = []
covs_rmse_numpyro = []
psi_rmse_unmarked = []
psi_rmse_numpyro = []
deployment_days_per_site_list = [7, 14, 28, 56, 112, 365]
for deployment_days_per_site in deployment_days_per_site_list:

    for simulation_seed in range(n_simulations):
    
        obs, site_covs, obs_covs, beta, alpha, z = simulate_occu_fp(n_sites=100, deployment_days_per_site=deployment_days_per_site, n_obs_covs=2, n_site_covs=2, prob_fp=0, random_seed=simulation_seed)

        site_covs_estimated, obs_covs_estimated, psi_estimated = fit_r(obs, site_covs, obs_covs)
        covs_rmse_unmarked += [(np.mean((site_covs_estimated - beta) ** 2) + np.mean((obs_covs_estimated - alpha) ** 2)) ** 0.5]

        psi_rmse_unmarked += [np.mean((psi_estimated - z) ** 2) ** 0.5]

        model_comparison_df, mcmc = run(occu, site_covs, obs_covs, obs=obs)

        site_covs_estimated = np.array(
            [model_comparison_df[f'cov_state_intercept'].mean()] +
            [model_comparison_df[f'cov_state_{i}'].mean() for i in range(site_covs.shape[1])]
        )
        obs_covs_estimated = np.array(
            [model_comparison_df[f'cov_det_intercept'].mean()] +
            [model_comparison_df[f'cov_det_{i}'].mean() for i in range(obs_covs.shape[2])]
        )
        covs_rmse_numpyro += [(np.mean((site_covs_estimated - beta) ** 2) + np.mean((obs_covs_estimated - alpha) ** 2)) ** 0.5]

        psi_estimated = model_comparison_df.Predicted.to_numpy()
        psi_rmse_numpyro += [np.mean((psi_estimated - z) ** 2) ** 0.5]



covs_rmse_unmarked = np.array(covs_rmse_unmarked).reshape(-1, n_simulations).mean(axis=1)
covs_rmse_numpyro = np.array(covs_rmse_numpyro).reshape(-1, n_simulations).mean(axis=1)
psi_rmse_unmarked = np.array(psi_rmse_unmarked).reshape(-1, n_simulations).mean(axis=1)
psi_rmse_numpyro = np.array(psi_rmse_numpyro).reshape(-1, n_simulations).mean(axis=1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(deployment_days_per_site_list, covs_rmse_unmarked, label='Unmarked')
ax1.plot(deployment_days_per_site_list, covs_rmse_numpyro, label='NumPyro')
ax1.set_xlabel('# Deployment days per site')
ax1.set_ylabel('Covariate RMSE')
ax2.plot(deployment_days_per_site_list, psi_rmse_unmarked, label='Unmarked')
ax2.plot(deployment_days_per_site_list, psi_rmse_numpyro, label='NumPyro')
ax2.set_xlabel('# Deployment days per site')
ax2.set_ylabel('Psi RMSE')
ax2.legend()
os.makedirs(os.path.join(os.path.dirname(__file__), 'figures'), exist_ok=True)
fig.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'simulations_duration.png'), bbox_inches='tight')