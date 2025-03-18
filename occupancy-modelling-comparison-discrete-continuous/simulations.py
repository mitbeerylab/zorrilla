import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import rpy2
import rpy2.robjects as robjects
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()

from biolith.models import occu, simulate
from biolith.utils import fit, predict


py2r = ro.conversion.get_conversion().py2rpy
r2py = ro.conversion.get_conversion().rpy2py


def fit_unmarked(obs_train, site_covs_train, obs_covs_train, site_covs_test, obs_covs_test, random_seed=0):

    n_site_covs = site_covs_train.shape[1]
    n_obs_covs = obs_covs_train.shape[2]

    obs_train = pd.DataFrame(obs_train)
    site_covs_train = pd.DataFrame(site_covs_train, columns=[f"site_cov_{i + 1}" for i in range(n_site_covs)]) if site_covs_train.size > 0 else None
    site_covs_test = pd.DataFrame(site_covs_test, columns=[f"site_cov_{i + 1}" for i in range(n_site_covs)]) if site_covs_test.size > 0 else None
    obs_covs_train = pd.DataFrame(obs_covs_train.reshape(-1, n_obs_covs), columns=[f"obs_cov_{i + 1}" for i in range(n_obs_covs)]) if obs_covs_train.size > 0 else None

    with robjects.local_context() as rctx:
        rctx["y"] = py2r(obs_train)

        for obj, name in [
            (site_covs_train, "site_covs_train"),
            (obs_covs_train, "obs_covs_train"),
            (site_covs_test, "site_covs_test"),
        ]:
            if obj is not None:
                rctx[name] = py2r(obj)
            else:
                rctx[name] = ro.r('NULL')

        ro.r(f'''
            set.seed({random_seed})
            library(unmarked)
            umf <- unmarkedFrameOccu(y = as.matrix(y), siteCovs=site_covs_train, obsCovs=obs_covs_train)
            mod = occu(
                data = umf,
                formula = ~ 1{(' + ' if n_obs_covs > 0  else '') + ' + '.join(['obs_cov_' + str(i + 1) for i in range(n_obs_covs)])} ~ 1{(' + ' if n_site_covs > 0  else '') + ' + '.join(['site_cov_' + str(i + 1) for i in range(n_site_covs)])},
                method="Nelder-Mead"
            )
        ''')

        site_covs_estimated = np.array(
            [ro.r(f"mod@estimates@estimates['state']$state@estimates['(Intercept)']").item()] +
            [ro.r(f"mod@estimates@estimates['state']$state@estimates['site_cov_{i + 1}']").item() for i in range(n_site_covs)]
        )

        obs_covs_estimated = np.array(
            [ro.r(f"mod@estimates@estimates['det']$det@estimates['(Intercept)']").item()] +
            [ro.r(f"mod@estimates@estimates['det']$det@estimates['obs_cov_{i + 1}']").item() for i in range(n_obs_covs)]
        )

        psi_estimated_train = r2py(ro.r("predict(mod, type = 'state')")).Predicted.to_numpy()
        psi_estimated_test = r2py(ro.r("predict(mod, type = 'state', newdata = site_covs_test)")).Predicted.to_numpy()

        return site_covs_estimated, obs_covs_estimated, psi_estimated_train, psi_estimated_test


def fit_spocc(obs_train, site_covs_train, obs_covs_train, site_covs_test, obs_covs_test, random_seed=0):
    n_site_covs = site_covs_train.shape[1]
    n_obs_covs = obs_covs_train.shape[2]

    obs_covs_train = robjects.ListVector({f"obs_cov_{i + 1}": obs_covs_train[..., i] for i in range(obs_covs_train.shape[-1])})
    site_covs_colnames = [f"site_cov_{i + 1}" for i in range(n_site_covs)]

    with robjects.local_context() as rctx:
        rctx["y"] = py2r(obs_train)

        for obj, name in [
            (site_covs_train, "site_covs_train"),
            (obs_covs_train, "obs_covs_train"),
            (site_covs_test, "site_covs_test"),
            (site_covs_colnames, "site_covs_colnames"),
        ]:
            if obj is not None:
                rctx[name] = py2r(obj)
            else:
                rctx[name] = ro.r('NULL')

        ro.r(f'''
            set.seed({random_seed})
            library(spOccupancy)
            n.samples = 2000
            n.burn = 1000
            n.report = 1000
            colnames(site_covs_train) <- site_covs_colnames
            colnames(site_covs_test) <- site_covs_colnames
            occ.formula = ~ 1{(' + ' if n_site_covs > 0  else '') + ' + '.join(['site_cov_' + str(i + 1) for i in range(n_site_covs)])}
            det.formula = ~ 1{(' + ' if n_obs_covs > 0  else '') + ' + '.join(['obs_cov_' + str(i + 1) for i in range(n_obs_covs)])}
            data <- list(y=y, occ.covs=site_covs_train, det.covs=obs_covs_train)
            priors <- list(alpha.normal = list(mean = 0, var = 1), beta.normal = list(mean = 0, var = 1))
            inits <- list()
            out = PGOcc(occ.formula, det.formula, data, inits, priors, n.samples, 
                n.omp.threads = {os.getenv("SLURM_JOB_CPUS_PER_NODE", "1")}, verbose = TRUE, n.report = n.report, 
                n.burn = n.burn, n.thin = 1, n.chains = 1,
            )

            pred = predict(out, cbind(1, site_covs_test))
        ''')

        samples_train = {var: r2py(ro.r(f"out${var}.samples")) for var in ["alpha", "beta", "psi"]}
        samples_test = {var: r2py(ro.r(f"pred${var}.0.samples")) for var in ["alpha", "beta", "psi"]}

        site_covs_estimated = samples_train["beta"].mean(axis=0)
        obs_covs_estimated = samples_train["alpha"].mean(axis=0)
        psi_estimated_train = samples_train["psi"].mean(axis=0)
        psi_estimated_test = samples_test["psi"].mean(axis=0)

        return site_covs_estimated, obs_covs_estimated, psi_estimated_train, psi_estimated_test


def fit_numpyro(obs_train, site_covs_train, obs_covs_train, site_covs_test, obs_covs_test, random_seed=0):
    results_train = fit(occu, site_covs_train, obs_covs_train, obs_train, random_seed=random_seed)
    site_covs_estimated = np.array([results_train.samples[k].mean() for k in [f"cov_state_{i}" for i in range(len(true_params["beta"]))]])
    obs_covs_estimated = np.array([results_train.samples[k].mean() for k in [f"cov_det_{i}" for i in range(len(true_params["alpha"]))]])
    psi_estimated_train = results_train.samples["psi"].mean(axis=0)

    samples_test = predict(occu, results_train.mcmc, site_covs=site_covs_test, obs_covs=obs_covs_test, random_seed=random_seed)
    psi_estimated_test = samples_test["psi"].mean(axis=0)

    return site_covs_estimated, obs_covs_estimated, psi_estimated_train, psi_estimated_test


df = []
n_simulations = 10  # TODO: set to ~100
n_sites_list = [50, 100, 500, 1000, 10000]
deployment_days_per_site_list = [28, 56, 112, 365]

# # TODO: remove
# n_sites_list = [50, 100]
# deployment_days_per_site_list = [28, 56]


for n_sites in n_sites_list:
    for deployment_days_per_site in deployment_days_per_site_list:
        for replicate in range(n_simulations):
            data, true_params = simulate(n_sites=n_sites, deployment_days_per_site=deployment_days_per_site, n_obs_covs=2, n_site_covs=2, simulate_missing=False, random_seed=replicate)
            obs = data['obs']
            site_covs = data['site_covs']
            obs_covs = data['obs_covs']
            z = true_params['z']
            beta = true_params['beta']
            alpha = true_params['alpha']

            train_sites, test_sites = train_test_split(np.arange(n_sites), test_size=0.2, random_state=replicate)

            for fit_method, method_name in [
                (fit_spocc, 'spOcc'),
                (fit_unmarked, 'Unmarked'),
                (fit_numpyro, 'NumPyro'),
            ]:
                time_start = time.monotonic()
                site_covs_estimated, obs_covs_estimated, psi_estimated_train, psi_estimated_test = fit_method(obs[train_sites], site_covs[train_sites], obs_covs[train_sites], site_covs[test_sites], obs_covs[test_sites], random_seed=replicate)
                time_end = time.monotonic()

                for split_name, split_sites, psi_estimated in [("train", train_sites, psi_estimated_train), ("test", test_sites, psi_estimated_test)]:
                    covs_rmse = (np.mean((site_covs_estimated - beta) ** 2) + np.mean((obs_covs_estimated - alpha) ** 2)) ** 0.5
                    psi_rmse = np.mean((psi_estimated - z[split_sites]) ** 2) ** 0.5
                    z_acc = np.mean((psi_estimated > 0.5) == z[split_sites])

                    df.append(dict(
                        n_sites=n_sites,
                        deployment_days_per_site=deployment_days_per_site,
                        replicate=replicate,
                        method=method_name,
                        split=split_name,
                        covs_rmse=covs_rmse,
                        psi_rmse=psi_rmse,
                        z_acc=z_acc,
                        time=time_end - time_start,
                    ))

        output_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(df).to_csv(os.path.join(output_dir, "simulation_results.csv"), index=False)