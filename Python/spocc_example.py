import os
import io
from multiprocessing import Pool
import contextlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()


from rutils import rnamedlist


utils = importr("utils")
base = importr("base")
spOccupancy = importr("spOccupancy")


def inv_logit(x):
    return np.exp(x)/(1+np.exp(x))


def generate_and_fit(run_idx):

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):

        # seed RNG for reproducibility
        np.random.seed(run_idx)

        # Generate site occupancy
        nsites = 200 # number of sites
        occ_covar = np.random.normal(size=nsites, loc=0, scale=1)
        occ_beta0 = -1
        occ_beta1 = 1
        lin_occ_function = occ_beta0 + occ_beta1*occ_covar
        prop_occupancy = inv_logit(lin_occ_function)
        z = np.random.binomial(size=nsites, n=1, p=prop_occupancy) # vector of latent occupancy status for each site

        prop_images_present = 0.025 # given an occupied site, proportion of images where species is present
        images_per_day = 100 # number of camera trap images per day
        days_per_year = 365 # number of days of operation per year
        false_positive_lin_function = -10 # can modify to be a linear model of some sort, for now this gives a very small false positive rate 
        false_positive_rate = inv_logit(false_positive_lin_function) 

        detections = np.full((nsites, days_per_year, images_per_day), fill_value=-1, dtype=np.int64)

        # Generate detections
        for i in range(nsites):
            for j in range(days_per_year):
                for k in range(images_per_day):
                    if z[i] == 0: # Unoccupied sites will only have false positives
                        detections[i,j,k] = np.random.binomial(size=1, n=1, p=false_positive_rate).item()
                    else: # Occupied sites
                        detections[i,j,k] = np.random.binomial(size=1, n=1, p=false_positive_rate + prop_images_present).item()
        assert np.all(detections >= 0)

        site_daily_detections_sum = np.sum(detections, axis=-1)
        site_daily_detections_binary = np.max(detections, axis=-1)

        occ_formula = robjects.Formula("~covs")
        occ_formula.environment["covs"] = pd.Series(occ_covar)
        det_formula = robjects.Formula("~1")

        inits = rnamedlist({
            "alpha": 0, 
            "beta":  0, 
            "z":     np.max(site_daily_detections_binary, axis=-1).tolist(),
        })
        priors = rnamedlist({
            "alpha.normal": {"mean": 0, "var": 3}, 
            "beta.normal":  {"mean": 0, "var": 3},
        })
        n_samples = 5000
        n_burn = 3000
        n_thin = 2
        n_chains = 3
        input_data = rnamedlist({
            "y": site_daily_detections_binary,
            "occ.covs": pd.Series(occ_covar),
        })

        # TODO: find a more idiomatic way to define 'input_data'
        robjects.globalenv["site_daily_detections_binary"] = site_daily_detections_binary
        robjects.globalenv["occ_covar"] = occ_covar
        input_data = robjects.r("list( y = site_daily_detections_binary, occ.covs = as.data.frame(occ_covar))")

        out = spOccupancy.PGOcc(
            occ_formula=occ_formula, 
            det_formula=det_formula, 
            data=input_data, 
            inits=inits, 
            n_samples=n_samples, 
            priors=priors, 
            n_omp_threads=1, 
            verbose=True, 
            n_report=1000, 
            n_burn=n_burn, 
            n_thin=n_thin, 
            n_chains=n_chains,
        )

        base.summary(out)

    summary = io.StringIO()
    with contextlib.redirect_stdout(summary), contextlib.redirect_stderr(summary):

        # Posterior predictive check
        ppc_out = spOccupancy.ppcOcc(out, fit_stat="freeman-tukey", group=1)
        base.summary(ppc_out)

    return summary.getvalue()


if __name__ == "__main__":
    n_proc = os.cpu_count()
    n_runs = 10

    with Pool(n_proc) as p:
        print("\n\n\n".join(list(tqdm(p.imap(generate_and_fit, range(n_runs)), total=n_runs))))