DOI: [doi.org/10.5281/zenodo.10782748](https://doi.org/10.5281/zenodo.10782748)

[[_TOC_]]

# The 5 models compared

Two discrete models:

- Bernoulli Process (BP) - MacKenzie *et al.* (2002)
- Counting Occurrences Process (COP) - Adapted from Emmet *et al.* (2021) ; see details in Appendix 2.

Three continuous models:

- Poisson Process (PP) - Guillera-Arroita *et al.* (2011)
- Two States Modulated Markov Poisson Process (2-MMPP) - Guillera-Arroita *et al.* (2011)
- Interrupted Poisson Process (IPP) - Guillera-Arroita *et al.* (2011)

# How to run the lynx analysis?

The data was presented in [Gimenez *et al*. (2022)](https://doi.org/10.57750/yfm2-5f45) and is available in the [git repository associated to this publication](https://github.com/oliviergimenez/computo-deeplearning-occupany-lynx). In this comparison, we only use data from the Ain county. 

The data is loaded and analysed in the `Ain_lynx_occupancy.Rmd` file. The notebook with its outputs is also available directly as a html file (`Ain_lynx_occupancy.html`).

# How to run the simulation code?

## Prerequisites

### R version

These scripts were developed and ran with R version 4.3.1.

### R packages

The following R packages are used and can be installed in their latest stable version by executing the following R commands.

```R
install.packages('plyr')       # Data formatting
install.packages('tidyverse')  # Data formatting
install.packages('ggplot2')    # Plot
install.packages('gridExtra')  # Plot
install.packages('glue')       # Python equivalent of f"text {variable}"
install.packages('expm')       # Exponential of a matrix
install.packages('jsonlite')   # Manage json files
install.packages('unmarked')   # BP model
install.packages('progress')   # Progress bar
install.packages('latex2exp')  # LaTeX to R expression for plots
```

If the packages were updated, causing the scripts to malfunction, you should be able to retrieve the specific versions of these packages used in the script by executing the following R commands. You may need to install the latest version of `devtools` to do so, by running the R command `install.packages('devtools')`

```R
require(devtools)
install_version("plyr", version = "1.8.8", repos = "http://cran.us.r-project.org")
install_version("tidyverse", version = "2.0.0", repos = "http://cran.us.r-project.org")
install_version("ggplot2", version = "3.4.3", repos = "http://cran.us.r-project.org")
install_version("gridExtra", version = "2.3", repos = "http://cran.us.r-project.org")
install_version("glue", version = "1.6.2", repos = "http://cran.us.r-project.org")
install_version("expm", version = "0.999.7", repos = "http://cran.us.r-project.org")
install_version("jsonlite", version = "1.8.7", repos = "http://cran.us.r-project.org")
install_version("unmarked", version = "1.3.2", repos = "http://cran.us.r-project.org")
install_version("progress", version = "1.2.2", repos = "http://cran.us.r-project.org")
install_version("latex2exp", version = "0.9.6", repos = "http://cran.us.r-project.org")
```

## Run the same comparisons as the article

#### Run with a command line

The script `run_comparisons.R` can be used for this. It can be used in command-lines, with those arguments:

```bash
Rscript run_comparisons.R <try_seed_min> <try_seed_max> <output_path> <run_suffix> <optim_method>
```

With `try_seed_min` and `try_seed_max` integers, the minimum and maximum value of the seed that will be set to produce reproducible data sets and results. `optim_method` is a string for the optimisation method in the `optim` function in R (`"Nelder-Mead", "BFGS", "CG", "L-BFGS-B", "SANN", "Brent"`, see `?optim`). `output_path` and `run_suffix` will define the output file path and name, such as the file produced will be in directory `<output_path>` and named `OccModComp_S100_R1_T100_seed0-500_Nelder-Mead_<run_suffix>.json`.

To have the same comparisons as the article, the seeds are from 0 to 500 and the optimisation method is Nelder-Mead. The arguments can be, for example:

```bash
Rscript run_comparisons.R 0 500 './output/' '2023-10-19' 'Nelder-Mead'
```

This code was not parallelised or optimised. Therefore, to run it faster, you can also run it on different machines or in different threads of the same machine by running simultaneously several command lines with varying seed values:

```bash
Rscript run_comparisons.R 0 250 './output/' '2023-10-19' 'Nelder-Mead'
Rscript run_comparisons.R 251 500 './output/' '2023-10-19' 'Nelder-Mead'
```

>  Note: this command line was only tested with Linux and may need to be adapted with other OS

#### Run interactively in an IDE (*e.g.* RStudio)

If you want to run the script in an IDE such as RStudio, the following values are initialised in lines 19 to 23 of the script:

```R
try_seed_min = 0
try_seed_max = 500
output_path = './output/'
run_suffix = Sys.Date()
optim_method = 'Nelder-Mead'
```

## Run your own comparisons

If you want to run your own comparisons, you can adapt the "COMPARISONS TO DO" part of the script, changing the following variables:

- `NbSites`: the number of sites in the simulated data set
- `DeployementTimeValues`: the duration of the deployment, in a given time-unit (e.g day)
- `try_seed`: a vector of integers, for the seed tried for each comparison scenario
- `try_psi`: a vector of numeric (between 0 and 1 included), for the occupancy probabilities (psi $\psi$)
- `try_lambda_mu`: a vector of strings, describing lambda and mu for the detection parameters, with each string in this format: `"lambda_1, lambda_2 ; mu_12, mu_21"`, for example `"0, 1 ; 0.0667, 30"`
- `try_SessionLength`: a vector of numerics, for the length of a session, in the same time-unit as `DeployementTimeValues`

> :warning: `NbDeployPerSite = 1` should not be changed because the calculation of the log-likelihood for the discrete models were not adapted for several deployments per site.

You could also directly create a dataframe with the same columns as `experimental_plan`, with at least the following columns:

|  psi | lambda_1 | lambda_2 |  mu_12 | mu_21 | SessionLength | seed | run_continuous |
| ---: | -------: | -------: | -----: | ----: | :------------ | ---: | :------------- |
| 0.10 |        0 |        1 | 0.0667 |    30 | 30.0000       |    0 | TRUE           |
| 0.25 |        0 |        1 | 0.0667 |    30 | 30.0000       |    0 | TRUE           |
| 0.50 |        0 |        1 | 0.0667 |    30 | 30.0000       |    0 | TRUE           |

## Read and visualise the comparison's results

To read the results that were written into a json, you can use the following code, with the function `Flattener` defined in script `./utils/0_general_functions.R`.

```R
jsonlite::read_json(path='/path/to/result/file.json', simplifyVector = T) %>%
      Flattener()
```

The script `get_figures_article.R` produces figures that are displayed in the article and more.

## Warnings and errors

### Warning: Hessian is singular

```R
Hessian is singular. Try providing starting values or using fewer covariates.
```

This warning is normal and expected for the 2-MMPP model when the data is simulated according to an IPP framework. It is because $\lambda_1$ is simulated as 0, so $\lambda_1$ is estimated as 0. The Hessian matrix can therefore look like this example:

```
                            psi lambda_1.lambda_1 lambda_2.lambda_2   mu_12.mu_12   mu_21.mu_21
psi                1.959365e+00                 0     -1.211387e-05 -2.349232e-07 -7.176926e-06
lambda_1.lambda_1  0.000000e+00                 0      0.000000e+00  0.000000e+00  0.000000e+00
lambda_2.lambda_2 -1.211387e-05                 0      1.827388e+01 -6.992659e-01 -7.586247e+00
mu_12.mu_12       -2.349232e-07                 0     -6.992659e-01  1.953935e+01  1.861288e+00
mu_21.mu_21       -7.176926e-06                 0     -7.586247e+00  1.861288e+00  1.537949e+01
```

We can see that all values are 0 in both the column and the row named `lambda_1.lambda_1`.  The Hessian is singular: the parameters are collinear.

# Project structure

The main script to run all the simulations is `./run_comparisons.R`. The folder `./utils/` contains scripts that define the functions used in this comparison:

- `./utils/1_detection_simulation.R`
  - Functions that simulate the data set with a 2-MMPP (`switch_state`, `simulate_2MMPP`, `simulate_2MMPP_V2`)
  - Functions that get the detection probability pf having at least one detection in an occupied site during a given time depending on each model parameters (`get_p_from_2MMPP_param`, `get_p_from_PP_param`, `get_p_from_COP_param`)
  - Functions that plot the simulated data set and extract summary informations from it  (`extract_simulated_infos`, `plot_detection_times`)
  - Functions that discretise the simulated data to use as inputs of the COP and BP models (`get_nbdetec_per_session`, `get_detected_per_session`)
- `./utils/2_occupancy_models.R`
  - Functions to transform the probability from $[0,1]$ to ℝ (`logit`, `invlogit`)
  - Functions to calculate the log-likelihood for a 2-MMPP (`get_2MMPP_M_ij`, `get_2MMPP_loglikelihood`, `get_2MMPP_neg_loglikelihood`)
  - Functions to calculate the log-likelihood for a IPP (`get_IPP_loglikelihood`, `get_IPP_neg_loglikelihood`)
  - Functions to calculate the log-likelihood for a PP (`get_PP_loglikelihood`, `get_PP_neg_loglikelihood`)
  - Functions to calculate the log-likelihood for a COP (`get_COP_loglikelihood`, `get_COP_neg_loglikelihood`)
- `./utils/comparison_of_occupancy_models.R` : contains only one function, `run_one_comparison`. For one simulation scenario, this function will simulate a data set, and fit all five models by likelihood maximisation. It returns the results of all five models and can export the results to a json if required.
- `./utils/0_general_functions.R`: General functions that are used throughout the procedure (`newpb`, `solvable`, `Flattener`, `calcul_rmse`, `SessionLength_to_text`, `stars.pval`)

All the functions in these scripts are documented using the following pattern:

```
  "
  Function name
  Function short description
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  input1
    Description
  
  input2
    (facultative)
    Description
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  Description
  
  USE ──────────────────────────────────────────────────────────────────────────
  If necessary, an example of use of the function
  "
```

# References

Emmet, R.L., Long, R.A. and Gardner, B. (2021)  Modeling multi-scale occupancy for monitoring rare and highly mobile species , *Ecosphere*, 12(7), p. e03637. Available at: https://doi.org/10.1002/ecs2.3637.

Guillera-Arroita, G. *et al.* (2011)  Species Occupancy Modeling for Detection Data Collected Along a Transect , *Journal of Agricultural, Biological, and Environmental Statistics*, 16(3), pp. 301 317. Available at: https://doi.org/10.1007/s13253-010-0053-3.

MacKenzie, D.I. *et al.* (2002)  Estimating Site Occupancy Rates When Detection Probabilities Are Less Than One , *Ecology*, 83(8), pp. 2248 2255. Available at: https://doi.org/10.2307/3072056.
