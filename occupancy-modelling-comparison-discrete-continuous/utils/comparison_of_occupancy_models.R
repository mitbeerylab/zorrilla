# ╭─────────────────────────────────────────────────────────────────────────────╮
# │                                                                             │
# │ Comparing occupancy models                                                  │
# │ Léa Pautrel, TerrOïko | CEFE | IRMAR                                        │
# │ Last update: October 2023                                                   │
# │                                                                             │
# ╰─────────────────────────────────────────────────────────────────────────────╯

"
The aim of this script is to compare and evaluate several occupancy models, 
depending on how time is taking into account.

The models that will be compared are:

- Three continous time occupancy model, 2-MMPP, IPP, PP 
  (Guillera-Arroita et al., 2011)

- A discrete count occupancy model, COP
  (inspired from Emmet et al., 2021, but simplified by dropping the usage 
  parameter and the secondary sampling occasion)

- A discrete detection/non detection occupancy model, BP
  (MacKenzie et al., 2002)

For this, purpose, we will simulate a detection dataset according to a 
2-states Markov Modulated Poisson Process (2-MMPP).

SOURCES

- Guillera-Arroita, G., Morgan, B.J.T., Ridout, M.S., Linkie, M., 2011. 
  Species Occupancy Modeling for Detection Data Collected Along a Transect. 
  JABES 16, 301–317. https://doi.org/10.1007/s13253-010-0053-3

- Emmet, R.L., Long, R.A., Gardner, B., 2021. 
  Modeling multi-scale occupancy for monitoring rare and highly mobile species. 
  Ecosphere 12, e03637. https://doi.org/10.1002/ecs2.3637

- Kellner, K.F., Parsons, A.W., Kays, R., Millspaugh, J.J., Rota, C.T., 2022. 
  A Two-Species Occupancy Model with a Continuous-Time Detection Process Reveals Spatial and Temporal Interactions. 
  JABES 27, 321–338. https://doi.org/10.1007/s13253-021-00482-y


RUN THE CODE WITH EXAMPLE PARAMETERS

If you want to run the code with example parameters instead of running the whole function at once, 
you can use the following arguments for example:

NbSites = 100
NbDeployPerSite = 1
DeployementTimeValues = 100
psi = 0.5
lambda = c('lambda_1' = 0, 'lambda_2' = 5)
mu = c('mu_12' = 1 / 15, 'mu_21' = 1)
SessionLength = 7
ComparisonResult_ExportPath = NULL
quiet = FALSE
seed = NULL
run_continuous_models = TRUE
run_discrete_models = TRUE
optim_method = 'Nelder-Mead'
"

# ──────────────────────────────────────────────────────────────────────────────
# LIBRARIES                                                                 ----
# ──────────────────────────────────────────────────────────────────────────────

require(plyr)       # Data formatting
require(tidyverse)  # Data formatting
require(ggplot2)    # Plot
require(gridExtra)  # Plot
require(glue)       # Python equivalent of f"text {variable}"
require(expm)       # Exponential of a matrix
require(jsonlite)   # Manage json files
require(unmarked)   # BP model
require(progress)   # Progress bar

# ──────────────────────────────────────────────────────────────────────────────
# SOURCE FUNCTIONS                                                          ----
# ──────────────────────────────────────────────────────────────────────────────

# Importing the generally useful functions (progress bar, data transformation..)
source("./utils/0_general_functions.R")

# Importing the functions to simulate the detection dataset
source("./utils/1_detection_simulation.R")

# Importing the occupancy model functions
source("./utils/2_occupancy_models.R")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION TO RUN 1 COMPARISON                                              ----
# ──────────────────────────────────────────────────────────────────────────────

run_one_comparison = function(NbSites,
                              NbDeployPerSite,
                              DeployementTimeValues,
                              psi,
                              lambda,
                              mu,
                              SessionLength,
                              ComparisonResult_ExportPath = NULL,
                              optim_method = "Nelder-Mead",
                              quiet = FALSE,
                              seed = NULL,
                              run_continuous_models = TRUE,
                              run_discrete_models = TRUE) {
  # browser()
  "
  run_one_comparison
  For one simulation scenario, this function will simulate a data set, 
  and fit all five models by likelihood maximisation. It returns the results of 
  all five models and can export the results to a json if required.
  
  INPUTS ───────────────────────────────────────────────────────────────────────

  NbSites
    Number of sites
    Integer

  NbDeployPerSite
    Number of deployments per site
    Integer or vector of integers
    (random selection with replacement from values in vector)

  DeployementTimeValues
    Deployment time in each site and deployment
    Numeric or vector of numerics
    (random selection with replacement from values in vector)

  psi
    Occupancy probability for the simulations
    Numeric between 0 and 1 included

  lambda
    Detection rates of the 2-MMPP for the simulations
    If one of the lambda is 0, this is an Interrupted Poisson Process (IPP)
    Given as a vector of c(lambda_1, lambda_2), with positive or nul numerics

  mu
    Switching rates between Markov states of the 2-MMPP for the simulations
    Given as a vector of c(mu_12, mu_21), with positive or nul numerics

  SessionLength
    Session length for discretisation
    Numeric, in the same time-unit as DeploymentTimeValues
  
  ComparisonResult_ExportPath
    (facultative, NULL by default)
    Path to the json to export the result of the comparison.
    If NULL, there will be no export.
    String or NULL
  
  optim_method
    (facultative, 'Nelder-Mead' by default)
    Optimisation method. See the possible values with `?optim`.
    String
  
  quiet
    (facultative, FALSE by default)
    Boolean, FALSE to display the results, TRUE to print nothing.
  
  seed
    (facultative, NULL by default)
    For reproducibility, a random seed can be set.
    For true random, use seed=NULL.
    Integer or NULL.
  
  run_continuous_models
    (facultative, TRUE by default)
    If TRUE, fit the continuous models (2-MMPP, IPP, PP).
    If FALSE, skip these models.
    Boolean. 
  
  run_discrete_models
    (facultative, TRUE by default)
    If TRUE, fit the discrete models (COP, BP).
    If FALSE, skip these models.
    Boolean. 


  OUTPUT ───────────────────────────────────────────────────────────────────────
  Result of the occupancy model for this simulation.
  In a list format similar to the json export, described below.
  
   $ param
  ..$ seed: given in the function inputs
  ..$ psi: given in the function inputs
  ..$ lambda_1: given in the function inputs
  ..$ lambda_2: given in the function inputs
  ..$ mu_12: given in the function inputs
  ..$ mu_21: given in the function inputs
  ..$ SessionLength: given in the function inputs
  ..$ NbSites: given in the function inputs
  ..$ NbDeployPerSite: given in the function inputs
  ..$ DeployementTimeValues: given in the function inputs
  ..$ p:  probability of having at least one detection in an occupied site
      given the parameters lambda and mu of the 2-MMPP during the
      deployment of duration DeployementTimeValues
  ..$ nb_sites_occupied: number of occupied sites in the simulated data set
  ..$ nb_detec_total: total number of detections in the simulated data set
      (in all sites, during all the deployments)
  ..$ nb_detec_average_per_deployment_when_occupied: average number of detections
      in occupied sites per deployment in the simulated data set
  ..$ detec_proba_data_deploy1: measured probability of having at least one 
      detection during the first deployment in the simulated data set
  ..$ Time_Simulation: Time (in seconds) taken to create the simulated data set
  ..$ optim_method: given in the function inputs
  
 $ twoMMPP: Results of the 2-MMPP model fit
  ..$ psi: estimation of the parameter
  ..$ lambda_1: estimation of the parameter
  ..$ lambda_2: estimation of the parameter
  ..$ mu_12: estimation of the parameter
  ..$ mu_21: estimation of the parameter
  ..$ fitting_time: Time (in seconds) taken to fit the model with optim
  ..$ hessian: string to get the hessian matrix from optim as an R matrix
  ..$ psi_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ psi_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ lambda_1_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ lambda_1_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ lambda_2_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ lambda_2_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ mu_12_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ mu_12_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ mu_21_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ mu_21_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation

 $ IPP: Results of the IPP model fit
  ..$ psi: estimation of the parameter
  ..$ lambda_1: 0
  ..$ lambda_2: estimation of the parameter
  ..$ mu_12: estimation of the parameter
  ..$ mu_21: estimation of the parameter
  ..$ fitting_time: Time (in seconds) taken to fit the model with optim
  ..$ hessian: string to get the hessian matrix from optim as an R matrix
  ..$ psi_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ psi_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ lambda_2_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ lambda_2_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ mu_12_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ mu_12_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ mu_21_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ mu_21_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation
  
 $ PP: Results of the PP model fit
  ..$ psi: estimation of the parameter
  ..$ lambda: estimation of the parameter
  ..$ fitting_time: Time (in seconds) taken to fit the model with optim
  ..$ hessian: string to get the hessian matrix from optim as an R matrix
  ..$ psi_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ psi_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ lambda_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ lambda_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation
  
 $ COP: Results of the COP model fit
  ..$ psi: estimation of the parameter
  ..$ lambda_Tsession: estimation of the parameter
  ..$ fitting_time: Time (in seconds) taken to fit the model with optim
  ..$ hessian: string to get the hessian matrix from optim as an R matrix
  ..$ psi_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ psi_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ lambda_Tsession_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ lambda_Tsession_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation

 $ BP: Results of the BP model fit
  ..$ psi: estimation of the parameter
  ..$ p: estimation of the parameter
  ..$ fitting_time: Time (in seconds) taken to fit the model with optim
  ..$ hessian: string to get the hessian matrix from optim as an R matrix
  ..$ psi_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ psi_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ p_lower_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation 
  ..$ p_upper_95CI: if the hessian is solvable, 95% confidence interval for the parameter estimation

 $ total_time: Time (in seconds) taken to run the entire comparison

  
  USE ──────────────────────────────────────────────────────────────────────────
  You can try this function by running the following code:

  res = run_one_comparison(
    NbSites = 100,
    NbDeployPerSite = 1,
    DeployementTimeValues = 100,
    psi = 0.5,
    lambda = c('lambda_1' = 0, 'lambda_2' = 5),
    mu = c('mu_12' = 1/15, 'mu_21' = 1),
    SessionLength = 7,
    ComparisonResult_ExportPath = NULL,
    quiet = FALSE,
    seed = NULL,
    run_continuous_models = TRUE,
    run_discrete_models = TRUE
  )
  print(res)
  "
  
  begintime = Sys.time()
  
  # For reproductibility, a random seed is set.
  set.seed(seed)
  
  # Theme for plots
  ggplot2::theme_set(theme_light())
  
  # log error file
  if (!is.null(ComparisonResult_ExportPath)) {
    log_ExportPath = ComparisonResult_ExportPath %>%
      str_replace(pattern='.json', '_ERROR_LOG.txt')
  } else {
    log_ExportPath = NULL
  }
  
  # Check if the output path exists.
  if (!is.null(ComparisonResult_ExportPath)) {
    if (!dir.exists(dirname(ComparisonResult_ExportPath))) {
      cat(
        glue::glue(
          "ComparisonResult_ExportPath = {ComparisonResult_ExportPath}"
        ),
        fill = T
      )
      stop(glue::glue(
        "The directory '{dirname(ComparisonResult_ExportPath)}' does not exist."
      ))
    }
  }
  
  # ──────────────────────────────────────────────────────────────────────────────
  # SIMULATION                                                                ----
  # ──────────────────────────────────────────────────────────────────────────────
  
  ## Simulate the dataset --------------------------------------------------------
  cat("Simulation")
  beforetime = Sys.time()
  SimulatedDataset = simulate_2MMPP(
    NbSites = NbSites,
    NbDeployPerSite = NbDeployPerSite,
    DeployementTimeValues = DeployementTimeValues,
    psi = psi,
    lambda = lambda,
    mu = mu
  )
  aftertime = Sys.time()
  Time_Simulation = as.numeric(aftertime - beforetime, units = "secs")
  
  ## What does the simulated dataset look like? ----------------------------------
  
  # Retrieve the main informations from the simulated dataset
  main_simul_infos = extract_simulated_infos(SimulatedDataset, quiet = quiet)
  
  # z_i[i] is the occupation state of site i (1 if occupied, 0 if not)
  z_i = main_simul_infos[["z_i"]]
  
  # list_R_i[i] is the number of deployments at site i
  list_R_i = main_simul_infos[["list_R_i"]]
  
  # list_T_ij[[i]][j]] is the time during which a sensor is deployed for deployment j at site i
  list_T_ij = main_simul_infos[["list_T_ij"]]
  
  # RecapNDetec is a dataframe that summarises the number of detections per site and deployment
  RecapNDetec = main_simul_infos[["RecapNDetec"]]
  
  # Cleaning up R environment
  rm(main_simul_infos)
  invisible(gc())
  
  # Lets calculate the average lambda of the 2MMPP process
  # (for the PP model and the COP model)
  pi_vect = matrix(c(
    "pi_1" = mu[2] / (mu[1] + mu[2]),
    "pi_2" = mu[1] / (mu[1] + mu[2])
  ),
  nrow = 1,
  ncol = 2)
  
  # Calculation of p 
  p = get_p_from_2MMPP_param(lambda = lambda, mu = mu, pT = DeployementTimeValues)
  p_session = get_p_from_2MMPP_param(lambda = lambda, mu = mu, pT = SessionLength)
  
  # Calculation iof expected number of detections
  Expected_N = unname((lambda[1] * pi_vect[1] + lambda[2] * pi_vect[2]) * DeployementTimeValues)
  
  # Print
  if (!quiet) {
    cat(
      glue::glue(
        "-> p probability of having at least 1 detection = {paste(round(p,3),collapse = ', ')}, ",
        "-> Expected number of detections = {paste(round(Expected_N,2),collapse = ', ')}, ",
        "With simulation parameters:",
        "\tdeployment time = {paste(DeployementTimeValues, collapse = ', ')}",
        "\tmu = ({paste(round(mu, 4), collapse = ', ')})",
        "\tlambda = ({paste(round(lambda, 4), collapse = ', ')})\n",
        .sep = "\n"
      )
    )
  }
  
  # The number of sites that are occupied but in which
  # there is no detection during the 1st deployment:
  nb_sites_occupied_no_detec_deploy1 = RecapNDetec %>%
    filter(Occupied == 1) %>%
    mutate(nodetec = Deployment1 == 0) %>%
    pull(nodetec) %>%
    sum()
  
  # Plot the simulated dataset (only the occupied sites)
  if (!quiet) {
    
    
    # Number of detections per site
    gg = ggplot(data = RecapNDetec %>% filter(Occupied==1)) +
      geom_bar(aes(x = Deployment1, fill = ifelse(Deployment1>0, 'Yes', 'No'))) +
      labs(x = "Number of detections during the 1st deployment", y = "Number of sites", fill = "Detections") +
      geom_label(
        aes(
          label = glue::glue(
            "Occupancy probability psi = {psi}
          Occupied sites: {sum(z_i==1)} occupied sites
          Unoccupied sites: {sum(z_i==0)}

          Detection probability p = {round(p,2)}
          Occupied sites with detections: {round(NbSites-nb_sites_occupied_no_detec_deploy1/sum(z_i==1)*100)} %
          "
          ),
          x = max(RecapNDetec$Deployment1),
          y = RecapNDetec %>%  dplyr::filter(Occupied == 1) %>%  dplyr::group_by(Deployment1) %>% 
            dplyr::summarise(n = n()) %>%  dplyr::pull(n) %>% max() - 1
        ),
        fill = "white", hjust = "right", vjust = "top"
      ) +
      theme(legend.position = 'none')
    print(gg)
    
    # Sample only occupied sites
    sampled_sites = names(RecapNDetec$Occupied[RecapNDetec$Occupied == 1] == 1) %>%
      readr::parse_number() %>%
      sort()
    
    # Plot
    plots = plot_detection_times(
      SimulatedDataset = SimulatedDataset[sampled_sites],
      z_i = z_i[sampled_sites],
      list_R_i = list_R_i[sampled_sites],
      list_T_ij = list_T_ij[sampled_sites],
      RecapNDetec = RecapNDetec[sampled_sites, ],
      lambda = lambda,
      mu = mu
    )
    print(plots)
  }
  
  
  # lambda equivalent for the PP (for optim initialisation)
  lambda_PP <- unname(pi_vect[1] * lambda[1] + pi_vect[2] * lambda[2])
  
  # From "SimulatedDataset" to "detection_times" format
  detection_times <- lapply(SimulatedDataset, function(x) {
    lapply(x[-length(x)], function(y) {
      y[[2]]
    })
  })
  
  
  
  # ──────────────────────────────────────────────────────────────────────────────
  # RESULT LIST INITIALISATION                                                ----
  # ──────────────────────────────────────────────────────────────────────────────
  
  ComparisonResult = list()
  ComparisonResult[[1]] <- list(
    
    "param" = list(
      "seed" = ifelse(is.null(seed), "NULL", seed),
      "psi" = psi,
      "lambda_1" = unname(lambda[1]),
      "lambda_2" = unname(lambda[2]),
      "mu_12" = unname(mu[1]),
      "mu_21" = unname(mu[2]),
      "SessionLength" = SessionLength,
      "NbSites" = paste(NbSites, collapse = ", "),
      "NbDeployPerSite" = paste(NbDeployPerSite, collapse = ", "),
      "DeployementTimeValues" = paste(DeployementTimeValues, collapse = ", "),
      "p" = p,
      "nb_sites_occupied" = sum(z_i),
      "nb_detec_total" = sum(RecapNDetec %>% pull(starts_with("Deployment"))),
      "nb_detec_average_per_deployment_when_occupied" = mean(RecapNDetec %>% filter(Occupied == 1) %>% pull(starts_with("Deployment"))),
      "detec_proba_data_deploy1" = round(
        (NbSites - nb_sites_occupied_no_detec_deploy1 / sum(z_i == 1) * 100) / 100,
        2),
      "Time_Simulation"=Time_Simulation,
      "optim_method" = optim_method
    ),
    
    "twoMMPP" = vector(mode = "list", length = 6) %>%
      set_names(c(
        "psi", "lambda_1", "lambda_2", "mu_12", "mu_21", "fitting_time"
      )),
    
    "IPP" = vector(mode = "list", length = 6) %>%
      set_names(c(
        "psi", "lambda_1", "lambda_2", "mu_12", "mu_21", "fitting_time"
      )),
    
    "PP" = vector(mode = "list", length = 3) %>%
      set_names(c("psi", "lambda", "fitting_time")),
    
    "COP" = vector(mode = "list", length = 3) %>%
      set_names(c("psi", "lambda_Tsession", "fitting_time")),
    
    "BP" = vector(mode = "list", length = 3) %>%
      set_names(c("psi", "p", "fitting_time"))
  )
  
  
  # ──────────────────────────────────────────────────────────────────────────────
  # IF 0 DETECTION                                                            ----
  # ──────────────────────────────────────────────────────────────────────────────
  
  # In this case, we can not fit the models. We just export the informations
  # to keep track of this simulation.
  
  if (sum(RecapNDetec %>% pull(starts_with("Deployment"))) == 0) {
    cat("\n🛑️ Zero detections in this simulation. No model will be fitted.\n")
    
    # if result file exists, we modify it so that we can add the new result.
    if (!is.null(ComparisonResult_ExportPath)) {
      if (file.exists(ComparisonResult_ExportPath)) {
        # If last line of file is "]", we remove the last line ']' from the file
        if (system(glue::glue("tail -n 1 {ComparisonResult_ExportPath}"),
                   intern = T) == "]") {
          system(glue::glue("sed -i '$ d' {ComparisonResult_ExportPath}"))
        }
        
        # If the last character of file isn't "," ; we add "," to the end
        lastline = system(glue::glue("tail -n 1 {ComparisonResult_ExportPath}"),
                          intern = T)
        if (substr(lastline, nchar(lastline), nchar(lastline)) != ",") {
          system(glue::glue("sed -i '$ d' {ComparisonResult_ExportPath}"))
          cat(paste0(lastline, ","),
              file = ComparisonResult_ExportPath,
              append = T)
        }
        
        # Add new result
        jsonlite::toJSON(ComparisonResult, pretty = T) %>%
          as.character() %>%
          gsub(
            pattern = "[\n",
            replacement = "",
            x = .,
            fixed = T
          ) %>%
          cat(file = ComparisonResult_ExportPath, append = T)
        
      } else {
        # Add first result
        jsonlite::toJSON(ComparisonResult, pretty = T) %>%
          as.character() %>%
          cat(file = ComparisonResult_ExportPath, append = T)
      }
    }
    
    return(ComparisonResult[[1]])
  }
  
  
  
  # ──────────────────────────────────────────────────────────────────────────────
  # 2-MMPP OCCUPANCY MODEL                                                    ----
  # ──────────────────────────────────────────────────────────────────────────────
  if (run_continuous_models) {
    
  cat(" -- 2-MMPP")

  ### Plot L given psi -----------------------------------------------------------
  if (!quiet) {
    if (!exists("log_llh_psi_data")) {
      log_llh_psi_data = data.frame("psi" = seq(0.05, 0.95, by = 0.05))
    }
    
    pb = newpb(nrow(log_llh_psi_data))
    for (i in 1:nrow(log_llh_psi_data)) {
      log_llh_psi_data$loglikelihood_2MMPP[i] <- get_2MMPP_loglikelihood(
        param = c(
          'psi' = logit(log_llh_psi_data$psi[i]),
          'lambda_1' = log(lambda[1]),
          'lambda_2' = log(lambda[2]),
          "mu_12" = log(mu[1]),
          "mu_21" = log(mu[2])
        ),
        detection_times = detection_times,
        NbSites = NbSites,
        list_T_ij = list_T_ij,
        list_R_i = list_R_i,
        quick = TRUE
      )
      pb$tick()
    }
    
    psi_for_maxllh = log_llh_psi_data[which.max(log_llh_psi_data$loglikelihood_2MMPP), "psi"]
    middle_y = mean(c(
      min(log_llh_psi_data$loglikelihood_2MMPP[!is.infinite(log_llh_psi_data$loglikelihood_2MMPP)]),
      max(log_llh_psi_data$loglikelihood_2MMPP[!is.infinite(log_llh_psi_data$loglikelihood_2MMPP)])
    ))
    ggplot(data = log_llh_psi_data, aes(x = psi, y = loglikelihood_2MMPP)) +
      geom_line() +
      geom_vline(
        xintercept = psi_for_maxllh,
        linetype = "dotted",
        colour = "cadetblue"
      ) +
      geom_label(
        aes(
          x = psi_for_maxllh,
          y = middle_y,
          label = glue::glue("ψ = {psi_for_maxllh}")
        ),
        fill = "cadetblue",
        colour = "white"
      ) +
      geom_hline(
        yintercept = max(log_llh_psi_data$loglikelihood_2MMPP),
        linetype = "dotted",
        colour = "coral4"
      ) +
      geom_label(
        aes(
          x = 0.1,
          y = max(loglikelihood_2MMPP),
          label = glue::glue(
            "log-likelihood = {round(max(log_llh_psi_data$loglikelihood_2MMPP),2)}"
          )
        ),
        fill = "coral4",
        colour = "white"
      ) +
      labs(
        title = "Log-likelihood depending on ψ the occupancy probability",
        subtitle = "With fixed λ and μ values used for the simulation",
        caption = glue::glue(
          "Simulations values:\n",
          "[SIMULATED OCCUPANCY]\tψ = {round(psi,2)}\n",
          "[DETECTION}\tλ = ({round(lambda[1],2)}, {round(lambda[2],2)}) ; ",
          "μ = ({round(mu[1],2)}, {round(mu[2],2)}) -> ",
          "p = {round(p, 3)}, ExpN = {round(Expected_N,2)}\n",
          "{NbSites} sites with {NbDeployPerSite} deployment(s) of {DeployementTimeValues} days"
        )
      )
  }
  
  ### Parameter estimation -------------------------------------------------------
  
  
  # Initialisation with truth parameters to reduce calculation time
  beforetime = Sys.time()
  fitted_2MMPP <- try(optim(
    # Initial parameters
    par = c(
      'psi' = logit(psi),
      'lambda_1' = log(ifelse(lambda[1] == 0, 1e-20, lambda[1])),
      'lambda_2' = log(ifelse(lambda[2] == 0, 1e-20, lambda[2])),
      "mu_12" = log(ifelse(mu[1] == 0, 1e-20, mu[1])),
      "mu_21" = log(ifelse(mu[2] == 0, 1e-20, mu[2]))
    ),
    # Function to optimize
    fn = get_2MMPP_neg_loglikelihood,
    # Optim parameters
    method = optim_method,
    control = list(
      maxit = 1000,
      trace = !quiet,
      REPORT = 5
    ),
    hessian = T,
    # Other parameters of get_likelihood
    detection_times = detection_times,
    NbSites = length(SimulatedDataset),
    list_T_ij = list_T_ij,
    list_R_i = list_R_i,
    quick = FALSE,
    debug_print = FALSE
  ))
  aftertime = Sys.time()
  Time_2MMPP = as.numeric(aftertime - beforetime, units = "secs")
  
  
  if (inherits(fitted_2MMPP, "try-error")) {
    print("ERROR")
    print(fitted_2MMPP)
    print("Trying with new initial conditions")
    
    
    if (!exists("log_llh_psi_data")) {
      log_llh_psi_data = data.frame("psi" = seq(0.05, 0.95, by = 0.05))
      
      for (i in 1:nrow(log_llh_psi_data)) {
        log_llh_psi_data$loglikelihood_2MMPP[i] <-
          get_2MMPP_loglikelihood(
            param = c(
              'psi' = logit(log_llh_psi_data$psi[i]),
              'lambda_1' = log(lambda[1]),
              'lambda_2' = log(lambda[2]),
              "mu_12" = log(mu[1]),
              "mu_21" = log(mu[2])
            ),
            detection_times = detection_times,
            NbSites = NbSites,
            list_T_ij = list_T_ij,
            list_R_i = list_R_i,
            quick = TRUE
          )
      }
      
      psi_for_maxllh = log_llh_psi_data[which.max(log_llh_psi_data$loglikelihood_2MMPP), "psi"]
    }
    
    
    beforetime = Sys.time()
    fitted_2MMPP <- try(optim(
      # Initial parameters
      par = c(
        'psi' = logit(psi_for_maxllh),
        'lambda_1' = log(ifelse(lambda[1] == 0, 1e-20, lambda[1])),
        'lambda_2' = log(ifelse(lambda[2] == 0, 1e-20, lambda[2])),
        "mu_12" = log(ifelse(mu[1] == 0, 1e-20, mu[1])),
        "mu_21" = log(ifelse(mu[2] == 0, 1e-20, mu[2]))
      ),
      # Function to optimize
      fn = get_2MMPP_neg_loglikelihood,
      # Optim parameters
      method = optim_method,
      control = list(
        maxit = 1000,
        trace = !quiet,
        REPORT = 5
      ),
      hessian = T,
      # Other parameters of get_likelihood
      detection_times = detection_times,
      NbSites = length(SimulatedDataset),
      list_T_ij = list_T_ij,
      list_R_i = list_R_i,
      quick = FALSE,
      debug_print = FALSE
    ))
    aftertime = Sys.time()
    Time_2MMPP = as.numeric(aftertime - beforetime, units = "secs")
    
    if (inherits(fitted_2MMPP, "try-error")) {
      print("ERROR")
      print(fitted_2MMPP)
      
      if (!is.null(log_ExportPath)) {
        paste0(
          deparse(ComparisonResult[[1]][["param"]]) %>%
            paste(collapse = "") %>%
            str_replace_all('"', "'") %>%
            gsub(
              pattern = "\\s+",
              replacement = " ",
              x = .
            ),
          "\n"
        ) %>%
          cat(file = log_ExportPath, append = T)
      }
      return()
    }
  }
  
  # Adding results to the result list
  ComparisonResult[[1]]$twoMMPP$psi <- unname(invlogit(fitted_2MMPP$par[1]))
  ComparisonResult[[1]]$twoMMPP$lambda_1 <- unname(exp(fitted_2MMPP$par[2]))
  ComparisonResult[[1]]$twoMMPP$lambda_2 <- unname(exp(fitted_2MMPP$par[3]))
  ComparisonResult[[1]]$twoMMPP$mu_12 <- unname(exp(fitted_2MMPP$par[4]))
  ComparisonResult[[1]]$twoMMPP$mu_21 <- unname(exp(fitted_2MMPP$par[5]))
  ComparisonResult[[1]]$twoMMPP$fitting_time <- Time_2MMPP
  
  # Calculation of 95% confidence interval
  ComparisonResult[[1]]$twoMMPP$hessian <- deparse(fitted_2MMPP$hessian) %>% 
    paste(collapse = " ") %>%
    str_replace_all(pattern = '"', replacement = "'")
  if (solvable(fitted_2MMPP$hessian)) {
    fisher_info <- solve(fitted_2MMPP$hessian)
    if (any(diag(fisher_info) < 0)) {
      se <- sqrt(diag(fisher_info) + 0i)
    } else{
      se <- sqrt(diag(fisher_info))
    }
    upper95CI_fitted_2MMPP <- fitted_2MMPP$par + 1.96 * se
    lower95CI_fitted_2MMPP <- fitted_2MMPP$par - 1.96 * se
    
    # Adding results
    ComparisonResult[[1]]$twoMMPP$psi_lower_95CI <- unname(invlogit(lower95CI_fitted_2MMPP['psi']))
    ComparisonResult[[1]]$twoMMPP$psi_upper_95CI <- unname(invlogit(upper95CI_fitted_2MMPP['psi']))
    
    ComparisonResult[[1]]$twoMMPP$lambda_1_lower_95CI <-unname(exp(lower95CI_fitted_2MMPP[2]))
    ComparisonResult[[1]]$twoMMPP$lambda_1_upper_95CI <-unname(exp(upper95CI_fitted_2MMPP[2]))
    
    ComparisonResult[[1]]$twoMMPP$lambda_2_lower_95CI <-unname(exp(lower95CI_fitted_2MMPP[3]))
    ComparisonResult[[1]]$twoMMPP$lambda_2_upper_95CI <-unname(exp(upper95CI_fitted_2MMPP[3]))
    
    ComparisonResult[[1]]$twoMMPP$mu_12_lower_95CI <-unname(exp(lower95CI_fitted_2MMPP[4]))
    ComparisonResult[[1]]$twoMMPP$mu_12_upper_95CI <-unname(exp(upper95CI_fitted_2MMPP[4]))
    
    ComparisonResult[[1]]$twoMMPP$mu_21_lower_95CI <-unname(exp(lower95CI_fitted_2MMPP[5]))
    ComparisonResult[[1]]$twoMMPP$mu_21_upper_95CI <-unname(exp(upper95CI_fitted_2MMPP[5]))
    
  }
  
  # Printing results
  if (!quiet) {
    data.frame(
      "parameter" = c("psi", "lambda_1", "lambda_2", "mu_12", "mu_21", "p"),
      "truth" = c(psi, lambda[1], lambda[2], mu[1], mu[2], p),
      "estimated" = c(invlogit(fitted_2MMPP$par[1]), 
                      exp(fitted_2MMPP$par[2:5]),
                      get_p_from_2MMPP_param(
                        lambda = exp(fitted_2MMPP$par[2:3]),
                        mu = exp(fitted_2MMPP$par[4:5]),
                        pT = DeployementTimeValues
                      ))
      ) %>%
      as_tibble() %>%
      mutate_if(is.numeric, round, 5) %>% 
      print()
  }
  
  # ──────────────────────────────────────────────────────────────────────────────
  # IPP OCCUPANCY MODEL                                                       ----
  # ──────────────────────────────────────────────────────────────────────────────
  
  cat(" -- IPP")

  ### Plot L given psi -----------------------------------------------------------
  if (!quiet) {
    if (!exists("log_llh_psi_data")) {
      log_llh_psi_data = data.frame("psi" = seq(0.05, 0.95, by = 0.05))
    }
    
    pb = newpb(nrow(log_llh_psi_data))
    for (i in 1:nrow(log_llh_psi_data)) {
      log_llh_psi_data$loglikelihood_IPP[i] <- get_IPP_loglikelihood(
        param = c(
          'psi' = logit(log_llh_psi_data$psi[i]),
          'lambda_2' = log(lambda[2]),
          "mu_12" = log(mu[1]),
          "mu_21" = log(mu[2])
        ),
        detection_times = detection_times,
        NbSites = NbSites,
        list_T_ij = list_T_ij,
        list_R_i = list_R_i,
        quick = TRUE
      )
      pb$tick()
    }
    
    psi_for_maxllh = log_llh_psi_data[which.max(log_llh_psi_data$loglikelihood_IPP), "psi"]
    middle_y = mean(c(
      min(log_llh_psi_data$loglikelihood_IPP[!is.infinite(log_llh_psi_data$loglikelihood_IPP)]),
      max(log_llh_psi_data$loglikelihood_IPP[!is.infinite(log_llh_psi_data$loglikelihood_IPP)])
    ))
    ggplot(data = log_llh_psi_data, aes(x = psi, y = loglikelihood_IPP)) +
      geom_line() +
      geom_vline(
        xintercept = psi_for_maxllh,
        linetype = "dotted",
        colour = "cadetblue"
      ) +
      geom_label(
        aes(
          x = psi_for_maxllh,
          y = middle_y,
          label = glue::glue("ψ = {psi_for_maxllh}")
        ),
        fill = "cadetblue",
        colour = "white"
      ) +
      geom_hline(
        yintercept = max(log_llh_psi_data$loglikelihood_IPP),
        linetype = "dotted",
        colour = "coral4"
      ) +
      geom_label(
        aes(
          x = 0.1,
          y = max(loglikelihood_IPP),
          label = glue::glue(
            "log-likelihood = {round(max(log_llh_psi_data$loglikelihood_IPP),2)}"
          )
        ),
        fill = "coral4",
        colour = "white"
      ) +
      labs(
        title = "Log-likelihood depending on ψ the occupancy probability",
        subtitle = "With fixed λ and μ values used for the simulation",
        caption = glue::glue(
          "Simulations values:\n",
          "ψ = {round(psi,2)}\n",
          "λ = ({round(lambda[1],2)}, {round(lambda[2],2)}) ; ",
          "μ = ({round(mu[1],2)}, {round(mu[2],2)}) -> ",
          "p = {round(p, 2)}\n",
          "{NbSites} sites with {NbDeployPerSite} deployment(s) of {DeployementTimeValues} days"
        )
      )
  }
  
  
  ### Parameter estimation -------------------------------------------------------
  
  # Initialisation with truth parameters to reduce calculation time
  beforetime = Sys.time()
  fitted_IPP <- try(optim(
    # Initial parameters
    par = c(
      'psi' = logit(psi),
      'lambda_2' = log(ifelse(max(lambda) == 0, 1e-20, max(lambda))),
      "mu_12" = log(ifelse(mu[1] == 0, 1e-20, mu[1])),
      "mu_21" = log(ifelse(mu[2] == 0, 1e-20, mu[2]))
    ),
    # Function to optimize
    fn = get_IPP_neg_loglikelihood,
    # Optim parameters
    method = optim_method,
    control = list(
      maxit = 1000,
      trace = !quiet,
      REPORT = 5
    ),
    hessian = TRUE,
    # Other parameters of get_likelihood
    detection_times = detection_times,
    NbSites = length(SimulatedDataset),
    list_T_ij = list_T_ij,
    list_R_i = list_R_i,
    quick = FALSE
  ))
  aftertime = Sys.time()
  Time_IPP = as.numeric(aftertime - beforetime, units = "secs")
  
  if(inherits(fitted_IPP, "try-error")){
    print("ERROR")
    print(fitted_IPP)
    if (!is.null(log_ExportPath)) {
      paste0(
        deparse(ComparisonResult[[1]][["param"]]) %>%
          paste(collapse = "") %>%
          str_replace_all('"', "'") %>%
          gsub(
            pattern = "\\s+",
            replacement = " ",
            x = .
          ),
        "\n"
      ) %>%
        cat(file = log_ExportPath, append = T)
    }
    return()
  }
  
  # Adding results to the result list
  ComparisonResult[[1]]$IPP$psi <- unname(invlogit(fitted_IPP$par[1]))
  ComparisonResult[[1]]$IPP$lambda_1 <- 0
  ComparisonResult[[1]]$IPP$lambda_2 <- unname(exp(fitted_IPP$par[2]))
  ComparisonResult[[1]]$IPP$mu_12 <- unname(exp(fitted_IPP$par[3]))
  ComparisonResult[[1]]$IPP$mu_21 <- unname(exp(fitted_IPP$par[4]))
  ComparisonResult[[1]]$IPP$fitting_time <- Time_IPP
  
  
  # Calculation of 95% confidence interval
  # Code adapted from: https://stats.stackexchange.com/a/27133
  ComparisonResult[[1]]$IPP$hessian <- deparse(fitted_IPP$hessian) %>% 
    paste(collapse = " ") %>%
    str_replace_all(pattern = '"', replacement = "'")
  if (solvable(fitted_IPP$hessian)) {
    fisher_info <- solve(fitted_IPP$hessian)
    if (any(diag(fisher_info) < 0)) {
      se <- sqrt(diag(fisher_info) + 0i)
    } else{
      se <- sqrt(diag(fisher_info))
    }
    upper95CI_fitted_IPP <- fitted_IPP$par + 1.96 * se
    lower95CI_fitted_IPP <- fitted_IPP$par - 1.96 * se
    
    # Adding results
    ComparisonResult[[1]]$IPP$psi_lower_95CI <- unname(invlogit(lower95CI_fitted_IPP['psi']))
    ComparisonResult[[1]]$IPP$psi_upper_95CI <- unname(invlogit(upper95CI_fitted_IPP['psi']))
    
    ComparisonResult[[1]]$IPP$lambda_2_lower_95CI <-unname(exp(lower95CI_fitted_IPP[2]))
    ComparisonResult[[1]]$IPP$lambda_2_upper_95CI <-unname(exp(upper95CI_fitted_IPP[2]))
    
    ComparisonResult[[1]]$IPP$mu_12_lower_95CI <-unname(exp(lower95CI_fitted_IPP[3]))
    ComparisonResult[[1]]$IPP$mu_12_upper_95CI <-unname(exp(upper95CI_fitted_IPP[3]))
    
    ComparisonResult[[1]]$IPP$mu_21_lower_95CI <-unname(exp(lower95CI_fitted_IPP[4]))
    ComparisonResult[[1]]$IPP$mu_21_upper_95CI <-unname(exp(upper95CI_fitted_IPP[4]))
    
  }
  
  # Printing results
  if (!quiet) {
    data.frame(
      "parameter" = c("psi", "lambda_1", "lambda_2", "mu_12", "mu_21", "p"),
      "truth" = c(psi, lambda[1], lambda[2], mu[1], mu[2], p),
      "estimated" = c(
        invlogit(fitted_IPP$par[1]),
        0,
        exp(fitted_IPP$par[2:4]),
        get_p_from_2MMPP_param(
          lambda = c(exp(fitted_IPP$par[2]), 0),
          mu = exp(fitted_IPP$par[3:4]),
          pT = DeployementTimeValues
        )
      )
    ) %>%
      as_tibble() %>%
      mutate_if(is.numeric, round, 5) %>% 
      print()
    
  }
  
  
  
  
  # ──────────────────────────────────────────────────────────────────────────────
  # PP OCCUPANCY MODEL                                                       ----
  # ──────────────────────────────────────────────────────────────────────────────
  
  cat(" -- PP")
  
  ### Plot L given psi -----------------------------------------------------------
  if (!quiet) {
    if (!exists("log_llh_psi_data")) {
      log_llh_psi_data = data.frame("psi" = seq(0.05, 0.95, by = 0.05))
    }
    
    for (i in 1:nrow(log_llh_psi_data)) {
      log_llh_psi_data$loglikelihood_PP[i] <- get_PP_loglikelihood(
        param = c(
          'psi' = logit(log_llh_psi_data$psi[i]),
          'lambda' = log(lambda_PP)
        ),
        detection_times = detection_times,
        NbSites = NbSites,
        list_T_ij = list_T_ij,
        list_R_i = list_R_i
      )
    }

    psi_for_maxllh = log_llh_psi_data[which.max(log_llh_psi_data$loglikelihood_PP), "psi"]
    middle_y = mean(c(
      min(log_llh_psi_data$loglikelihood_PP[!is.infinite(log_llh_psi_data$loglikelihood_PP)]),
      max(log_llh_psi_data$loglikelihood_PP[!is.infinite(log_llh_psi_data$loglikelihood_PP)])
    ))
    ggplot(data = log_llh_psi_data, aes(x = psi, y = loglikelihood_PP)) +
      geom_line() +
      geom_vline(
        xintercept = psi_for_maxllh,
        linetype = "dotted",
        colour = "cadetblue"
      ) +
      geom_label(
        aes(
          x = psi_for_maxllh,
          y = middle_y,
          label = glue::glue("ψ = {psi_for_maxllh}")
        ),
        fill = "cadetblue",
        colour = "white"
      ) +
      geom_hline(
        yintercept = max(log_llh_psi_data$loglikelihood_PP),
        linetype = "dotted",
        colour = "coral4"
      ) +
      geom_label(
        aes(
          x = 0.1,
          y = max(loglikelihood_PP),
          label = glue::glue(
            "log-likelihood = {round(max(log_llh_psi_data$loglikelihood_PP),2)}"
          )
        ),
        fill = "coral4",
        colour = "white"
      ) +
      labs(
        title = "Log-likelihood depending on ψ the occupancy probability",
        subtitle = "With λ estimated from 2-MMPP λ and μ values used for the simulation",
        caption = glue::glue(
          "Simulations values:\n",
          "ψ = {round(psi,2)}\n",
          "λ = ({round(lambda[1],2)}, {round(lambda[2],2)}) ; ",
          "μ = ({round(mu[1],2)}, {round(mu[2],2)}) -> ",
          "p = {round(p, 2)}\n",
          "λ_PP ≃ {round(lambda_PP,2)}\n",
          "{NbSites} sites with {NbDeployPerSite} deployment(s) of {DeployementTimeValues} days"
        )
      )
  }
  
  ### Parameter estimation -------------------------------------------------------
  
  # Initialisation with true parameter for psi to reduce calculation time
  # lambda starts at equivalent of detecting half the animals
  beforetime = Sys.time()
  fitted_PP <- optim(
    # Initial parameters
    par = c(
      'psi' = logit(psi),
      'lambda' = log(lambda_PP)
    ),
    # Function to optimize
    fn = get_PP_neg_loglikelihood,
    # Optim parameters
    method = optim_method,
    control = list(
      maxit = 1000,
      trace = !quiet,
      REPORT = 5
    ),
    # Other parameters of get_likelihood
    detection_times = detection_times,
    NbSites = length(SimulatedDataset),
    list_T_ij = list_T_ij,
    list_R_i = list_R_i,
    hessian=T
  )
  aftertime = Sys.time()
  Time_PP = as.numeric(aftertime - beforetime, units = "secs")
  
  # Calculation of 95% confidence interval
  ComparisonResult[[1]]$PP$hessian <- deparse(fitted_PP$hessian) %>%
    paste(collapse = " ") %>%
    str_replace_all(pattern = '"', replacement = "'")
  if (solvable(fitted_PP$hessian)) {
    fisher_info <- solve(fitted_PP$hessian)
    if (any(diag(fisher_info) < 0)) {
      se <- sqrt(diag(fisher_info) + 0i)
    } else{
      se <- sqrt(diag(fisher_info))
    }
    upper95CI_fitted_PP <- fitted_PP$par + 1.96 * se
    lower95CI_fitted_PP <- fitted_PP$par - 1.96 * se
    ComparisonResult[[1]]$PP$psi_lower_95CI <- unname(invlogit(lower95CI_fitted_PP['psi']))
    ComparisonResult[[1]]$PP$psi_upper_95CI <- unname(invlogit(upper95CI_fitted_PP['psi']))
    
    ComparisonResult[[1]]$PP$lambda_lower_95CI <- unname(exp(lower95CI_fitted_PP['lambda']))
    ComparisonResult[[1]]$PP$lambda_upper_95CI <- unname(exp(upper95CI_fitted_PP['lambda']))
    
  }
  # Adding results to the result list
  ## psi
  ComparisonResult[[1]]$PP$psi <- unname(invlogit(fitted_PP$par[1]))
  ## lambda
  ComparisonResult[[1]]$PP$lambda <- unname(exp(fitted_PP$par[2]))
  ## Calculation time
  ComparisonResult[[1]]$PP$fitting_time <- Time_PP
  
  
  # Printing results
  if (!quiet) {
    data.frame(
      "parameter" = c("psi", "lambda", "p"),
      "truth" = c(psi, lambda_PP, p),
      "estimated" = c(
        invlogit(fitted_PP$par[1]),
        exp(fitted_PP$par[2]),
        get_p_from_PP_param(
          lambda = exp(fitted_PP$par[2]),
          pT = DeployementTimeValues
        )
      )
    ) %>%
      as_tibble() %>%
      mutate_if(is.numeric, round, 5) %>%
      print()
  }
  
  }
  
  
  if (run_discrete_models) {
  # ──────────────────────────────────────────────────────────────────────────────
  # COP OCCUPANCY MODEL                                           ----
  # ──────────────────────────────────────────────────────────────────────────────
  
  cat(" -- COP")

  ### Discretisation -------------------------------------------------------------
  SimulatedDataset_NbDetecsPerSession = get_nbdetec_per_session(
    SimulatedDataset = SimulatedDataset,
    SessionLength = SessionLength,
    list_R_i = list_R_i, 
    list_T_ij = list_T_ij
  )
  
  if (!quiet) {
    ### Plot L given psi & lambda --------------------------------------------------
    
    test = expand.grid(round(seq(0, 5, length.out = 25), 2),
                       round(seq(0, 1, length.out = 15), 2)) %>%
      setNames(., c("lambda", "psi")) %>%
      as_tibble() %>%
      mutate(
        'psi_transformed' = logit(psi),
        'lambda_transformed' = log(lambda)
      )
    
    
    for (k in 1:nrow(test)) {
      test[k, "loglikelihood"] = get_COP_loglikelihood(
        param = c(
          "psi_transformed" = test$psi_transformed[k],
          "lambda_transformed" = test$lambda_transformed[k]
        ),
        SimulatedDataset_NbDetecsPerSession = SimulatedDataset_NbDetecsPerSession
      )
    }
    
    ### Plot L given psi -----------------------------------------------------------
    
    test = expand.grid(1.9,
                       round(seq(0.05, 0.95, length.out = 50), 2)) %>%
      setNames(., c("lambda", "psi")) %>%
      as_tibble() %>%
      mutate(
        'psi_transformed' = logit(psi),
        'lambda_transformed' = log(lambda)
      )
    
    
    for (k in 1:nrow(test)) {
      test[k, "loglikelihood"] = get_COP_loglikelihood(
        param = c(
          "psi_transformed" = test$psi_transformed[k],
          "lambda_transformed" = test$lambda_transformed[k]
        ),
        SimulatedDataset_NbDetecsPerSession = SimulatedDataset_NbDetecsPerSession
      )
    }
    
    resgg = test %>%
      mutate(lambda = factor(lambda)) %>%
      ggplot(data = .) +
      geom_line(aes(x = psi, y = loglikelihood, col = lambda))
    print(resgg)
    
    
    ### Plot L given lambda --------------------------------------------------------
    test = expand.grid(seq(1, 5, length.out = 50), 
                       0.75) %>%
      setNames(., c("lambda", "psi")) %>%
      as_tibble() %>%
      mutate(
        'psi_transformed' = logit(psi),
        'lambda_transformed' = log(lambda)
      )
    
    for (k in 1:nrow(test)) {
      test[k, "loglikelihood"] = get_COP_loglikelihood(
        param = c(
          "psi_transformed" = test$psi_transformed[k],
          "lambda_transformed" = test$lambda_transformed[k]
        ),
        SimulatedDataset_NbDetecsPerSession = SimulatedDataset_NbDetecsPerSession
      )
    }
    
    resgg = test %>%
      mutate(psi = factor(psi)) %>%
      ggplot(data = .) +
      geom_line(aes(x = lambda, y = loglikelihood, col = psi))
    print(resgg)
  }
  
  ### Parameter estimation -------------------------------------------------------
  beforetime = Sys.time()
  fitted_COP <- try(optim(
    # Initial parameters
    par = c('psi' = logit(psi),
            'lambda' = log(lambda_PP)),
    # Function to optimize
    fn = get_COP_neg_loglikelihood,
    # Optim parameters
    method = optim_method,
    control = list(
      maxit = 1000,
      trace = !quiet,
      REPORT = 5
    ),
    # Other parameters of get_likelihood
    SimulatedDataset_NbDetecsPerSession = SimulatedDataset_NbDetecsPerSession,
    hessian=T
  ))
  aftertime = Sys.time()
  Time_COP = as.numeric(aftertime - beforetime, units = "secs")
  
  if(inherits(fitted_COP, "try-error")){
    print("ERROR")
    print(fitted_COP)
    if (!is.null(log_ExportPath)) {
      paste0(
        deparse(ComparisonResult[[1]][["param"]]) %>%
          paste(collapse = "") %>%
          str_replace_all('"', "'") %>%
          gsub(
            pattern = "\\s+",
            replacement = " ",
            x = .
          ),
        "\n"
      ) %>%
        cat(file = log_ExportPath, append = T)
    }
    return()
  }
  
  # Adding results to the result list
  ComparisonResult[[1]]$COP$psi <- unname(invlogit(fitted_COP$par[1]))
  ComparisonResult[[1]]$COP$lambda_Tsession <- unname(exp(fitted_COP$par[2]))
  ComparisonResult[[1]]$COP$fitting_time <- Time_COP
  
  # Calculation of 95% confidence interval
  # Code adapted from: https://stats.stackexchange.com/a/27133
  ComparisonResult[[1]]$COP$hessian <- deparse(fitted_COP$hessian) %>% 
    paste(collapse = " ") %>%
    str_replace_all(pattern = '"', replacement = "'")
  if (solvable(fitted_COP$hessian)) {
    fisher_info <- solve(fitted_COP$hessian)
    se <- sqrt(diag(fisher_info))
    upper95CI_fitted_COP <- fitted_COP$par + 1.96 * se
    lower95CI_fitted_COP <- fitted_COP$par - 1.96 * se
    
    # Adding results
    ComparisonResult[[1]]$COP$psi_lower_95CI <- unname(invlogit(lower95CI_fitted_COP['psi']))
    ComparisonResult[[1]]$COP$psi_upper_95CI <- unname(invlogit(upper95CI_fitted_COP['psi']))
    
    ComparisonResult[[1]]$COP$lambda_Tsession_lower_95CI <-unname(exp(lower95CI_fitted_COP[2]))
    ComparisonResult[[1]]$COP$lambda_Tsession_upper_95CI <-unname(exp(upper95CI_fitted_COP[2]))
  }
  
  # Printing results
  if (!quiet) {
    data.frame(
      "parameter" = c("psi", "lambda*NbSites", "p"),
      "truth" = c(psi, lambda_PP * SessionLength, p),
      "estimated" = c(
        invlogit(fitted_COP$par[1]),
        exp(fitted_COP$par[2]),
        get_p_from_COP_param(
          lambda = exp(fitted_COP$par[2]),
          pT = DeployementTimeValues,
          SessionLength = SessionLength,
          NbDeployPerSite = NbDeployPerSite
        )
      )
    ) %>%
      as_tibble() %>%
      mutate_if(is.numeric, round, 5) %>% 
      print()
  }
  
  # ──────────────────────────────────────────────────────────────────────────────
  # BP OCCUPANCY MODEL                                                  ----
  # ──────────────────────────────────────────────────────────────────────────────
  
  cat(" -- BP")

  # Discretisation
  SimulatedDataset_DetectedPerDay = get_detected_per_session(SimulatedDataset_NbDetecsPerSession)
  
  if (!quiet) {
    plot(unmarked::unmarkedFrameOccu(SimulatedDataset_DetectedPerDay))
  }
  
  # Tests occupancy classique w/ unmarked
  beforetime = Sys.time()
  occu_model = try(unmarked::occu(
    formula =  ~ 1 ~ 1,
    data = unmarkedFrameOccu(SimulatedDataset_DetectedPerDay),
    starts = c('psi' = logit(psi), 'p' = logit(p_session)),
    method = optim_method,
    control = list(
      maxit = 1000,
      trace = !quiet,
      REPORT = 5
    )
  ))
  aftertime = Sys.time()
  Time_BP = as.numeric(aftertime - beforetime, units = "secs")
  
  if(inherits(occu_model, "try-error")){
    print("ERROR")
    print(occu_model)
    if (!is.null(log_ExportPath)) {
      paste0(
        deparse(ComparisonResult[[1]][["param"]]) %>%
          paste(collapse = "") %>%
          str_replace_all('"', "'") %>%
          gsub(
            pattern = "\\s+",
            replacement = " ",
            x = .
          ),
        "\n"
      ) %>%
        cat(file = log_ExportPath, append = T)
    }
    return()
  }
  
  
  if (!quiet) {
    print(occu_model)
    cat("\n\n--------------------------\nOccupancy probability:\n")
    print(backTransform(occu_model, type = "state")) # Occupancy probability
    cat("\n\n--------------------------\nDetection probability:\n")
    print(backTransform(occu_model, type = "det")) # Detection probability
  }
  
  # Adding results to the result list
  ComparisonResult[[1]]$BP$psi <- backTransform(occu_model, type = "state")@estimate
  ComparisonResult[[1]]$BP$p <- backTransform(occu_model, type = "det")@estimate
  ComparisonResult[[1]]$BP$fitting_time <- Time_BP
  
  # Calculation of 95% confidence interval
  # Code adapted from: https://stats.stackexchange.com/a/27133
  ComparisonResult[[1]]$BP$hessian <- deparse(hessian(occu_model)) %>% 
    paste(collapse = " ") %>%
    str_replace_all(pattern = '"', replacement = "'")
  if (solvable(hessian(occu_model))) {
    fisher_info <- solve(hessian(occu_model))
    se <- sqrt(diag(fisher_info))
    BP_fitted_par = c(
      'psi' = unname(occu_model@estimates@estimates$state@estimates),
      'p' =  unname(occu_model@estimates@estimates$det@estimates)
    )
    upper95CI_fitted_BP <- BP_fitted_par + 1.96 * se
    lower95CI_fitted_BP <- BP_fitted_par - 1.96 * se
    
    # Adding results
    ComparisonResult[[1]]$BP$psi_lower_95CI <- unname(invlogit(lower95CI_fitted_BP['psi']))
    ComparisonResult[[1]]$BP$psi_upper_95CI <- unname(invlogit(upper95CI_fitted_BP['psi']))
    
    ComparisonResult[[1]]$BP$p_lower_95CI <-unname(invlogit(lower95CI_fitted_BP[2]))
    ComparisonResult[[1]]$BP$p_upper_95CI <-unname(invlogit(upper95CI_fitted_BP[2]))
  }
  
  }

  # ──────────────────────────────────────────────────────────────────────────────
  # RESULT LIST EXPORT                                                        ----
  # ──────────────────────────────────────────────────────────────────────────────
  
  cat(" --\n")
  # Add the total time
  endtime = Sys.time()
  Time_Total = as.numeric(endtime - begintime, units = "secs")
  ComparisonResult[[1]]$total_time <- Time_Total
  
  if (!quiet) {
    print(ComparisonResult)
  }
  
  # if result file exists, we modify it so that we can add the new result.
  if (!is.null(ComparisonResult_ExportPath)) {
    if (file.exists(ComparisonResult_ExportPath) & file.size(ComparisonResult_ExportPath) > 0) {
      # If last line of file is "]", we remove the last line ']' from the file
      if (system(glue::glue("tail -n 1 {ComparisonResult_ExportPath}"),
                 intern = T) == "]") {
        system(glue::glue("sed -i '$ d' {ComparisonResult_ExportPath}"))
      }
      
      # If the last character of file isn't "," ; we add "," to the end
      lastline = system(glue::glue("tail -n 1 {ComparisonResult_ExportPath}"),
                        intern = T)
      if (substr(lastline, nchar(lastline), nchar(lastline)) != ",") {
        system(glue::glue("sed -i '$ d' {ComparisonResult_ExportPath}"))
        cat(paste0(lastline, ","),
            file = ComparisonResult_ExportPath,
            append = T)
      }
      
      # Add new result
      jsonlite::toJSON(ComparisonResult, pretty = T, digits = 10) %>%
        as.character() %>%
        gsub(
          pattern = "[\n",
          replacement = "",
          x = .,
          fixed = T
        ) %>%
        cat(file = ComparisonResult_ExportPath, append = T)
      
    } else {
      # Add first result
      jsonlite::toJSON(ComparisonResult, pretty = T, digits = 10) %>%
        as.character() %>%
        cat(file = ComparisonResult_ExportPath, append = T)
    }
  }
  
  
  return(ComparisonResult[[1]])
}
