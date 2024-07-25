# ╭─────────────────────────────────────────────────────────────────────────────╮
# │                                                                             │
# │ Functions to simulate detections (2-MMPP)                                   │
# │ Léa Pautrel, TerrOïko | CEFE | IRMAR                                        │
# │ Last update: October 2023                                                   │
# │                                                                             │
# ╰─────────────────────────────────────────────────────────────────────────────╯


# ──────────────────────────────────────────────────────────────────────────────
# PACKAGES                                                                  ----
# ──────────────────────────────────────────────────────────────────────────────

library(tidyverse)
library(ggplot2)
library(progress)
library(glue)

# ──────────────────────────────────────────────────────────────────────────────
# DATASET SIMULATION                                                        ----
# ──────────────────────────────────────────────────────────────────────────────

switch_state = function(current_state) {
  "
  switch_state
  Switch between state 1  and state 2
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  current_state
    1 or 2
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  2 or 1 
  "
  ifelse(current_state == 1, 2, 1)
}


simulate_2MMPP <- function(NbSites,
                           NbDeployPerSite,
                           DeployementTimeValues,
                           psi,
                           lambda,
                           mu) {
  "
  simulate_2MMPP
  Simulate detection data set in continuous time with a 2-MMPP occupancy model.
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  NbSites
    Number of sites (integer)
  
  NbDeployPerSite
    Number of deployments per site (integer or vector of integers)
  
  DeployementTimeValues
    Duration of one deployment in one site (numeric or vector of numerics)
  
  psi
    Occupancy probability for the simulated sites (numeric between 0 and 1)
  
  lambda
    Vector of c('lambda_1' = numeric, 'lambda_2' = numeric), 
    with lambda_1 the detection rate in state 1
    and lambda_2 the detection rate in state 2
    e.g.  if lambda_1 = 5; there are on average 5 detections per time-unit when
          the system is in state 1
  
  mu
    Vector of c('mu_12' = numeric, 'mu_21' = numeric), 
    with mu_12 the switching rate from state 1 to state 2
    and mu_21 the switching rate from state 2 to state 1
    e.g.  if mu_12 = 1/15; there is 1/15 switch to state 2 per day on average,
          corresponding to 15 days spent on average in state 1 before switching

  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  A list (here called 'SimulatedDataset' with these informations:
  
    SimulatedDataset[[site i]][[deployment j]][['DeploymentTime']]
        The duration of the jth deployment in site i (numeric)
  
    SimulatedDataset[[site i]][[deployment j]][['DetectionsTimes']]
        A vector with the time of all detections of the jth deployment in site i
  
    SimulatedDataset[[site i]][[deployment j]][['HiddenState']]
        A dataframe with two columns: 'State' and 'BeginningTime' 
        to describe all switchs between states
  "
  
  # Retrieving parameters
  lambda_1 = lambda[1]
  lambda_2 = lambda[2]
  mu_12 = mu[1]
  mu_21 = mu[2]
  
  # Number of deployments per site
  # (random selection from values in vector)
  if (length(NbDeployPerSite) > 1) {
    list_R_i = sample(NbDeployPerSite, size = NbSites, replace = TRUE)
  } else {
    list_R_i = rep(NbDeployPerSite, NbSites)
  }
  
  # Deployment time in each site and deployment (in days)
  list_T_ij = vector(length = NbSites, mode = "list")
  if (length(DeployementTimeValues) > 1) {
    for (i in 1:NbSites) {
      list_T_ij[[i]] = sample(DeployementTimeValues,
                              size = list_R_i[i],
                              replace = TRUE)
    }
  } else{
    for (i in 1:NbSites) {
      list_T_ij[[i]] = rep(DeployementTimeValues, list_R_i[i])
    }
  }
  
  # The steady-state vector of the Markov chain is pi (Fisher & Meier-Hellstern 1992)
  # Initial distribution of the 2-state markov process
  # (general property, see Fisher 1992 or Rydén 1994)
  pi_prim = c("pi'_1" = mu_21 / (mu_12 + mu_21),
              "pi'_2" = mu_12 / (mu_12 + mu_21))
  
  # Simulating the true occupancy state z for the site
  z_i <-
    sample(
      x = c(0, 1),
      size = NbSites,
      prob = c(1 - psi, psi),
      replace = TRUE
    )
  
  # Simulating the time of detections for each site
  SimulatedDataset = setNames(vector(mode = "list", length = NbSites),
                              paste0("Site", 1:NbSites))
  pb = newpb(sum(list_R_i), txt = "Deployment", show_after = 1)
  for (i in 1:NbSites) {
    # Update of the result list with the number of detections
    SimulatedDataset[[i]] <-
      setNames(vector(mode = "list", length = list_R_i[i] + 1),
               c(paste0("Deployment", 1:list_R_i[i]),
                 "Occupied"))
    SimulatedDataset[[i]]$Occupied <- z_i[i]
    
    # For each deployment j at site i
    for (j in 1:(list_R_i[i])) {
      # Initialisation for SimulatedDataset[[i]][[j]]
      SimulatedDataset[[i]][[j]] <-
        setNames(
          vector(mode = "list", length = 3),
          c("DeploymentTime", "DetectionsTimes", "HiddenState")
        )
      

      
      # Begin the simulation
      current_time = 0
      current_state = sample(c(1, 2), size = 1, prob = pi_prim)
      switch_cpt = 2
      detecs_cpt = 1
      
      # Initialisation of the result vectors
      detecs = rep(NA, round(max(lambda) * list_T_ij[[i]][j] * 1.25))
      states = data.frame(
        "State" = c(current_state, rep(NA, max(mu) * list_T_ij[[i]][j] * 1.25)),
        "BeginningTime" = c(current_time, rep(NA, max(mu) * list_T_ij[[i]][j] * 1.25))
      )
      
      if (z_i[i] == 1) {
        while (current_time < list_T_ij[[i]][j]) {
          time_before_switching_state = rexp(n = 1, rate = mu[current_state])
          if (current_time + time_before_switching_state > list_T_ij[[i]][j]) {
            time_before_switching_state = list_T_ij[[i]][j] - current_time
          }
          
          # getting the number of detections while we're in this state
          nb_detecs = rpois(n = 1,
                            lambda = lambda[current_state] * time_before_switching_state)
          
          if (nb_detecs > 0) {
            # getting the time of those detections
            time_of_detections = runif(
              n = nb_detecs,
              min = current_time,
              max = current_time + time_before_switching_state
            )
            detecs[detecs_cpt:(detecs_cpt + nb_detecs - 1)] <-
              sort(time_of_detections)
            detecs_cpt = detecs_cpt + nb_detecs
          }
          
          current_time = current_time + time_before_switching_state
          current_state = switch_state(current_state)
          if (current_time != list_T_ij[[i]][j]) {
            states[switch_cpt,] <- c(current_state, current_time)
            switch_cpt = switch_cpt + 1
          }
        }
      }
      
      # We remove NA in the results vectors if there are any
      # (They were initialized with a given number of NA, and if there were 
      # not enough detections or states-switch to fill up all the spots,
      # there are still meaningless NA.)
      detecs = detecs[!is.na(detecs)]
      states = na.omit(states)

      # Write the simulation results
      SimulatedDataset[[i]][[j]][["DeploymentTime"]] <- list_T_ij[[i]][j]
      SimulatedDataset[[i]][[j]][["DetectionsTimes"]] <- detecs
      SimulatedDataset[[i]][[j]][["HiddenState"]] <- states
      pb$tick()
      
    } # end deployment j loop
  } # end site i loop
  
  return(SimulatedDataset)
}


simulate_2MMPP_V2 <- function(NbSites,
                              NbDeployPerSite,
                              DeployementTimeValues,
                              psi,
                              lambda,
                              mu) {
  "
  simulate_2MMPP_V2
  Simulate detection data set in continuous time with a 2-MMPP occupancy model.
  Produce the same results as simulate_2MMPP but using a different algorithm.
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  NbSites
    Number of sites (integer)
  
  NbDeployPerSite
    Number of deployments per site (integer or vector of integers)
  
  DeployementTimeValues
    Duration of one deployment in one site (numeric or vector of numerics)
  
  psi
    Occupancy probability for the simulated sites (numeric between 0 and 1)
  
  lambda
    Vector of c('lambda_1' = numeric, 'lambda_2' = numeric), 
    with lambda_1 the detection rate in state 1
    and lambda_2 the detection rate in state 2
    e.g.  if lambda_1 = 5; there are on average 5 detections per time-unit when
          the system is in state 1
  
  mu
    Vector of c('mu_12' = numeric, 'mu_21' = numeric), 
    with mu_12 the switching rate from state 1 to state 2
    and mu_21 the switching rate from state 2 to state 1
    e.g.  if mu_12 = 1/15; there is 1/15 switch to state 2 per day on average,
          corresponding to 15 days spent on average in state 1 before switching

  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  A list (here called 'SimulatedDataset' with these informations:
  
    SimulatedDataset[[site i]][[deployment j]][['DeploymentTime']]
        The duration of the jth deployment in site i (numeric)
  
    SimulatedDataset[[site i]][[deployment j]][['DetectionsTimes']]
        A vector with the time of all detections of the jth deployment in site i
  
    SimulatedDataset[[site i]][[deployment j]][['HiddenState']]
        A dataframe with two columns: 'State' and 'BeginningTime' 
        to describe all switchs between states
  "
  
  
  # Retrieving parameters
  lambda_1 = lambda[1]
  lambda_2 = lambda[2]
  mu_12 = mu[1]
  mu_21 = mu[2]
  
  # Number of deployments per site
  # (random selection from values in vector)
  if (length(NbDeployPerSite) > 1) {
    list_R_i = sample(NbDeployPerSite, size = NbSites, replace = TRUE)
  } else {
    list_R_i = rep(NbDeployPerSite, NbSites)
  }
  
  # Deployment time in each site and deployment (in days)
  list_T_ij = vector(length = NbSites, mode = "list")
  if (length(DeployementTimeValues) > 1) {
    for (i in 1:NbSites) {
      list_T_ij[[i]] = sample(DeployementTimeValues,
                              size = list_R_i[i],
                              replace = TRUE)
    }
  } else{
    for (i in 1:NbSites) {
      list_T_ij[[i]] = rep(DeployementTimeValues, list_R_i[i])
    }
  }
  
  # The steady-state vector of the Markov chain is pi (Fisher & Meier-Hellstern 1992)
  # Initial distribution of the 2-state markov process
  # (general property, see Fisher 1992 or Rydén 1994)
  pi_prim = c("pi'_1" = mu_21 / (mu_12 + mu_21),
              "pi'_2" = mu_12 / (mu_12 + mu_21))
  
  # Simulating the true occupancy state z for the site
  z_i <-
    sample(
      x = c(0, 1),
      size = NbSites,
      prob = c(1 - psi, psi),
      replace = TRUE
    )
  
  # Simulating the time of detections for each site
  SimulatedDataset = setNames(vector(mode = "list", length = NbSites),
                              paste0("Site", 1:NbSites))
  pb = newpb(sum(list_R_i), txt = "Deployment", show_after = 1)
  for (i in 1:NbSites) {
    # Update of the result list with the number of detections
    SimulatedDataset[[i]] <-
      setNames(vector(mode = "list", length = list_R_i[i] + 1),
               c(paste0("Deployment", 1:list_R_i[i]),
                 "Occupied"))
    SimulatedDataset[[i]]$Occupied <- z_i[i]
    
    # For each deployment j at site i
    for (j in 1:(list_R_i[i])) {
      
      # Initialisation for SimulatedDataset[[i]][[j]]
      SimulatedDataset[[i]][[j]] <-
        setNames(vector(mode = "list", length = 3),
                 c("DeploymentTime", "DetectionsTimes", "HiddenState"))
      
      
      # Begin the simulation
      current_time = 0
      current_state = sample(c(1, 2), size = 1, prob = pi_prim)
      switch_cpt = 2
      detecs_cpt = 1
      
      # Initialisation of the result vectors
      detecs = rep(NA, round(max(lambda) * list_T_ij[[i]][j] * 1.25))
      states = data.frame(
        "State" = c(current_state, rep(NA, max(mu) * list_T_ij[[i]][j] * 1.25)),
        "BeginningTime" = c(current_time, rep(NA, max(mu) * list_T_ij[[i]][j] * 1.25))
      )
      
      if (z_i == 1) {
        while (current_time <= list_T_ij[[i]][j]) {
          time_of_event = rexp(n = 1, rate = mu[current_state] + lambda[current_state])
          type_of_event = sample(
            c("detection", "switch"),
            prob = c(
              "detection" = lambda[current_state] / (mu[current_state] + lambda[current_state]),
              "switch" = mu[current_state] / (mu[current_state] + lambda[current_state])
            ),
            size = 1
          )
          current_time <- current_time + time_of_event
          
          if (current_time <= list_T_ij[[i]][j]) {
            if (type_of_event == "switch") {
              current_state <- switch_state(current_state)
              states[switch_cpt, ] <- c(current_state, current_time)
              switch_cpt <- switch_cpt + 1
            } else {
              detecs[detecs_cpt] <- current_time
              detecs_cpt <- detecs_cpt + 1
            }
          }
        }
      }
      
      # We remove NA in the results vectors if there are any
      # (They were initialized with a given number of NA, and if there were 
      # not enough detections or states-switch to fill up all the spots,
      # there are still meaningless NA.)
      detecs = detecs[!is.na(detecs)]
      states = na.omit(states)
      
      # Write the simulation results
      SimulatedDataset[[i]][[j]][["DeploymentTime"]] <- list_T_ij[[i]][j]
      SimulatedDataset[[i]][[j]][["DetectionsTimes"]] <- detecs
      SimulatedDataset[[i]][[j]][["HiddenState"]] <- states
      pb$tick()
      
    } # end deployment j loop
  } # end site i loop
  
  return(SimulatedDataset)
}

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETERS                                                                ----
# ──────────────────────────────────────────────────────────────────────────────

get_p_from_2MMPP_param = function(lambda, mu, pT) {
  "
  get_p_from_2MMPP_param
  Calculates the detection probability, that is the probability of having
  at least one detection during a given duration pT depending on the detection
  parameters of a 2-MMPP, lambda (lambda_1, lambda_2) and mu (mu_12, mu_21).
  The equation comes from Guillera-Arroita et al (2011), p. 311:
  'the probabilities of detecting the species at an occupied site, 1 − π exp(CL)e'

  
  INPUTS ───────────────────────────────────────────────────────────────────────
  lambda
    Vector of c('lambda_1' = numeric, 'lambda_2' = numeric), 
    with lambda_1 the detection rate in state 1
    and lambda_2 the detection rate in state 2
    e.g.  if lambda_1 = 5; there are on average 5 detections per time-unit when
          the system is in state 1
  
  mu
    Vector of c('mu_12' = numeric, 'mu_21' = numeric), 
    with mu_12 the switching rate from state 1 to state 2
    and mu_21 the switching rate from state 2 to state 1
    e.g.  if mu_12 = 1/15; there is 1/15 switch to state 2 per day on average,
          corresponding to 15 days spent on average in state 1 before switching
  
  pT
    Duration for the detection probability (numeric or vector of numerics)
  

  OUTPUT ───────────────────────────────────────────────────────────────────────
  Numeric or vector of numerics of the same length as pT between 0 and 1.
  "
  
  # Retrieving parameters
  lambda_1 = lambda[1]
  lambda_2 = lambda[2]
  mu_12 = mu[1]
  mu_21 = mu[2]
  
  res = rep(NA, length(pT))
  for (j in 1:length(pT)) {
    pTunique = pT[j]
    res[j] =
      (
        1 - matrix(c(
          mu_21 / (mu_12 + mu_21),
          mu_12 / (mu_12 + mu_21)
        ),
        nrow = 1) %*% expm::expm((
          matrix(
            data = c(-mu_12,
                     mu_12,
                     mu_21,-mu_21),
            ncol = 2,
            byrow = T
          ) - matrix(
            data = c(lambda_1,
                     0,
                     0,
                     lambda_2),
            ncol = 2,
            byrow = T
          )
        ) * pTunique) %*% matrix(c(1, 1), nrow = 2)
      )[1, 1]
  }
  return(res)
}



get_p_from_PP_param <- function(lambda, pT) {
  "
  get_p_from_PP_param
  Calculates the detection probability, that is the probability of having
  at least one detection during a given duration pT depending on the detection
  parameter of a PP lambda.
  The equation comes from Guillera-Arroita et al (2011), p. 305:
  'where λ∗ = 1 − exp(−λL)'
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  lambda
    Numeric, the detection rate.
    e.g.  if lambda = 5; there are on average 5 detections per time-unit
  
  pT
    Duration for the detection probability (numeric or vector of numerics)
  

  OUTPUT ───────────────────────────────────────────────────────────────────────
  Numeric or vector of numerics of the same length as pT between 0 and 1.
  "
  
  return(1 - exp(-lambda * pT))
}




get_p_from_COP_param <- function(lambda, SessionLength, NbDeployPerSite, pT) {
  "
  get_p_from_COP_param
  Calculates the detection probability, that is the probability of having
  at least one detection during a given duration pT depending on the detection
  parameters of a COP, lambda and SessionLength.
  The equation comes from Emmet et al (2021), p. 5: 'p0 = 1−exp(−(λ0L)/K)'
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  lambda
    Numeric, the detection rate.
    e.g.  if lambda = 5; there are on average 5 detections per time-unit
  
  SessionLength
    Numeric, duration of the discretised session

  NbDeployPerSite
    Numeric, the number of deployments per site

  pT
    Duration for the detection probability (numeric or vector of numerics)
  

  OUTPUT ───────────────────────────────────────────────────────────────────────
  Numeric or vector of numerics of the same length as pT between 0 and 1.
  "
  
  return(1 - exp(- lambda * (pT %/% SessionLength * SessionLength) / NbDeployPerSite))
}


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATED DATASET EXPLORATION                                             ----
# ──────────────────────────────────────────────────────────────────────────────

extract_simulated_infos = function(SimulatedDataset, quiet = TRUE) {
  "
  extract_simulated_infos
  Extract the main informations from the simulated dataset
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  SimulatedDataset
    List produced by function simulate_2MMPP
  
  quiet
    (facultative)
    Boolean. If FALSE, the main informations are printed
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  A list with:
    'z_i':  a vector of length NbSites, with the occupation states per site (1 if occupied, 0 if not)
    'list_R_i': a vector of length NbSites, the number of deployments per site (integer)
    'list_T_ij': a list, list_T_ij[[i]][j]] is the time during which a sensor is deployed for deployment j at site i
    'RecapNDetec': a dataframe that summarises the number of detections per site and deployment
  "
  
  # z_i[i] is the occupation state of site i (1 if occupied, 0 if not)
  z_i <- sapply(SimulatedDataset, function(x) {
    x[[length(x)]]
  })
  if (!quiet) {
    cat("z_i[i] is the occupation state of site i (1 if occupied, 0 if not)",
        fill = T)
    print(table(z_i))
    cat(glue::glue("Part of occupied sites: {sum(z_i == 1) / length(z_i)}"),
        fill = T)
    cat("\n")
  }
  
  
  # list_R_i[i] is the number of deployments at site i
  list_R_i <- sapply(SimulatedDataset, function(x) {
    length(x) - 1
  })
  if (!quiet) {
    cat("list_R_i[i] is the number of deployments at site i", fill = T)
    print(table(list_R_i))
    cat("\n")
  }
  
  
  # list_T_ij[[i]][j]] is the time during which a sensor is deployed for deployment j at site i
  list_T_ij <- lapply(SimulatedDataset, function(x) {
    unlist(sapply(x[-length(x)], function(y) {
      unname(y[1])
    }))
  })
  if (!quiet) {
    cat(
      "list_T_ij[[i]][j]] is the time during which a sensor is deployed for deployment j at site i",
      fill = T
    )
    print(summary(unlist(list_T_ij)))
    cat("\n")
  }
  
  # RecapNDetec is a dataframe that summarises the number of detections per site and deployment
  RecapNDetec = lapply(SimulatedDataset,
                       function(x) {
                         sapply(x[-length(x)], function(y) {
                           sum(!is.na(y[[2]]))
                         })
                       }) %>%
    plyr::ldply(., rbind) %>%
    tibble::as_tibble() %>%
    dplyr::mutate(Occupied = z_i) %>%
    dplyr::relocate(Site = .id, Occupied)
  if (!quiet) {
    cat(
      "RecapNDetec is a dataframe that summarises the number of detections per site and deployment",
      fill = T
    )
    print(RecapNDetec)
    print(summary(RecapNDetec))
    cat("\n")
  }
  
  return(
    list(
      "z_i" = z_i,
      "list_R_i" = list_R_i,
      "list_T_ij" = list_T_ij,
      "RecapNDetec" = RecapNDetec
    )
  )
}


plot_detection_times = function(SimulatedDataset,
                                z_i = NULL,
                                list_R_i = NULL,
                                list_T_ij = NULL,
                                RecapNDetec = NULL,
                                lambda = NULL,
                                mu = NULL,
                                nplots_by_grid = 20) {
  "
  plot_detection_times
  Plot the detection times per site
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  SimulatedDataset
    List produced by function simulate_2MMPP
  
  z_i, list_R_i, list_T_ij, RecapNDetec
    (facultative)
    Produced by the function extract_simulated_infos
  
  lambda, mu
    (facultative)
    Detection parameters of the 2-MMPP
  
  nplots_by_grid
    (facultative)
    Number of plots grouped in one page within a grid 
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  A list of ggplots
  "
  
  
  # Retrieve the number of sites
  NbSites = length(SimulatedDataset)
  SiteNames = parse_number(names(SimulatedDataset))
  
  # If any recapitulative infos is missing, retrieve it
  if (is.null(RecapNDetec) | is.null(list_R_i) | is.null(list_T_ij) | is.null(z_i)) {
    main_simul_infos = extract_simulated_infos(SimulatedDataset, quiet = TRUE)
    RecapNDetec = main_simul_infos[["RecapNDetec"]]
    list_R_i = main_simul_infos[["list_R_i"]]
    list_T_ij = main_simul_infos[["list_T_ij"]]
    z_i = main_simul_infos[["z_i"]]
    rm(main_simul_infos)
    invisible(gc())
  }
  
  # Plotting the detection times
  plots = vector(mode = "list", length = length(NbSites))
  cpt1 = 0
  ggplot2::theme_set(ggplot2::theme_minimal())
  for (i in 1:NbSites) {
    site_name = SiteNames[i]
    
    # Storing the "events" (ie detections) data per deployment (columns)
    data_events = matrix(ncol = list_R_i[i],
                  nrow = max(as.vector(
                    dplyr::select(RecapNDetec, starts_with("Deployment"))[i,], mode = "integer"
                  ), na.rm = T))
    
    if (nrow(data_events) > 0) {
      # Retrieving the states and detection data for each deployment
      # for (j in 1:list_R_i[i]) { 
      # the loop is useful to plot several deployments,
      # but we'll only plot the 1st deployment for now.
      j = 1
      {
        # Detections
        ndetec_ij = as.vector(RecapNDetec[i, j + 2], mode = "integer")
        if (ndetec_ij > 0) {
          data_events[1:ndetec_ij, j] <-
            SimulatedDataset[[i]][[j]]$DetectionsTimes
        }
        
        # States
        tmp = SimulatedDataset[[i]][[j]]$HiddenState
        tmp$EndTime = c(tmp$BeginningTime[-1], list_T_ij[[i]])
        tmp$Deployment = j
        if (exists("data_states")) {
          data_states = rbind(data_states, tmp)
        } else {
          data_states = tmp
        }
      }
      
      data_states = data_states %>% 
        mutate(State = factor(State),
               Deployment = as.factor(Deployment))
      
      if (!is.null(lambda) & !is.null(mu)){
        x = 1
        txt_state_1 = glue::glue("{x} ({lambda[x]} detections/day ; µ{ifelse(x==1,'_12','_21')} = {round(mu[x],3)})")
        
        x = 2
        txt_state_2 = glue::glue("{x} ({lambda[x]} detections/day ; µ{ifelse(x==1,'_12','_21')} = {round(mu[x],3)})")
        
        data_states$State <- factor(
          sapply(data_states$State, function(x) {
            ifelse(x == 1, txt_state_1, txt_state_2)
          }),
          levels = c(txt_state_1, txt_state_2),
          ordered = T
        )
      } else {
        txt_state_1 = "1"
        txt_state_2 = "2"
      }
      
      data_events = data_events %>%
        as.data.frame() %>%
        dplyr::mutate(DetectionIndex = dplyr::row_number()) %>%
        setNames(c(paste0("Deployment", 1:list_R_i[i]), "DetectionIndex")) %>%
        tidyr::pivot_longer(
          cols = starts_with("Deployment"),
          values_to = "TimeOfDetection",
          names_to = "Deployment",
          names_prefix = "Deployment"
        ) %>%
        dplyr::mutate(Deployment = as.factor(Deployment))
      
      
      gg <- data_events %>%
        ggplot2::ggplot(ggplot2::aes(y = DetectionIndex, x = TimeOfDetection), color = "black") +
        ggplot2::geom_rect(
          data = data_states,
          inherit.aes = FALSE,
          aes(
            xmin = BeginningTime,
            xmax = EndTime,
            ymin = -Inf,
            ymax = Inf,
            fill = State
          ),
          alpha = .5
        ) +
        ggplot2::geom_line(na.rm = TRUE) +
        ggplot2::geom_point(na.rm = TRUE) +
        ggplot2::labs(
          title = glue::glue("Site {site_name} (Deployment 1/{list_R_i[i]})"),
          y = glue::glue("{max(data_events$DetectionIndex)} detection{ifelse(max(data_events$DetectionIndex)>1, 's', '')}"),
          x = glue::glue("Time ({list_T_ij[[i]]} days)"),
          fill = "State"
        ) +
        scale_x_continuous(limits = c(0, list_T_ij[[i]]), expand = c(0, 0)) +
        ggplot2::theme(axis.text.y = element_blank(),
                       legend.position = c(0.1, 0.8),
                       legend.background = element_rect(fill = "white", colour = "white")) +
        ggplot2::scale_fill_manual(values = eval(parse(text=paste0("c(
          '", as.character(txt_state_1), "' = 'grey80',
          '", as.character(txt_state_2), "' = 'grey20'
        )"))),drop = FALSE)
      
    } else {
      
      # Retrieving the states data
      j = 1
      {
        # States
        tmp = SimulatedDataset[[i]][[j]]$HiddenState
        tmp$EndTime = c(tmp$BeginningTime[-1], list_T_ij[[i]])
        tmp$Deployment = j
        if (exists("data_states")) {
          data_states = rbind(data_states, tmp)
        } else {
          data_states = tmp
        }
      }
      
      data_states = data_states %>% 
        mutate(State = factor(State),
               Deployment = as.factor(Deployment))
      
      if (!is.null(lambda) & !is.null(mu)){
        x = 1
        txt_state_1 = glue::glue("{x} ({lambda[x]} detections/day ; µ{ifelse(x==1,'_12','_21')} = {round(mu[x],3)})")
        
        x = 2
        txt_state_2 = glue::glue("{x} ({lambda[x]} detections/day ; µ{ifelse(x==1,'_12','_21')} = {round(mu[x],3)})")
        
        data_states$State <- factor(
          sapply(data_states$State, function(x) {
            ifelse(x == 1, txt_state_1, txt_state_2)
          }),
          levels = c(txt_state_1, txt_state_2),
          ordered = T
        )
      }
      
      gg = ggplot2::ggplot() +
        ggplot2::labs(
          title = glue::glue("Site {site_name} (Deployment 1/{list_R_i[i]})"),
          y = glue::glue("No detections"),
          x = glue::glue("Time ({list_T_ij[[i]]} days)"),
          fill = "State"
        ) +
        ggplot2::geom_rect(
          data = data_states,
          inherit.aes = FALSE,
          aes(
            xmin = BeginningTime,
            xmax = EndTime,
            ymin = -Inf,
            ymax = Inf,
            fill = State
          ),
          alpha = .5
        ) +
        ggplot2::lims(y = c(0, 1)) +
        ggplot2::scale_fill_manual(values = eval(parse(text=paste0("c(
          '", as.character(txt_state_1), "' = 'grey80',
          '", as.character(txt_state_2), "' = 'grey20'
        )"))),drop = FALSE) +
        scale_x_continuous(limits = c(0, list_T_ij[[i]]), expand = c(0, 0)) +
        ggplot2::theme(
          axis.text.y = element_blank(),
          legend.position = c(0.1, 0.8),
          legend.background = element_rect(fill = "white", colour = "white")
        )
      }
    
    
    if (z_i[i] == 0) { 
      gg = gg + ggplot2::theme(plot.title = ggplot2::element_text(colour = "#8b0000", face = "bold", hjust = 0.5))
    } else{
      gg = gg + ggplot2::theme(plot.title = ggplot2::element_text(colour = "#006400", face = "bold", hjust = 0.5))
    }
    
    rm(data_states)
    plots[[i]] = gg
  }
  
  nb_plots_grouped = NbSites %/% nplots_by_grid + ifelse(NbSites %% nplots_by_grid > 0, 1, 0)
  plots_grouped = vector(mode="list", length = nb_plots_grouped)
  cpt = 1
  
  for (i in 1:nb_plots_grouped) {
    plots_grouped[[i]] = ggpubr::ggarrange(
      plotlist = plots[cpt:min(cpt + nplots_by_grid - 1, NbSites)],
      common.legend = TRUE
    )
    cpt = cpt + nplots_by_grid
  }
  return(plots_grouped)
}

# ──────────────────────────────────────────────────────────────────────────────
# DISCRETISATION                                                            ----
# ──────────────────────────────────────────────────────────────────────────────


get_nbdetec_per_session <- function(SimulatedDataset, list_R_i = NULL, list_T_ij = NULL, SessionLength = 1){
  "
  get_nbdetec_per_session
  Discretise the simulated data set with the number of detections per discretised
  session of length SessionLength.
  If the last session is incomplete, the last detections are dropped. 
  For example, with a 5-days deployment and a 2-days session with these 
  detection times in one site:
  detecs=c(0.4, 0.9, 1.7, 2.4, 4.7)
  we'll have 3 detections in session 1, 1 detection in session 2, 
  and we won't use the detection at the 4.7th day.
  
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  
  SimulatedDataset
    Data set simulated with function `simulate_2MMPP`
  
  SessionLength
    Integer giving the session length
    (in the same time unit that was given to `simulate_2MMPP`)
  
  list_R_i, list_T_ij
    (facultative)
    Produced by the function extract_simulated_infos

  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  
  A matrix 
    1 row = 1 deployment (in a given site)
    1 column = 1 session
    the values = number of detections during this session for this deployment
  "
  
  # If any recapitulative infos is missing, retrieve it
  if (is.null(list_R_i) | is.null(list_T_ij)) {
    main_simul_infos = extract_simulated_infos(SimulatedDataset, quiet = TRUE)
    list_R_i = main_simul_infos[["list_R_i"]]
    list_T_ij = main_simul_infos[["list_T_ij"]]
    rm(main_simul_infos)
    invisible(gc())
  }
  
  # Maximum deployment time, all deployments and all sites confounded
  MaxDeploymentTime = max(unlist(list_T_ij))
  
  # Number of sessions in this maximum deployment time
  MaxNbSessions = MaxDeploymentTime %/% SessionLength
  
  # Initialisation of the result matrix
  SimulatedDataset_Discretised_NbDetec = matrix(
    data = NA,
    nrow = sum(list_R_i), # Sites x Deployments per site
    ncol = MaxNbSessions # Sessions
  )
  
  # Initialise the iterator to keep track of which row to write
  row_index = 0
  row_names = rep(NA, sum(list_R_i))
  
  # Loop for each deployment
  for (i in 1:length(SimulatedDataset)) {
    for (j in 1:list_R_i[i]) {
      # Next row
      row_index = row_index + 1
      row_names[row_index] = paste0("Site", i, "_Deployment", j)
      
      # How long was this deployment?
      length_of_deployment = list_T_ij[[i]][j]
      
      # When were the detections for this deployment?
      detecs = SimulatedDataset[[i]][[j]]$DetectionsTimes %>% .[!is.na(.)]
      
      if (length(detecs) == 0) {
        # If there were no detections, we fill a vector of number of detections by session with 0
        tmp = rep(0, ncol(SimulatedDataset_Discretised_NbDetec))
      } else{
        # If there were detections, we fill a vector of number of detections by session
        tmp = cut(
          detecs,
          breaks = seq(from = 0, to = SessionLength * MaxNbSessions, by = SessionLength),
          labels = 1:MaxNbSessions
        ) %>%
          table() %>%
          as.vector()
      }
      
      # If not all sessions were covered by the deployment,
      # we replace number of detections that were not covered by NA
      if (length_of_deployment < length(tmp)) {
        tmp[(length_of_deployment + 1):length(tmp)] <- NA
      }
      
      SimulatedDataset_Discretised_NbDetec[row_index, ] = tmp
    }
  }
  
  # Add dimnames
  rownames(SimulatedDataset_Discretised_NbDetec) <- row_names
  colnames(SimulatedDataset_Discretised_NbDetec) <- paste0("Session", 1:MaxNbSessions)
  
  return(SimulatedDataset_Discretised_NbDetec)
}


get_detected_per_session = function(SimulatedDataset_NbDetecsPerSession) {
  "
  get_detected_per_session
  Transforms a matrix with the number of detections per site and session
  into a matrix of detection/non-detection
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  SimulatedDataset_NbDetecsPerSession
    Matrix produced by the function get_nbdetec_per_session

  OUTPUT ───────────────────────────────────────────────────────────────────────
  A matrix 
    1 row = 1 site
    1 column = 1 session
    the values = 0 if no detection, 1 if detection
  "
  
  return((SimulatedDataset_NbDetecsPerSession > 1) * 1)
}

