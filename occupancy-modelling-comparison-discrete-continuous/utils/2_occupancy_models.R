# ╭─────────────────────────────────────────────────────────────────────────────╮
# │                                                                             │
# │ Functions to run occupancy models                                           │
# │ Léa Pautrel, TerrOïko | CEFE | IRMAR                                        │
# │ Last update: October 2023                                                   │
# │                                                                             │
# ╰─────────────────────────────────────────────────────────────────────────────╯

# ──────────────────────────────────────────────────────────────────────────────
# PACKAGES                                                                  ----
# ──────────────────────────────────────────────────────────────────────────────

library(tidyverse)
library(ggplot2)
library(expm)

# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORMATIONS                                                           ----
# ──────────────────────────────────────────────────────────────────────────────

# The following code helps understanding the mathematical transformations 
# that change the domain of definition of a parameters.
# 
# Some parameters can take only some values (e.g. a probability ∈ [0, 1]),
# and we transform them so that they can take any value (param_transformed ∈ ℝ)
# to facilitate the maximum likelihood estimation.

## p ∈ [0, 1] (probability) ----------------------------------------------------

# Example: psi the occupancy probability.
# psi without transformation is between 0 and 1
# psi_transformed = log(1 / psi - 1))
# psi = 1 / (1 + exp(psi_transformed))


logit = function(p){
  x = log(p / (1 - p))
  return(x)
}

invlogit = function(x) {
  p = 1 / (1 + exp(-x))
  return(p)
}

# Does not run when the script is sourced
if (FALSE) {
  data.frame("psi" = seq(-1, 2, by = 0.01),
             "psi_transformed" = logit(seq(-1, 2, by = 0.01))) %>%
    ggplot(data = ., aes(y = psi, x = psi_transformed)) +
    geom_line(na.rm = T)
  
  data.frame("psi_transformed" = seq(-40, 40, by = 0.1),
             "psi" = invlogit( seq(-40, 40, by = 0.1))) %>%
    ggplot(data = ., aes(y = psi, x = psi_transformed)) +
    geom_line(na.rm = T)
}


## pos ∈ ℝ+ (positive or nul reals) --------------------------------------------

# Example for lambda in the 2-MMPP:
# lambda without transformation is between 0 and +Inf
# lambda_transformed = log(lambda)
# lambda = exp(lambda_transformed)

# Do not run when the script is sourced
if (FALSE) {
  data.frame("lambda" = seq(-10, 50, by = 0.1),
             "lambda_transformed" = log(seq(-10, 50, by = 0.1))) %>%
    ggplot(data = ., aes(y = lambda, x = lambda_transformed)) +
    geom_line(na.rm = T)
  
  data.frame("lambda_transformed" = seq(-5, 10, by = 0.05),
             "lambda" = exp(seq(-5, 10, by = 0.05))) %>%
    ggplot(data = ., aes(y = lambda, x = lambda_transformed)) +
    geom_line(na.rm = T)
}



# ──────────────────────────────────────────────────────────────────────────────
# 2-MMPP 2 STATES MARKOV MODULATED POISSON PROCESS                          ----
# ──────────────────────────────────────────────────────────────────────────────

get_2MMPP_M_ij = function(t_ij, LAMBDA, mat_C, pi_vect, quick = F) {
  "
  get_2MMPP_M_ij is a function that returns M_ij.
  M_ij is the expression for the contribution to the likelihood of data 
  from deployment j at site i described as a 2−MMPP.
  See Guillera-Arroita et al (2011), p.308
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  
  t_ij    Inter-detection times from deployment j at site i
          Vector of dij+1 values where dij is the number of detections from j at i
  
  LAMBDA  𝚲 = diag{λ1, λ2}
          
  mat_C   mat_C = mat_Q - 𝚲
          with mat_Q the generator matrix
  
  pi_vect Equilibrium distribution of the underlying Markov process
          Vector of 2 values: 𝞹1 et 𝞹2
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  
  M_ij
  "
  
  mat_e = matrix(c(1, 1), ncol = 1, nrow = 2)

  # M_ij calculation (3.3 equation)
  for (k in 1:length(t_ij)) {
    
    mat_C_t_ij = mat_C * t_ij[k]
    
    if (any(is.infinite(mat_C_t_ij))) {
      for (isinf in which((is.infinite(mat_C_t_ij)))) {
        if ((mat_C_t_ij[isinf]) > 0) {
          mat_C_t_ij[isinf] <- .Machine$double.xmax
        } else{
          mat_C_t_ij[isinf] <- -.Machine$double.xmax
        }
      }
    }
    
    if (quick) {
      if (k == 1) {
        mat_res <-
          pi_vect %*% expm(mat_C_t_ij, method = "Taylor", order = 6) %*% LAMBDA
      } else if (k < length(t_ij)) {
        mat_res <-
          mat_res %*% expm(mat_C_t_ij, method = "Taylor", order = 6) %*% LAMBDA
      } else {
        mat_res <-
          mat_res %*% expm(mat_C_t_ij, method = "Taylor", order = 6)
      }
    } else {
      if (k == 1) {
        mat_res <- pi_vect %*% expm(mat_C_t_ij) %*% LAMBDA
      } else if (k < length(t_ij)) {
        mat_res <- mat_res %*% expm(mat_C_t_ij) %*% LAMBDA
      } else {
        mat_res <- mat_res %*% expm(mat_C_t_ij)
      }
    }
  }
  M_ij = (mat_res %*% mat_e)[1, 1]
  
  return(M_ij)
}



get_2MMPP_loglikelihood = function(param,
                                   detection_times,
                                   NbSites,
                                   list_T_ij,
                                   list_R_i, 
                                   quick = FALSE,
                                   debug_print = FALSE) {
  "
  get_2MMPP_loglikelihood
  Calculates the log-likelihood of observing the data detection_times given
  the 2-MMPP parameters in param.
  <!> In this function, there are no covariates taken into account.
      Occupancy probability psi and detection parameters lambda and mu are 
      constant across sites.
  
  INPUTS ───────────────────────────────────────────────────────────────────────

  param   Vector of parameters:
    psi_transformed       param[1]    logit(Occupancy probability)
    lambda_1_transformed  param[2]    log(Detection rate of the Markov state 1)
    lambda_2_transformed  param[3]    log(Detection rate of the Markov state 2)
    mu_12_transformed     param[4]    log(Switching rate from state 1 to 2)
    mu_21_transformed     param[5]    log(Switching rate from state 2 to 1)

  detection_times
          The times of detections, as a list of list
          detection_times[[site i]][[deployment j]] -> vector of detection times

  NbSites       Number of sites
          Integer

  list_T_ij
          Deployment time of deployment j at site i
          list_T_ij[[site i]] -> vector of the R_i deployments times at site i

  list_R_i
          Number of deployments at sites
          Vector of length NbSites


  OUTPUT ───────────────────────────────────────────────────────────────────────

  The value of the log-likelihood of these parameters given these data for a 2-MMPP
  "
  
  # Reading parameters
  psi = invlogit(param[1])
  
  # We choose lambda_1 < lambda_2
  # (as this seems to be the norm, see Skaug et al. 2006)
  if (param[2] < param[3]) {
    lambda_1 = unname(min(exp(param[2]), .Machine$double.xmax))
    lambda_2 = unname(min(exp(param[3]), .Machine$double.xmax))
    lambda = c("lambda_1" = lambda_1, "lambda_2" = lambda_2)
    
    mu_12 = unname(min(exp(param[4]), .Machine$double.xmax))
    mu_21 = unname(min(exp(param[5]), .Machine$double.xmax))
    mu = c("mu_12" = mu_12, "mu_21" = mu_21)
  } else {
    lambda_1 = unname(min(exp(param[3]), .Machine$double.xmax))
    lambda_2 = unname(min(exp(param[2]), .Machine$double.xmax))
    lambda = c("lambda_1" = lambda_1, "lambda_2" = lambda_2)
    
    mu_12 = unname(min(exp(param[5]), .Machine$double.xmax))
    mu_21 = unname(min(exp(param[4]), .Machine$double.xmax))
    mu = c("mu_12" = mu_12, "mu_21" = mu_21)
  }
  
  # pi_vect for this mu
  pi_vect = matrix(c(
    "pi'_1" = mu_21 / (mu_12 + mu_21),
    "pi'_2" = mu_12 / (mu_12 + mu_21)
  ),
  nrow = 1,
  ncol = 2)
  
  # mat_Q = Generator matrix
  mat_Q = matrix(
    data = c(-mu[1], mu[1],
             mu[2], -mu[2]),
    byrow = T, nrow = 2,
    dimnames = list(c("State 1", "State 2"), c("State 1", "State 2"))
  )
  
  # 𝚲 = diag{λ1,λ2}
  LAMBDA = matrix(
    data = c(lambda[1], 0,
             0, lambda[2]),
    byrow = T, nrow = 2,
    dimnames = list(c("State 1", "State 2"), c("State 1", "State 2"))
  )
  
  # mat_C = mat_Q - 𝚲
  mat_C = mat_Q - LAMBDA
  
  # for each site i in NbSites
  prod_i = rep(NA, NbSites)
  for (i in 1:NbSites) {
    # Initialisation
    prod_ij = rep(NA, list_R_i[i])
    
    # Number of detections in site (during all deployments)
    d_i = sum(sapply(detection_times[[i]], function(x) {
      length(x[!is.na(x)])
    }))
    
    for (j in 1:list_R_i[i]) {
      # Difference between detection times
      # 1st elem = time between the beginning of deployment and the 1st detection
      # last elem = time between the last detection and the end of deployment (list_T_ij[[i]][j])
      t_ij = diff(c(
        0,
        detection_times[[i]][[j]][!is.na(detection_times[[i]][[j]])],
        list_T_ij[[i]][j]
      ))
      
      # contribution to the likelihood of data from deployment j at site i
      M_ij = get_2MMPP_M_ij(
        t_ij = t_ij,
        LAMBDA = LAMBDA,
        mat_C = mat_C,
        pi_vect = pi_vect,
        quick = quick
      )
      
      prod_ij[j] <- M_ij
    }
    
    if (d_i > 0) {
      # if there were detections in site i
      prod_i[i] <- psi * prod(prod_ij)
    } else {
      # if there were NO detections in site i
      prod_i[i] <- psi * prod(prod_ij) + (1 - psi)
    }
    
  }
  
  # log_likelihood = log(prod(prod_i))
  # <!> if X are bigs then prod(X) can return Inf and then log(prod(X)) return Inf!!
  # so we write it as its equivalent: sum(log(prod_i))
  log_likelihood = sum(log(prod_i))
  if (debug_print) {
    cat("LLH =", log_likelihood, "\n")
    cat("psi =", psi, "\n")
    cat("lambda =", lambda, "\n")
    cat("mu =", mu, "\n")
  }
  return(log_likelihood)
}


get_2MMPP_neg_loglikelihood = function(param,
                                       detection_times,
                                       NbSites,
                                       list_T_ij,
                                       list_R_i,
                                       quick = FALSE, 
                                       debug_print = FALSE) {
  "
  get_2MMPP_neg_loglikelihood
  Returns the negative log-likelihood (calculated with get_2MMPP_loglikelihood).
  This function us useful for using optim for optimisation, which works best
  when minimising and not maximising.
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  See the documentation of the function get_2MMPP_loglikelihood
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  The negative log-likelihood of a 2-MMPP
  "
  -get_2MMPP_loglikelihood(param,
                           detection_times,
                           NbSites,
                           list_T_ij,
                           list_R_i,
                           quick = quick, 
                           debug_print = debug_print)
}



# ──────────────────────────────────────────────────────────────────────────────
# IPP INTERRUPTED POISSON PROCESS                                           ----
# ──────────────────────────────────────────────────────────────────────────────

# IPP means Interrupted Poisson Process
# This is a 2-MMPP where one of the lambda is fixed at 0
# (lets say lambda_2 = 0)

get_IPP_loglikelihood = function(param,
                                 detection_times,
                                 NbSites,
                                 list_T_ij,
                                 list_R_i,
                                 quick = FALSE) {
  "
  get_IPP_loglikelihood
  Calculates the log-likelihood of observing the data detection_times given
  the IPP parameters in param.
  <!> In this function, there are no covariates taken into account.
      Occupancy probability psi and detection parameters lambda and mu are 
      constant across sites.
  
  INPUTS ───────────────────────────────────────────────────────────────────────

  param   Vector of parameters:
    psi_transformed       param[1]    logit(Occupancy probability)
    lambda_2_transformed  param[2]    log(Detection rate of the Markov state 2)
    mu_12_transformed     param[3]    log(Switching (= transition) rate between Markov state 1 to 2)
    mu_21_transformed     param[4]    log(Switching (= transition) rate between Markov state 2 to 1)

  detection_times
          The times of detections, as a list of list
          detection_times[[site i]][[deployment j]] -> vector of detection times

  NbSites       Number of sites
          Integer

  list_T_ij
          Deployment time of deployment j at site i
          list_T_ij[[site i]] -> vector of the R_i deployments times at site i

  list_R_i
          Number of deployments at sites
          Vector of length NbSites


  OUTPUT ───────────────────────────────────────────────────────────────────────

  The value of the log-likelihood of these parameters given these data for a 2-MMPP
  "
  
  if (length(param) != 4) {
    stop("Parameters were not correctly given: `length(param) != 4`")
  }
  
  # Reading parameters
  psi_transformed = param[1]
  lambda_2_transformed = param[2]
  mu_12_transformed = param[3]
  mu_21_transformed = param[4]
  
  # Setting lambda_1 = 0
  lambda_1_transformed = log(0)
  
  # 2-MMPP with these parameters
  loglikelihood = get_2MMPP_loglikelihood(
    param = c(
      "psi" = psi_transformed,
      "lambda_1" = lambda_1_transformed,
      "lambda_2" = lambda_2_transformed,
      "mu_12" = mu_12_transformed,
      "mu_21" = mu_21_transformed
    ),
    detection_times = detection_times,
    NbSites = NbSites,
    list_T_ij = list_T_ij,
    list_R_i = list_R_i,
    quick = quick
  )
  
  return(loglikelihood)
}


get_IPP_neg_loglikelihood = function(param,
                                     detection_times,
                                     NbSites,
                                     list_T_ij,
                                     list_R_i,
                                     quick = FALSE) {
  "
  get_IPP_neg_loglikelihood
  Returns the negative log-likelihood (calculated with get_IPP_loglikelihood).
  This function us useful for using optim for optimisation, which works best
  when minimising and not maximising.
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  See the documentation of the function get_IPP_loglikelihood
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  The negative log-likelihood of a IPP
  "
  neg_loglikelihood = -get_IPP_loglikelihood(param,
                                             detection_times,
                                             NbSites,
                                             list_T_ij,
                                             list_R_i,
                                             quick = quick)
  return(neg_loglikelihood)
}



# ──────────────────────────────────────────────────────────────────────────────
# PP POISSON PROCESS                                                        ----
# ──────────────────────────────────────────────────────────────────────────────

# TODO :
# adjust the PP loglikelihood so that we can have different deployment times
# by deployments in the same site
get_PP_loglikelihood = function(param,
                                detection_times,
                                NbSites,
                                list_T_ij,
                                list_R_i){
  "
  get_PP_loglikelihood
  Calculates the log-likelihood of observing the data detection_times given
  the PP parameters in param.
  <!> In this function, there are no covariates taken into account.
      Occupancy probability psi and detection rate lambda are constant.
  

  INPUTS ───────────────────────────────────────────────────────────────────────

  param   Vector of parameters:
    psi_transformed       param[1]    Occupancy probability
                                      TRANSFORMATION: logit(psi)
    lambda_PP_transformed param[2]    Detection rate
                                      TRANSFORMATION: exp(lambda_PP)

  detection_times
          The times of detections, as a list of list
          detection_times[[site i]][[deployment j]] -> vector of detection times

  NbSites       Number of sites
          Integer

  list_T_ij
          Deployment time of deployment j at site i
          list_T_ij[[site i]] -> vector of the R_i deployments times at site i

  list_R_i
          Number of deployments at sites
          Vector of length NbSites


  OUTPUT ───────────────────────────────────────────────────────────────────────

  The value of the log-likelihood of these parameters given these data for a PP
  "
  
  # Reading parameters
  psi = invlogit(param[1])
  lambda_PP = exp(param[2])
  
  # for each site i in NbSites
  prod_i = rep(NA, NbSites)
  for (i in 1:NbSites) {
    # Was there detections at least once in this site?
    nb_detecs_in_i = sapply(detection_times[[i]], function(x) {
      length(x[!is.na(x)])
    })
    
    # llh_i_occupied is the likelihood of having these data when the site is occupied
    # llh_i_occupied = ψ * λ^di * exp(−λLi)
    # (Guillera-Arroita et al., 2011, part 2.1., 1st equation)
    
    # But first, we calculate exp(−λLi)
    # when the value of λ is highly unlikely, exp(−λLi) = 0 
    #       then llh_i_occupied = 0
    #       then there are -Inf in log(prod_i)
    #       so log_likelihood = -Inf
    #       so we can not run optim.
    # Hence, if  exp(−λLi) = 0 ; i replace this value by the
    # value closest to 0 for the system. So the log_lihekihood will be
    # very low, and it will not cause computationnal issues.
    exp_minuslambda_Li = exp(-lambda_PP * sum(list_T_ij[[i]]))
    if (exp_minuslambda_Li == 0) {
      exp_minuslambda_Li = .Machine$double.xmin
    }
    
    # for the same reason, we calculate λ^di
    lambda_puiss_ndetec = lambda_PP ^ sum(nb_detecs_in_i)
    if (lambda_puiss_ndetec == Inf) {
      lambda_puiss_ndetec = .Machine$double.xmax
    }
    
    # The likelihood of having these data when the site is occupied
    llh_i_occupied <- psi * lambda_puiss_ndetec * exp_minuslambda_Li 
    
    if (sum(nb_detecs_in_i) > 0) {
      # If there were detections
      prod_i[i] = llh_i_occupied
      
    } else {
      # If there were no detections
      prod_i[i] = llh_i_occupied + (1 - psi)
    }
  }
  
  # log_likelihood = log(prod(prod_i))
  # <!> if X are bigs then prod(X) can return Inf and then log(prod(X)) return Inf!!
  # so we write it as its equivalent: sum(log(prod_i))
  log_likelihood = sum(log(prod_i))
  return(log_likelihood)
}

# get_PP_neg_loglikelihood is created only for using
# optimisation methods afterwards.
get_PP_neg_loglikelihood = function(param,
                                    detection_times,
                                    NbSites,
                                    list_T_ij,
                                    list_R_i) {
  "
  get_PP_neg_loglikelihood
  Returns the negative log-likelihood (calculated with get_PP_loglikelihood).
  This function us useful for using optim for optimisation, which works best
  when minimising and not maximising.
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  See the documentation of the function get_PP_loglikelihood
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  The negative log-likelihood of a PP
  "
  
  -get_PP_loglikelihood(param,
                        detection_times,
                        NbSites,
                        list_T_ij,
                        list_R_i)
}

# ──────────────────────────────────────────────────────────────────────────────
# COP COUNTING OCCURRENCES PROCESS                                          ----
# ──────────────────────────────────────────────────────────────────────────────


get_COP_loglikelihood = function(param, SimulatedDataset_NbDetecsPerSession) {
  "
  get_COP_loglikelihood
  Calculates the log-likelihood of observing the data detection_times given
  the COP parameters in param.
  <!> In this function, there are no covariates taken into account.
      Occupancy probability psi and detection rate lambda are constant.
  

  INPUTS ───────────────────────────────────────────────────────────────────────

  param   Vector of parameters:
    psi_transformed       param[1]    Occupancy probability
                                      TRANSFORMATION: logit(psi)
    lambda_transformed    param[2]    Detection rate
                                      TRANSFORMATION: exp(lambda_PP)

  SimulatedDataset_NbDetecsPerSession
    A matrix produced by the function `get_nbdetec_per_session`
    (defined in ./utils/1_detection_simulation.R)

  OUTPUT ───────────────────────────────────────────────────────────────────────

  The value of the log-likelihood of these parameters given these data for a COP
  "
  
  # The probability of observing these data in a site with at least 1 detection
  # is given by the following equation:
  # P_{\psi,\lambda}(Y_i = y_i, y_i \ge 1) =  \psi \frac{ (\sum_{j=1}^n \lambda_{ij} L_{ij})^{y_i}  }{y_i !}e^{\sum_{j=1}^n \lambda_{ij} L_{ij}}
  # 
  # 
  # The probability in a site with no detection is given by:
  # P_{\psi,\lambda}(Y_i = 0) = (1-\psi)  +  \psi e^{\sum_{j=1}^n \lambda_{ij} L_{ij}}

  # Reading parameters
  psi = invlogit(param[1])
  lambda = exp(param[2])
  
  # Extracting informations from data
  S_x_deploy = nrow(SimulatedDataset_NbDetecsPerSession) 
  # <!> this function does not, as is, allow the possibility to calculate
  # the log likelihood if there are several deployments per site.
  n = ncol(SimulatedDataset_NbDetecsPerSession) # number of sessions
  list_T_ij = 0 + !(is.na(SimulatedDataset_NbDetecsPerSession))
  
  # Initialisation of the probability of getting these data for the NbSites sites
  res_loop_i = rep(NA, S_x_deploy)
  
  for (i in 1:S_x_deploy) {
    
    # Total number of observations on the site i
    y_i = sum(SimulatedDataset_NbDetecsPerSession[i, ], na.rm = T)
    
    # Initialisation of the sum of Lambda x L for the n sessions of site i
    sum_lamba_x_L = 0
    
    # Calculation of the sum of Lambda x L in each session
    for (j in 1:n) {
      sum_lamba_x_L = sum_lamba_x_L + lambda * list_T_ij[i,j]
    }
    
    if (y_i == 0) {
      res_loop_i[i] = (1 - psi) + psi * exp(-sum_lamba_x_L)
    } else {
      res_loop_i[i] = psi * (min(.Machine$double.xmax, (sum_lamba_x_L ^ y_i)) / min(.Machine$double.xmax, factorial(y_i))) * exp(-sum_lamba_x_L)
    }
    
    # Why is there `min(.Machine$double.xmax, factorial(y_i))` ?
    # To make optim work on all circumstances.
    # 
    # Example:
    # if y_i >= 171 (on my machine), factorial(y_i) = Inf 
    # so res_loop_i[i] = 0
    # so log(res_loop_i[i]) = -Inf
    # so loglikelihood of these parameters for these values = -Inf
    # so optim doesnt work :(
    # 
    # We want to maximise likelihood, and we don't do anything else with this function.
    # So as long as we have a tiny value for likelihood, it doesn't matter if its an approximation.
  }
  
  loglikelihood = sum(log(res_loop_i))
  
  return(loglikelihood)
}



get_COP_neg_loglikelihood = function(param, SimulatedDataset_NbDetecsPerSession) {
  "
  get_COP_neg_loglikelihood
  Returns the negative log-likelihood (calculated with get_CO_loglikelihood).
  This function us useful for using optim for optimisation, which works best
  when minimising and not maximising.
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  See the documentation of the function get_COP_loglikelihood
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  The negative log-likelihood of a COP
  "
  -get_COP_loglikelihood(param, SimulatedDataset_NbDetecsPerSession)
}

