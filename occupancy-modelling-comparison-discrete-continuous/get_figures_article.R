# ╭─────────────────────────────────────────────────────────────────────────────╮
# │                                                                             │
# │ Analyzing simulation study results                                          │
# │ Léa Pautrel, TerrOïko | CEFE | IRMAR                                        │
# │ Last update: October 2023                                                   │
# │                                                                             │
# ╰─────────────────────────────────────────────────────────────────────────────╯

# ──────────────────────────────────────────────────────────────────────────────
# LIBRARIES                                                                 ----
# ──────────────────────────────────────────────────────────────────────────────

library(stringr)
library(progress)
library(plyr)
library(tidyverse)
library(latex2exp)
library(lubridate)

# ──────────────────────────────────────────────────────────────────────────────
# FUNCTIONS                                                                 ----
# ──────────────────────────────────────────────────────────────────────────────

# Source all other functions
source("./utils/comparison_of_occupancy_models.R")


plot_detection_times_article = function(SimulatedDataset, lambda, mu, N100, param.letter, plot_title_base = "letter") {
  "
  plot_detection_times_article
  Plot examples of the time of detections depending on the detection parameters

  INPUTS ───────────────────────────────────────────────────────────────────────
  SimulatedDataset
    Produced with function simulate_2MMPP
  
  lambda
    Vector of lambda_1, lambda_2
  
  mu
    Vector of mu_12, mu_21
  
  N100
    Expected number of detections per 100 days of deployment
  
  param.letter
    Letter for the detection scenario as in the article
  
  plot_title_base
    (facultative, 'letter' by default)
    The title plot is the letter if 'letter' and a text with lambda, mu, N100 otherwise

  OUTPUT ───────────────────────────────────────────────────────────────────────
  A ggplot
  
  USE ──────────────────────────────────────────────────────────────────────────
  plot_detection_times_article(
    SimulatedDataset = simulate_2MMPP(
      NbSites = 1,
      NbDeployPerSite = 1,
      DeployementTimeValues = 100,
      psi = 1,
      lambda = c(0, 5),
      mu = c(0.0667, 1)
    ),
    lambda =  c(0, 5),
    mu = c(0.0667, 1),
    N100 = 31.26,
    param.letter = '(e)'
  )
  "
  
  if (plot_title_base == "letter") {
    plot_title = param.letter
  } else {
    plot_title = eval(parse(
      text = paste0(
        'expression(paste(lambda[1] == ',
        lambda[1],
        ', ", ", lambda[2] == ',
        lambda[2],
        ', ", ", mu[12] == "',
        mu[1] ,
        ', ", mu[12] == "',
        mu[2] ,
        '", " \t ", bold(E(N["100"])) == bold("',
        N100,
        '") ))'
      )
    ))
  }
  
  main_simul_infos = extract_simulated_infos(SimulatedDataset, quiet = TRUE)
  RecapNDetec = main_simul_infos[["RecapNDetec"]]
  list_R_i = main_simul_infos[["list_R_i"]]
  list_T_ij = main_simul_infos[["list_T_ij"]]
  z_i = main_simul_infos[["z_i"]]
  rm(main_simul_infos)
  invisible(gc())
  
  # DetectionsTimes
  j=1
  if (sum(RecapNDetec$Deployment1) == 0) {
    data_events = data.frame("i" = c(1, 2), "DetectionsTimes" = rep(-10, 2))
  } else {
    data_events = data.frame("i" = rep(NA, sum(RecapNDetec$Deployment1)),
                             "DetectionsTimes" = rep(NA, sum(RecapNDetec$Deployment1)))
    for (i in 1:length(SimulatedDataset)) {
      if ((RecapNDetec$Deployment1[i]) > 0) {
        cpt = (sum(RecapNDetec$Deployment1[1:i - 1]) + 1):sum(RecapNDetec$Deployment1[1:i])
        data_events[cpt, 1] <- i
        data_events[cpt, 2] <-
          SimulatedDataset[[i]][[j]]$DetectionsTimes
      }
    }
  }
  data_events$i <- as_factor(data_events$i)
  
  # States
  for (i in 1:length(SimulatedDataset)) {
    if (i == 1) {
      data_states = SimulatedDataset[[i]][[j]]$HiddenState %>%
        mutate(State = factor(State)) %>% mutate('i' = i) %>% relocate(i)
    } else {
      data_states = rbind(
        data_states,
        SimulatedDataset[[i]][[j]]$HiddenState %>%
          mutate(State = factor(State)) %>% mutate('i' = i) %>% relocate(i)
      )
    }
  }
  data_states$EndTime = c(data_states$BeginningTime[-1], list_T_ij[[i]])
  data_states$EndTime[data_states$EndTime == 0] <- list_T_ij[[i]]
  data_states$i <- as_factor(data_states$i)
  
  # Plot
  detecplot <- ggplot2::ggplot() +
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
    ggplot2::facet_grid(i ~ .) +
    ggplot2::geom_point(
      data = data_events,
      inherit.aes = FALSE,
      ggplot2::aes(y = 1, x = DetectionsTimes, shape= " "),
      color = "black",
      # shape = 4,
      na.rm = TRUE
    ) +
    ggplot2::labs(title = plot_title,
                  fill = "State") +
    scale_x_continuous(limits = c(-5, list_T_ij[[i]]+5), expand = c(0, 0)) +
    theme_minimal() +
    ggplot2::theme(
      axis.text = element_blank(),
      axis.title = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      legend.position = 'bottom',
      legend.background = element_rect(fill = "white", colour = "white"),
      legend.text = element_text(size = 7),
      legend.title = element_text(size = 9),
      plot.title = element_text(hjust = .5, size = 9, margin = margin(0,0,-1,0)),
      strip.text = element_blank()
    ) +
    ggplot2::scale_shape_manual(values = c(' ' = 4), name = "Detection") +
    ggplot2::scale_fill_manual(
      # values = c('1' = 'grey80', '2' = 'grey20'),
      values = c('1' = '#dec357', '2' = '#034742'),
      drop = FALSE
      ) ; detecplot
    # theme(axis.text.x = element_text(size =7),
    #       axis.ticks.x = element_line(),
    # )
  
  return(detecplot)
}

# ──────────────────────────────────────────────────────────────────────────────
# INPUTS                                                                    ----
# ──────────────────────────────────────────────────────────────────────────────

## Result files ----------------------------------------------------------------

result_files_path = "./output/"
result_files = list.files(
  result_files_path,
  pattern = ".*OccModComp_S.*_seed.*_.*.json",
  full.names = T
) 

## Filter of detections_unique_param
## (used at the 1st reading of the jsons)
filterResults = "filter(param.mu_21 != 30) %>% filter(param.seed !=0)"

# ──────────────────────────────────────────────────────────────────────────────
# DATA READING & PREPARATION                                                ----
# ──────────────────────────────────────────────────────────────────────────────

## General variables -----------------------------------------------------------
### From file name --- ---------------------------------------------------------
NbSites = result_files[1] %>% strsplit("/") %>% .[[1]] %>% .[length(.)] %>%
  strsplit("_") %>% .[[1]] %>% .[startsWith(x = ., prefix = "S")] %>% parse_number()
NbDeployPerSite = result_files[1] %>% strsplit("/") %>% .[[1]] %>% .[length(.)] %>%
  strsplit("_") %>% .[[1]] %>% .[startsWith(x = ., prefix = "R")] %>% parse_number()
DeployementTimeValues = result_files[1] %>% strsplit("/") %>% .[[1]] %>% .[length(.)] %>%
  strsplit("_") %>% .[[1]] %>% .[startsWith(x = ., prefix = "T")] %>% parse_number()

seeds_values = result_files %>%
  strsplit('_') %>%
  lapply(function(x) {
  x[startsWith(x = x, prefix = 'seed')] %>%
  str_replace(pattern = 'seed', '') %>% 
  strsplit('-') %>% 
  .[[1]] %>% 
  as.numeric()}) %>% 
  unlist()
seed_min = min(seeds_values)
seed_max = max(seeds_values)

optim_method = result_files[1] %>% strsplit("/") %>% .[[1]] %>% .[length(.)] %>%
  strsplit("_") %>% .[[1]]  %>% .[length(.)-1]

run_suffix = result_files[1] %>%
  strsplit('_') %>%
  .[[1]] %>%
  .[length(.)] %>%
  str_replace(pattern = '.json', replacement = '')

### Output paths ---------------------------------------------------------------

maxDateFile = max(lubridate::as_datetime(file.info(result_files)$mtime))
result_file_prepared = paste0(
  result_files_path, 
  '/OccModComp_S', NbSites,
  '_R', NbDeployPerSite,
  '_T', DeployementTimeValues,
  '_seed', seed_min, '-', seed_max,
  '_', optim_method,
  '_', format(maxDateFile, "%Y-%m-%d--%H:%M:%S"), '.csv'
)

first_time_reading = !file.exists(result_file_prepared)

path_figs_out <-
  paste0(
    dirname(result_files[1]),
    "/FIGURES_",
    'OccModComp_S', NbSites,
    '_R', NbDeployPerSite,
    '_T', DeployementTimeValues,
    '_seed', seed_min, '-', seed_max,
    '_', optim_method,
    '_', run_suffix,
    "/"
  )
if (!dir.exists(path_figs_out)) {
  dir.create(file.path(path_figs_out))
  cat("Created output path:", path_figs_out, fill = T)
}


## First time reading ----------------------------------------------------------
# Also runs if there were updates in the files
if (first_time_reading) {
  print("Reading all jsons...")
  for (result_file in result_files[-1]) {
    # Read parameters from file name
    if (NbSites != result_file %>% strsplit("/") %>% .[[1]] %>% .[length(.)] %>%
        strsplit("_") %>% .[[1]] %>% .[startsWith(x = ., prefix = "S")] %>% parse_number()) {
      stop("Number of sites not homogeneous")
    }
    if (NbDeployPerSite != result_file %>% strsplit("/") %>% .[[1]] %>% .[length(.)] %>%
        strsplit("_") %>% .[[1]] %>% .[startsWith(x = ., prefix = "R")] %>% parse_number()) {
      stop("Number of deployment per site not homogeneous")
    }
    if (DeployementTimeValues != result_file %>% strsplit("/") %>% .[[1]] %>% .[length(.)] %>%
        strsplit("_") %>% .[[1]] %>% .[startsWith(x = ., prefix = "T")] %>% parse_number()) {
      stop("Time of deployments not homogeneous")
    }
    if (optim_method != result_file %>% strsplit("/") %>% .[[1]] %>% .[length(.)] %>%
        strsplit("_") %>% .[[1]] %>% .[length(.)-1]) {
      stop("Optimisation methods not homogeneous")
    }
    
  }
  
  
  ### Read json -------------------------------------------------------------------
  pb = newpb(length(result_files), txt = "json")
  for (result_file in result_files) {
    res_tmp = read_json(result_file,
                        simplifyVector = T) %>%
      Flattener() %>%
      as_tibble() %>%
      mutate_at(vars(matches("param.")), as.numeric) %>%
      mutate_at(vars(matches("CI")), as.complex) %>%
      mutate_at(vars(ends_with("fitting_time")), as.numeric) %>%
      mutate_at(vars(ends_with("psi")), as.numeric) %>%
      mutate_at(vars(ends_with("lambda_1")), as.numeric) %>%
      mutate_at(vars(ends_with("lambda_2")), as.numeric) %>%
      mutate_at(vars(ends_with("mu_12")), as.numeric) %>%
      mutate_at(vars(ends_with("mu_21")), as.numeric) %>%
      mutate_at(vars(ends_with("lambda")), as.numeric) %>%
      mutate_at(vars(ends_with("lambda_Tsession")), as.numeric) %>%
      mutate_at(vars(ends_with("p")), as.numeric)
    
    if (result_file == result_files[1]) {
      results_init = res_tmp
    } else {
      # results_init = rbind(results_init, res_tmp)
      results_init = dplyr::bind_rows(results_init, res_tmp)
    }
    pb$tick()
  }
  
  colnames(results_init) <- colnames(results_init) %>%
    sapply(str_replace_all,
           pattern = "semicontinuous",
           replacement = "COP") %>%
    sapply(str_replace_all,
           pattern = "discrete",
           replacement = "BP")
  if (suppressWarnings(is.null(results_init$COP.lambda_Tsession))) {
    results_init$COP.lambda_Tsession <- results_init$COP.lambda
    results_init$COP.lambda <- results_init$COP.lambda_Tsession / results_init$param.DeployementTimeValues
  } else if (suppressWarnings(is.null(results_init$COP.lambda))) {
    results_init$COP.lambda <- results_init$COP.lambda_Tsession / results_init$param.DeployementTimeValues
  }
  rm(res_tmp)
  invisible(gc())
  
  results_init <- eval(parse(text = paste(
    "results_init %>%", filterResults
  )))
  
  ### Letter, Expected p, Nij --------------------------------------------------
  
  detections_unique_param = unique(
    select(
      results_init,
      param.lambda_1,
      param.lambda_2,
      param.mu_12,
      param.mu_21,
      param.DeployementTimeValues
    )
  )

  detections_unique_param$param.pi_1 <-
    detections_unique_param$param.mu_21 / (detections_unique_param$param.mu_12 + detections_unique_param$param.mu_21)
  detections_unique_param$param.avg_Nij_zi1 <-
    (
      detections_unique_param$param.pi_1 * detections_unique_param$param.lambda_1 + (1 - detections_unique_param$param.pi_1) * detections_unique_param$param.lambda_2
    ) * detections_unique_param$param.DeployementTimeValues
  
  detections_unique_param <- detections_unique_param %>%
    arrange(param.avg_Nij_zi1) %>%
    mutate(param.letter = paste0('(', letters, ')')[1:nrow(.)]) %>% 
    relocate(param.letter)
  
  ### Calculation of p1 -----------------------------------------------------------
  # Calculation of p the detection probability equivalent
  # for 2-MMPP simulation (param.p)
  
  for (i in 1:nrow(detections_unique_param)) {
    # Variance of the number of detections
    CALCUL_var_nbdetec_State1 = detections_unique_param[i, "param.DeployementTimeValues"] *
      (
        detections_unique_param[i, "param.lambda_1"] *
          detections_unique_param[i, "param.pi_1"] +
          2 / (
            detections_unique_param[i, "param.mu_12"] ^ 2 *
              detections_unique_param[i, "param.mu_21"] ^ 2 *
              (1 / detections_unique_param[i, "param.mu_12"] +
                 1 / detections_unique_param[i, "param.mu_21"]) ^ 3
          ) *
          detections_unique_param[i, "param.lambda_1"] ^ 2
      )
    
    CALCUL_var_nbdetec_State2 = detections_unique_param[i, "param.DeployementTimeValues"] *
      (
        detections_unique_param[i, "param.lambda_2"] *
          (1 - detections_unique_param[i, "param.pi_1"]) +
          2 / (
            detections_unique_param[i, "param.mu_12"] ^ 2 *
              detections_unique_param[i, "param.mu_21"] ^ 2 *
              (1 / detections_unique_param[i, "param.mu_12"] +
                 1 / detections_unique_param[i, "param.mu_21"]) ^ 3
          ) *
          detections_unique_param[i, "param.lambda_2"] ^ 2
      )
    
    detections_unique_param[i, "param.var_Nij_zi1"] <-
      CALCUL_var_nbdetec_State1 + CALCUL_var_nbdetec_State2
    
    # Proba of getting at least 1 detection in 1 time unit
    detections_unique_param[i, "param.p_1"] <- get_p_from_2MMPP_param(
      lambda = c(
        detections_unique_param[i, ]$param.lambda_1,
        detections_unique_param[i, ]$param.lambda_2
      ),
      mu = c(
        detections_unique_param[i, ]$param.mu_12,
        detections_unique_param[i, ]$param.mu_21
      ),
      pT = 1
    )
    
  }
  
  print(tibble(detections_unique_param))
  
  
  ### Jointure --------------------------------------------------------------------
  results <- left_join(
    x = results_init,
    y = detections_unique_param,
    by = c(
      "param.lambda_1",
      "param.lambda_2",
      "param.mu_12",
      "param.mu_21",
      "param.DeployementTimeValues"
    )
  ) %>% 
    relocate(param.letter)
  results$param.avg_Nij_zi01 <- results$param.avg_Nij_zi1 * results$param.psi
  
  
  
  # and estimation of p_1 for models
  pb = newpb(nrow(results))
  for (i in 1:nrow(results)) {
    # If there was no discretisation (2-MMPP, IPP, PP)
    if (!(
      results[i,] %>%
      select(
        `twoMMPP.psi`,
        `twoMMPP.lambda_1`,
        `twoMMPP.lambda_2`,
        `twoMMPP.mu_12`,
        `twoMMPP.mu_21`
      ) %>%
      t() %>% as.vector() %>%
      is.na() %>% any()
    )) {
      # p detection probability
      results[i, "twoMMPP.p_1"] = get_p_from_2MMPP_param(
        lambda = c(
          'lambda_1' = results[i,] %>% pull("twoMMPP.lambda_1"),
          'lambda_2' = results[i, ] %>% pull("twoMMPP.lambda_2")
        ),
        mu = c(
          'mu_12' = results[i, ] %>% pull("twoMMPP.mu_12"),
          'mu_21' = results[i, ] %>% pull("twoMMPP.mu_21")
        ),
        pT = 1
      )
      results[i, "IPP.p_1"] = get_p_from_2MMPP_param(
        lambda = c(
          'lambda_1' = results[i,] %>% pull("IPP.lambda_1"),
          'lambda_2' = results[i, ] %>% pull("IPP.lambda_2")
        ),
        mu = c(
          'mu_12' = results[i, ] %>% pull("IPP.mu_12"),
          'mu_21' = results[i, ] %>% pull("IPP.mu_21")
        ),
        pT = 1
      )
      
      
    }
    
    pb$tick()
  }
  
  
  results$COP.p_1 = 1 - exp(-results$COP.lambda)
  results$PP.p_1 = 1 - exp(-results$PP.lambda)
  results$BP.p_1 <- 1 - (1 - results$BP.p) ^ (1 / results$param.SessionLength)
  
  ### Estimated Nij ---------------------------------------------------------------
  # Average number of detections - 2MMPP
  results$twoMMPP.pi_1 <-
    results$twoMMPP.mu_21 / (results$twoMMPP.mu_12 + results$twoMMPP.mu_21)
  results$twoMMPP.avg_Nij_zi1 <-
    (
      results$twoMMPP.pi_1 * results$twoMMPP.lambda_1 + (1 - results$twoMMPP.pi_1) * results$twoMMPP.lambda_2
    ) * results$param.DeployementTimeValues
  results$twoMMPP.avg_Nij_zi01 <-
    results$twoMMPP.avg_Nij_zi1 * results$twoMMPP.psi
  
  # Average number of detections - IPP
  results$IPP.pi_1 <-
    results$IPP.mu_21 / (results$IPP.mu_12 + results$IPP.mu_21)
  results$IPP.avg_Nij_zi1 <-
    (
      results$IPP.pi_1 * results$IPP.lambda_1 + (1 - results$IPP.pi_1) * results$IPP.lambda_2
    ) * results$param.DeployementTimeValues
  results$IPP.avg_Nij_zi01 <-
    results$IPP.avg_Nij_zi1 * results$IPP.psi
  
  # Average number of detections - PP
  results$PP.avg_Nij_zi1 <-
    results$PP.lambda * results$param.DeployementTimeValues
  results$PP.avg_Nij_zi01 <- results$PP.avg_Nij_zi1 * results$PP.psi
  
  # Average number of detections - COP
  results$COP.avg_Nij_zi1 <-
    results$COP.lambda * results$param.DeployementTimeValues
  results$COP.avg_Nij_zi01 <-
    results$COP.avg_Nij_zi1 * results$COP.psi
  
  ### Rename models ---------------------------------------------------------------
  colnames(results) <-
    str_replace(colnames(results),
                pattern = "semicontinuous",
                replacement = "COP")
  colnames(results) <-
    str_replace(colnames(results),
                pattern = "discrete",
                replacement = "BP")
  colnames(results) <-
    str_replace(colnames(results),
                pattern = "twoMMPP",
                replacement = "2-MMPP")
  
  write.table(
    x = results,
    file = result_file_prepared,
    append = F,
    sep = ";",
    quote = T,
    na = "NA",
    dec = ".",
    row.names = F,
    col.names = T
  )
  
} else {
  ## Already prepared reading ----------------------------------------------------
  results <- read.csv(result_file_prepared, header = T, sep = ";", check.names = F)
}

## detections_unique_param ----
detections_unique_param = results %>% 
  dplyr::group_by(
    param.letter,
    param.lambda_1,
    param.lambda_2,
    param.mu_12,
    param.mu_21,
    param.pi_1,
    # param.p,
    param.p_1,
    param.avg_Nij_zi1,
    param.var_Nij_zi1,
  ) %>%
  dplyr::summarise(param.p_100 = mean(unique(param.p)), NbComp = n()) %>%
  arrange(param.avg_Nij_zi1)

detections_unique_param

# get values for session length for which no continuous models run
no_continuous_SessionLength = results %>% 
  dplyr::group_by(param.SessionLength) %>% 
  dplyr::summarise(notNA = sum(!is.na(`2-MMPP.psi`)),
                   .groups = 'keep') %>% 
  filter(notNA == 0) %>% 
  pull(param.SessionLength)

## Pivot for ggplot ------------------------------------------------------------
# Format data as readable for ggplot and data visualisations afterwards
results_for_visu_init <- results %>%
  as_tibble() %>%
  # pivot estimation of parameters by model to make it readable by ggplot
  pivot_longer(
    cols = -starts_with("param"),
    names_to = c("model", ".value"),
    names_sep = "\\."
  ) %>% 
  dplyr::mutate(
    
    # Add column that describes the session length as human understandable
    param.SessionLength.Text = factor(
      ifelse(
        model %in% c("2-MMPP", "IPP", "PP"),
        "No discretisation",
        SessionLength_to_text(param.SessionLength)
      ),
      ordered = T,
      levels = c("No discretisation", SessionLength_to_text(unique(results$param.SessionLength)))
    ),
    
    # Change order for models
    model = factor(
      model,
      ordered = T,
      levels = c("BP", "COP", "PP", "IPP", "2-MMPP")
    ),
    
    param.detection = factor(format(round(
      param.avg_Nij_zi1, 2
    ), nsmall = 2)),
    
    
    model.session = factor(
      ifelse(
        model %in% c('BP', 'COP'),
        paste0(model, " (", param.SessionLength.Text, ")"),
        as.character(model)
      ),
      levels = c(
        "BP (Month)",
        "BP (Week)",
        "BP (Day)",
        "COP (Month)",
        "COP (Week)",
        "COP (Day)",
        "PP",
        "IPP",
        "2-MMPP"
      ),
      ordered = TRUE
    ),
    
  ) %>% 
  filter(
    # Remove lines with all NA
    !is.na(model),
    # Remove lines of continuous models when they did not ran
    !(model %in% c("2-MMPP", "IPP", "PP") & param.SessionLength %in% no_continuous_SessionLength),
  )

## Nb of comparisons ----

# Header of comparisons with the less number of comparisons
results_for_visu_init %>% 
  group_by(param.psi, param.letter, param.SessionLength.Text, model) %>% 
  dplyr::summarise(
    NbComp = sum(!is.na(psi)),
    NbNA = sum(is.na(psi)),
    Tot = n(),
    .groups = "keep"
  ) %>%
  arrange(NbComp)

# With NA (that are due to nb detec total = 0)
results_for_visu_init %>% 
  group_by(param.psi, param.letter, param.SessionLength.Text, model) %>% 
  dplyr::summarise(NbComp = n(), .groups = "keep") %>% 
  pull(NbComp) %>% 
  table()

# Without NA
results_for_visu_init %>% 
  group_by(param.psi, param.letter, param.SessionLength.Text, model) %>% 
  filter(!is.na(psi)) %>% 
  dplyr::summarise(NbComp = n(), .groups = "keep") %>% 
  pull(NbComp) %>% 
  table()

# See detail of parameters without 500 comparisons
results_for_visu_init %>% 
  group_by(param.psi, param.letter, param.SessionLength.Text, model) %>% 
  dplyr::summarise(
    NbComp = sum(!is.na(psi)),
    NbNA = sum(is.na(psi)),
    Tot = n(),
    .groups = "keep"
  ) %>%
  filter(NbComp < 500) %>% 
  arrange(NbComp) %>% 
  print(n = 100) 


# Remove NA due to 0 detections in the simulated dataset
# <!> IF NbNA != 0 SOMEWHERE THERE'S AN ISSUE
results_for_visu <- results_for_visu_init %>%
  filter(param.nb_detec_total > 0)
results_for_visu %>% 
  group_by(param.psi, param.letter, param.SessionLength.Text, model) %>% 
  dplyr::summarise(NbComp = sum(!is.na(psi)), NbNA = sum(is.na(psi))) %>% 
  filter(NbComp < 500) %>% 
  arrange(NbComp) %>% 
  print(n = 100) 

rm(results_for_visu_init)
gc()




# ──────────────────────────────────────────────────────────────────────────────
# SIMULATED DATASETS                                                        ----
# ──────────────────────────────────────────────────────────────────────────────

## Example plots ---------------------------------------------------------------
simul_scenario_plots = vector(mode = "list", length = nrow(detections_unique_param))
set.seed(23) # to get the same plot simulated each time
for (simul_scenario in 1:nrow(detections_unique_param)) {
  lambda = as.character(
    c(
      detections_unique_param[simul_scenario, ]$param.lambda_1,
      detections_unique_param[simul_scenario, ]$param.lambda_2
    )
  )
  mu = c(
    paste0("1/", round(
      1 / detections_unique_param[simul_scenario,]$param.mu_12
    )),
    ifelse(
      !(round(1 / detections_unique_param[simul_scenario,]$param.mu_21) %in% c(0,1)),
      paste0("1/", round(
        1 / detections_unique_param[simul_scenario,]$param.mu_21
      )),
      detections_unique_param[simul_scenario,]$param.mu_21
    )
  )
  N100 = format(round(detections_unique_param[simul_scenario, ]$param.avg_Nij_zi1, 2),
                nsmall = 2)
  param.letter = detections_unique_param[simul_scenario, ]$param.letter
  
  simul_scenario_plot = plot_detection_times_article(
    SimulatedDataset = simulate_2MMPP(
      NbSites = 2,
      NbDeployPerSite = 1,
      DeployementTimeValues = unique(results$param.DeployementTimeValues),
      psi = 1,
      lambda = c(eval(parse(text = lambda[1])), eval(parse(text = lambda[2]))),
      mu = c(eval(parse(text = mu[1])), eval(parse(text = mu[2])))
    ),
    lambda = lambda,
    mu = mu,
    N100 = N100,
    param.letter = param.letter
  )
  simul_scenario_plots[[simul_scenario]] <- simul_scenario_plot
}

simul_plots_all_scenarios = ggpubr::ggarrange(
  plotlist = c(
    simul_scenario_plots[-length(simul_scenario_plots)], 
    list(
      simul_scenario_plots[[length(simul_scenario_plots)]] +
        theme(axis.text.x = element_text(size =7),
              axis.ticks.x = element_line(),
              )
      )
    ),
  ncol = 1,
  heights = c(rep(1, length(simul_scenario_plots)-1), 1.1),
  common.legend = TRUE
)
print(simul_plots_all_scenarios)

ggsave(
  paste0(path_figs_out, "simulated_dataset_examples_", optim_method, ".pdf"),
  plot = simul_plots_all_scenarios,
  width =  22,
  height = 12.5,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "simulated_dataset_examples_", optim_method, ".jpeg"),
  width = 22,
  height = 12.5,
  unit = "cm",
  res = 100
)
print(simul_plots_all_scenarios)
dev.off()

set.seed(NULL) # reinitialize seed

## Get informations ------------------------------------------------------------
if (!interactive()) {
  pdf(NULL)
}

results %>%
  ggplot() +
  geom_histogram(aes(x = param.nb_detec_total, y = after_stat(count)), bins=30) +
  facet_wrap(~ glue::glue("psi: {param.psi}; detection: {param.letter}"), scales = "free", nrow = 5) +
  labs(x = "Total number of detections")

hist(results$param.nb_detec_total)
summary(results$param.nb_detec_total)

if(!is.null(results$param.nb_detec_average_per_deployment_when_occupied)) {
  summary(results$param.nb_detec_average_per_deployment_when_occupied)
  
  results %>%
    group_by(param.avg_Nij_zi1, param.psi) %>%
    dplyr::summarise(
      min_N100 = min(param.nb_detec_average_per_deployment_when_occupied),
      max_N100 = max(param.nb_detec_average_per_deployment_when_occupied)
    )
}

if (!interactive()) {
  dev.off()
}

# ──────────────────────────────────────────────────────────────────────────────
# TABLES                                                                    ----
# ──────────────────────────────────────────────────────────────────────────────

# Equivalent of table 3 of Guillera-Arroita et al. (2011)
df_estimated_psi <- results %>%
  filter(!is.na(`2-MMPP.psi`)) %>%
  dplyr::mutate(
    # param.detection = factor(
    #   glue::glue(
    #     "λ1={param.lambda_1}, λ2={param.lambda_2}, µ12={param.mu_12}, µ21={param.mu_21}"
    #   )
    # ),
    
    "λ, μ" = factor(param.letter)
  ) %>%
  rowwise() %>%
  dplyr::group_by(param.psi, `λ, μ`, param.avg_Nij_zi1) %>%
  dplyr::summarise(
    BP.psi = glue::glue(
      "{round(mean(BP.psi), 2)} [{round(calcul_rmse(pred = BP.psi, true = param.psi), 3)}]"
    ),
    COP.psi = glue::glue(
      "{round(mean(COP.psi), 2)} [{round(calcul_rmse(pred = COP.psi, true = param.psi), 3)}]"
    ),
    PP.psi = glue::glue(
      "{round(mean(PP.psi), 2)} [{round(calcul_rmse(pred = PP.psi, true = param.psi), 3)}]"
    ),
    IPP.psi = glue::glue(
      "{round(mean(IPP.psi), 2)} [{round(calcul_rmse(pred = IPP.psi, true = param.psi), 3)}]"
    ),
    `2-MMPP.psi` = glue::glue(
      "{round(mean(`2-MMPP.psi`), 2)} [{round(calcul_rmse(pred = `2-MMPP.psi`, true = param.psi), 3)}]"
    ),
    # twoMMPP.avg_Nij_zi1 = glue::glue(
    #   "{round(mean(twoMMPP.avg_Nij_zi1), 2)} [{round(calcul_mse(pred = twoMMPP.avg_Nij_zi1, true = param.avg_Nij_zi1), 3)}]"
    # ),
    .groups = "keep"
  ) %>% 
  arrange(param.psi, param.avg_Nij_zi1)
sink(paste0(path_figs_out, "PSI--rmse_table.md"))
cat(knitr::kable(df_estimated_psi, digits = 2, align = ('c')) %>% paste(collapse = "\n"))
sink()

# ──────────────────────────────────────────────────────────────────────────────
# df_psi_metrics                                                            ----
# ──────────────────────────────────────────────────────────────────────────────

df_psi_metrics = results_for_visu %>%
  mutate(param.letter = factor(
    param.letter,
    labels = sort(unique(param.letter), decreasing = T),
    levels = sort(unique(param.letter), decreasing = T),
    ordered = T
  )) %>% 
  dplyr::filter(!is.na(psi)) %>%
  dplyr::group_by(model,
                  param.SessionLength.Text,
                  param.avg_Nij_zi1,
                  param.psi,
                  param.letter) %>%
  dplyr::mutate(
    psi_CI_range = abs(psi_upper_95CI - psi_lower_95CI),
    psi.AB = (psi - param.psi),
    psi.RB = ((psi - param.psi) / param.psi),
    psi.within.95CI = ifelse((
      !is.na(psi_lower_95CI) &
        !is.na(psi_upper_95CI) &
        Im(psi_lower_95CI) == 0 & Im(psi_upper_95CI) == 0
    ),
    (param.psi >= Re(psi_lower_95CI) & param.psi <= Re(psi_upper_95CI)),
    NA)
  ) %>%
  dplyr::summarise(
    # Number of simulations per row
    NbSimul = n(),
    nb_psi_CI = sum(!is.na(psi_lower_95CI)),
    
    # General descriptive statistics
    psi.avg = mean(psi),
    psi.median = median(psi),
    
    # Point estimates: error and bias
    psi.pointestimate.CI.lower = quantile(psi, 0.025),
    psi.pointestimate.CI.upper = quantile(psi, 0.975),
    psi.rmse = calcul_rmse(pred = psi, true = param.psi),
    psi.AB.avg = mean(psi.AB),
    psi.RB.avg = mean(psi.RB),
    
    # Coverage
    psi.coverage_all = mean(psi.within.95CI, na.rm = T),
    psi.coverage.NbSimul = sum(!is.na(psi.within.95CI)),
    psi.coverage.NbNA = sum(is.na(psi.within.95CI)),
    
    # CI
    psi.CI.lower.avg = mean(Re(psi_lower_95CI), na.rm = T),
    psi.CI.upper.avg = mean(Re(psi_upper_95CI), na.rm = T),
    psi.CI.range.avg_all = mean(psi_CI_range, na.rm = T),
    
    .groups = "keep"
  ) %>%
  dplyr::mutate(
    # bias
    psi.avg.AB = (psi.avg - param.psi),
    psi.avg.RB = ((psi.avg - param.psi) / param.psi),

    # remove coverage/range when < 100 results
    psi.CI.range.avg = ifelse(nb_psi_CI >= 100, psi.CI.range.avg_all, NA),
    psi.coverage = ifelse(nb_psi_CI >= 100, psi.coverage_all, NA),
    
    # Unique column for models + session length for discrete models
    model.session = factor(
      ifelse(
        model %in% c('BP', 'COP'),
        paste0(model, " (", param.SessionLength.Text, ")"),
        as.character(model)
      ),
      levels = c(
        "BP (Month)",
        "BP (Week)",
        "BP (Day)",
        "COP (Month)",
        "COP (Week)",
        "COP (Day)",
        "PP",
        "IPP",
        "2-MMPP"
      ),
      ordered = TRUE
    ),
    
    # param.detection as the expected number of detections in occupied sites
    param.detection = factor(format(round(
      param.avg_Nij_zi1, 2
    ), nsmall = 2))
  )


tmp = results_for_visu %>%
  mutate(psi_CI_range = abs(psi_upper_95CI - psi_lower_95CI)) %>%
  group_by(param.letter, param.psi, model, param.SessionLength.Text) %>%
  summarise(
    psi_lower_95CI_avg = mean(Re(psi_lower_95CI), na.rm = T),
    psi_upper_95CI_avg = mean(Re(psi_upper_95CI), na.rm = T),
    range_95CI_avg_all = mean(psi_CI_range, na.rm = T),
    nb_tot = n(),
    nb_psi_CI = sum(!is.na(psi_lower_95CI))
  ) %>%
  mutate(range_95CI_avg = ifelse(nb_psi_CI >= 100, range_95CI_avg_all, NA))


df_psi_metrics %>% 
  mutate(tmp = psi.AB.avg == psi.avg.AB) %>%
  pull(tmp) %>% 
  table()

df_psi_metrics %>% 
  mutate(tmp = round(psi.AB.avg,3) == round(psi.avg.AB,3)) %>%
  pull(tmp) %>% 
  table()

df_psi_metrics$psi.label <- paste0("' '*widehat(psi) * {phantom() == phantom()} * ",
                                      format(round(df_psi_metrics$psi.avg, 2), nsmall = 2),
                                      "*' \n ' * '[' *'",
                                      format(round(df_psi_metrics$psi.pointestimate.CI.lower, 2), nsmall=2),", ",
                                      format(round(df_psi_metrics$psi.pointestimate.CI.upper,2),nsmall=2),"' * ']'"
)

# TeX(input = r"( $\widehat{\psi} = 1$ )", output = "character")
df_psi_metrics$psi.avg.label <- paste0(
  "' '*widehat(psi) * {phantom() == phantom()} * \"",
  format(round(df_psi_metrics$psi.avg, 2), nsmall = 2),
  "\"*' '"
)
df_psi_metrics$psi.avg.label <- paste0(
  "' '*bar(widehat(psi)) * {phantom() == phantom()} * \"",
  format(round(df_psi_metrics$psi.avg, 2), nsmall = 2),
  "\"*' '"
)

write.csv(df_psi_metrics, paste0(path_figs_out, "PSI--metrics_table.csv"))

## Values used in the paper ----------------------------------------------------

# RMSE & bias for all models per detection parameters
df_psi_metrics %>% 
  dplyr::mutate(param.letter.grouped = ifelse(
    param.letter %in% c('(g)', '(f)', '(e)', '(d)'),
    '(d, e, f, g)',
    as.character(param.letter)
  )) %>%
  dplyr::group_by(param.letter.grouped) %>% 
  dplyr::summarise(
    psi.RMSE.avg = mean(psi.rmse),
    psi.RMSE.min = min(psi.rmse),
    psi.RMSE.max = max(psi.rmse),
    
    psi.AB.avg.avg = mean(psi.AB.avg),
    psi.AB.abs.min = min(abs(psi.AB.avg)),
    psi.AB.min = min(psi.AB.avg),
    psi.AB.max = max(psi.AB.avg),
    
    .groups='keep'
  )


# RMSE and bias for BP & (c)
df_psi_metrics %>% 
  mutate(model.grouped=ifelse(model=="BP","BP", "not BP")) %>% 
  filter(param.letter == '(c)') %>% 
  dplyr::group_by(param.letter, model.grouped) %>% 
  dplyr::summarise(
    psi.RMSE.avg = mean(psi.rmse),
    psi.RMSE.min = min(psi.rmse),
    psi.RMSE.max = max(psi.rmse),
    
    psi.AB.avg.avg = mean(psi.AB.avg),
    psi.AB.abs.min = min(abs(psi.AB.avg)),
    psi.AB.min = min(psi.AB.avg),
    psi.AB.max = max(psi.AB.avg),
    
    .groups='keep'
  )


# Median differences between BP day and the others with (e)&.1; (e)&.9; (f)&.9
df_psi_metrics %>% 
  mutate(param.psi.letter = paste(param.psi, param.letter)) %>% 
  filter(param.letter == "(e)" & param.psi == .1 |
           param.letter == '(e)' &
           param.psi == .9 | param.letter == "(f)" & param.psi == .9) %>% 
  ungroup() %>% 
  select(model.session, param.psi.letter, psi.median) %>% 
  pivot_wider(names_from = model.session,values_from = psi.median)


# ──────────────────────────────────────────────────────────────────────────────
# TILE PLOT BIAS                                                            ----
# ──────────────────────────────────────────────────────────────────────────────

## Colors ----------------------------------------------------------------------

fill_colours = c(
  "chocolate4", "chocolate", "darkgoldenrod3", "goldenrod", "goldenrod1", "gold",
  "#bee8b0",
  "#3975e5", "#4a5fcb", "#524ab0","#543595","#521e7b", "#4d0060"
)
fill_colours = c(
  "#b1631d", "#c2761c", "#d18a19", "#df9f16", "#eab513", "#f4cc13",
  "#bee8b0",
  "#5c8de9", "#5f7cdc", "#636bce","#6759bf","#6a47ae", "#6d329c"
)


AB_max = ceiling(max(abs(df_psi_metrics$psi.AB.avg)) / .25) * .25
fill_values_AB = seq(from = -AB_max, to = AB_max, length.out = length(fill_colours))
fill_txt_AB = round(fill_values_AB, 2)
fill_txt_AB[seq(2, length(fill_txt_AB), by = 2)] <- ""

RB_max = ceiling(max(abs(df_psi_metrics$psi.RB.avg)) / .25) * .25
fill_values_RB = seq(from = -RB_max, to = RB_max, length.out = length(fill_colours))
fill_txt_RB = round(fill_values_RB, 2)
fill_txt_RB[seq(2, length(fill_txt_RB), by = 2)] <- ""



## Absolute bias (AB) ----------------------------------------------------------

### facet_top = model ----------------------------------------------------------
# Discrete plot
plot_AB.discrete = df_psi_metrics %>%
  filter(model %in% c("BP", "COP")) %>%
  ggplot(
    aes(x = param.SessionLength.Text, y = param.letter, fill = psi.AB.avg)
  ) +
  geom_tile() +
  geom_text(
    aes(
      label = psi.avg.label
    ),
    color = "black",
    parse = T,
    size = 2.7
  ) +
  labs(
    title = "Discrete models",
    x = "Session length for discretisation",
    y = "Simulation parameters of rarity and detectability",
    fill = parse(text=latex2exp::TeX(r"( Absolute bias $\bar{ \left( \widehat{\psi} - \psi \right) }$ )", output = "character"))
  ) +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed,
    switch = "y"
  ) +
  theme_minimal() +
  theme(
    strip.background.y = element_rect(fill = "gray93", colour = "white"),
    strip.text.y = element_text(colour = "black", face = "bold"),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    axis.text.y = element_text(hjust=.5),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  scale_x_discrete(position = "top") +
  scale_fill_gradientn(
    colours = fill_colours,
    breaks = fill_values_AB,
    labels = fill_txt_AB,
    limits = c(-AB_max, AB_max)
  ) +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 30,
      label.position = "bottom",
      title.position = "top"
    )) #; print(plot_AB.discrete)


plot_AB.continuous = df_psi_metrics %>%
  filter(model %in% c("2-MMPP", "IPP", "PP")) %>%
  mutate(param.SessionLength.Text = "") %>%
  ggplot(
    aes(x = param.SessionLength.Text, y = param.letter, fill = psi.AB.avg)
  ) +
  geom_tile() +
  geom_text(
    aes(
      label = psi.avg.label
    ),
    color = "black",
    parse = T,
    size = 2.7
  ) +
  labs(
    title = "Continuous models",
    x = "",
    y = "Simulation parameters of rarity and detectability",
    fill = parse(text=latex2exp::TeX(r"( Absolute bias $\bar{ \left( \widehat{\psi} - \psi \right) }$ )", output = "character"))
  ) +
  theme_minimal() +
  theme(
    strip.background.y = element_blank(),
    strip.text.y = element_blank(),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    axis.text.y = element_blank(),
    axis.title.y = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  scale_x_discrete(position = "top") +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed,
    switch = "y"
  ) +
  scale_fill_gradientn(
    colours = fill_colours,
    breaks = fill_values_AB,
    labels = fill_txt_AB,
    limits = c(-AB_max, AB_max)
  ) +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 30,
      label.position = "bottom",
      title.position = "top"
    ))

plot_AB = ggpubr::ggarrange(plot_AB.discrete, plot_AB.continuous,
                                   common.legend = TRUE,
                                   legend = "bottom",
                                   ncol = 2,
                                   widths = c(1, 0.7))+
  theme(plot.background = element_rect(fill = "white"))
print(plot_AB)

ggsave(
  paste0(
    path_figs_out,
    "PSI--bias_absolute_tileplot_topmod_", optim_method, ".pdf"
  ),
  plot = plot_AB,
  width =  18,
  height = 20,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--bias_absolute_tileplot_topmod_", optim_method, ".jpeg"),
  width = 18,
  height = 20,
  unit = "cm",
  res = 100
)
print(plot_AB)
dev.off()

### y = model ------------------------------------------------------------------

plot_AB_ymod = df_psi_metrics %>%
  ggplot(aes(
    x = 1,
    y = model.session,
    fill = psi.AB.avg
  )) +
  geom_tile(colour='white') +
  facet_grid(
    reorder(
      x = factor(
        param.avg_Nij_zi1,
        levels = sort(unique(param.avg_Nij_zi1)),
        labels = paste0("N[100] == ", sort(unique(param.detection))),
        ordered = T
      ),
      X = param.avg_Nij_zi1,
      decreasing = T
    )
    # reorder(
    #   x = factor(
    #     param.avg_Nij_zi1,
    #     levels = sort(unique(param.avg_Nij_zi1)),
    #     labels = paste0(sort(unique(param.letter))),
    #     ordered = T
    #   ),
    #   X = param.avg_Nij_zi1,
    #   decreasing = T
    # )
    ~
      factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
    labeller = label_parsed
  ) +
  scale_fill_gradientn(
    colours = fill_colours,
    breaks = fill_values_AB,
    labels = fill_txt_AB,
    limits = c(-AB_max, AB_max)
  ) +
  labs(
    fill = parse(text=latex2exp::TeX(r"( Absolute bias $\bar{ \left( \widehat{\psi} - \psi \right) }$ )", output = "character"))
  ) +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 20,
      label.position = "bottom",
      title.position = "top"
    )) +
  theme_minimal() +
  theme(
    strip.background.x = element_rect(fill = "gray93", color = "white"),
    strip.background.y = element_rect(fill = "gray80", color = "white"),
    strip.text = element_text(colour = "black", face = "bold"),
    axis.title.y = element_blank(),
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    plot.title = element_blank()#element_text(hjust = 0.5)
  )
print(plot_AB_ymod)

ggsave(
  paste0(
    path_figs_out,
    "PSI--bias_absolute_tileplot_ymod_", optim_method, ".pdf"
  ),
  plot = plot_AB_ymod,
  width =  13,
  height = 25,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--bias_absolute_tileplot_ymod_", optim_method, ".jpeg"),
  width = 13,
  height = 25,
  unit = "cm",
  res = 100
)
print(plot_AB_ymod)
dev.off()






## Relative bias (RB) ----------------------------------------------------------

### facet_top = model ----------------------------------------------------------
# Discrete plot
plot_RB.discrete = df_psi_metrics %>%
  filter(model %in% c("BP", "COP")) %>%
  ggplot(
    aes(x = param.SessionLength.Text, y = param.letter, fill = psi.RB.avg)
  ) +
  geom_tile() +
  geom_text(
    aes(
      label = psi.avg.label
    ),
    color = "black",
    parse = T,
    size = 2.7
  ) +
  labs(
    title = "Discrete models",
    x = "Session length for discretisation",
    y = "Simulation parameters of rarity and detectability",
    fill = parse(text=latex2exp::TeX(r"( Relative bias $\bar{ \left( \frac{\widehat{\psi} - \psi}{\psi} \right) }$ )", output = "character"))
  ) +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed,
    switch = "y"
  ) +
  theme_minimal() +
  theme(
    strip.background.y = element_rect(fill = "gray93", colour = "white"),
    strip.text.y = element_text(colour = "black", face = "bold"),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    axis.text.y = element_text(hjust=.5),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  scale_x_discrete(position = "top") +
  scale_fill_gradientn(
    colours = fill_colours,
    breaks = fill_values_RB,
    labels = fill_txt_RB,
    limits = c(-RB_max, RB_max)
  ) +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 30,
      label.position = "bottom",
      title.position = "top"
    )) #; print(plot_RB.discrete)


plot_RB.continuous = df_psi_metrics %>%
  filter(model %in% c("2-MMPP", "IPP", "PP")) %>%
  mutate(param.SessionLength.Text = "") %>%
  ggplot(
    aes(x = param.SessionLength.Text, y = param.letter, fill = psi.RB.avg)
  ) +
  geom_tile() +
  geom_text(
    aes(
      label = psi.avg.label
    ),
    color = "black",
    parse = T,
    size = 2.7
  ) +
  labs(
    title = "Continuous models",
    x = "",
    y = "Simulation parameters of rarity and detectability",
    fill = parse(text=latex2exp::TeX(r"( Relative bias $\bar{ \left( \frac{\widehat{\psi} - \psi}{\psi} \right) }$ )", output = "character"))
  ) +
  theme_minimal() +
  theme(
    strip.background.y = element_blank(),
    strip.text.y = element_blank(),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    axis.text.y = element_blank(),
    axis.title.y = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  scale_x_discrete(position = "top") +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed,
    switch = "y"
  ) +
  scale_fill_gradientn(
    colours = fill_colours,
    breaks = fill_values_RB,
    labels = fill_txt_RB,
    limits = c(-RB_max, RB_max)
  ) +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 30,
      label.position = "bottom",
      title.position = "top"
    ))

plot_RB = ggpubr::ggarrange(plot_RB.discrete, plot_RB.continuous,
                            common.legend = TRUE,
                            legend = "bottom",
                            ncol = 2,
                            widths = c(1, 0.7))+
  theme(plot.background = element_rect(fill = "white"))
print(plot_RB)

ggsave(
  paste0(
    path_figs_out,
    "PSI--bias_relative_tileplot_topmod_", optim_method, ".pdf"
  ),
  plot = plot_RB,
  width =  18,
  height = 20,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--bias_relative_tileplot_topmod_", optim_method, ".jpeg"),
  width = 18,
  height = 20,
  unit = "cm",
  res = 100
)
print(plot_RB)
dev.off()

### y = model ------------------------------------------------------------------

plot_RB_ymod = df_psi_metrics %>%
  ggplot(aes(
    x = 1,
    y = model.session,
    fill = psi.RB.avg
  )) +
  geom_tile(colour='white') +
  facet_grid(
    reorder(
      x = factor(
        param.avg_Nij_zi1,
        levels = sort(unique(param.avg_Nij_zi1)),
        labels = paste0("N[100] == ", sort(unique(param.detection))),
        ordered = T
      ),
      X = param.avg_Nij_zi1,
      decreasing = T
    )
    # reorder(
    #   x = factor(
    #     param.avg_Nij_zi1,
    #     levels = sort(unique(param.avg_Nij_zi1)),
    #     labels = paste0(sort(unique(param.letter))),
    #     ordered = T
    #   ),
    #   X = param.avg_Nij_zi1,
    #   decreasing = T
    # )
    ~
      factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
    labeller = label_parsed,
  ) +
  scale_fill_gradientn(
    colours = fill_colours,
    breaks = fill_values_RB,
    labels = fill_txt_RB,
    limits = c(-RB_max, RB_max)
  ) +
  labs(
    fill = parse(text=latex2exp::TeX(r"( Relative bias $\bar{ \left( \frac{\widehat{\psi} - \psi}{\psi} \right) }$ )", output = "character"))
  ) +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 20,
      label.position = "bottom",
      title.position = "top"
    )) +
  theme_minimal() +
  theme(
    strip.background.x = element_rect(fill = "gray93", color = "white"),
    strip.background.y = element_rect(fill = "gray80", color = "white"),
    strip.text = element_text(colour = "black", face = "bold"),
    axis.title.y = element_blank(),
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    plot.title = element_blank()#element_text(hjust = 0.5)
  )
print(plot_RB_ymod)

ggsave(
  paste0(
    path_figs_out,
    "PSI--bias_relative_tileplot_ymod_", optim_method, ".pdf"
  ),
  plot = plot_RB_ymod,
  width =  13,
  height = 25,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--bias_relative_tileplot_ymod_", optim_method, ".jpeg"),
  width = 13,
  height = 25,
  unit = "cm",
  res = 100
)
print(plot_RB_ymod)
dev.off()



# ──────────────────────────────────────────────────────────────────────────────
# TILE PLOT RMSE                                                            ----
# ──────────────────────────────────────────────────────────────────────────────

## Colors, percentiles ---------------------------------------------------------


######## Colours without percentiles
fill_colours_all = c(
  "0.0" = "#1f5115",
  "0.1" = "#3ea22a",
  "0.2" = "#dcde98",
  "0.3" = "#E2C870",
  "0.4" = "#e0ae36",
  "0.5" = "#f7910c",
  "0.6" = "#E05407",
  "0.7" = "#da1e1e",
  "0.8" = "#901414",
  "0.9" = "#700f0f",
  "1.0" = "#300707"
)
fill_colours = fill_colours_all[1:(ceiling(max(df_psi_metrics$psi.rmse) * 10) + 1)]
fill_values = as.numeric(names(fill_colours))
fill_txt = as.numeric(names(fill_colours))
df_psi_metrics$psi.rmse.norm = df_psi_metrics$psi.rmse * 1 / max(fill_values)

######## Colours with percentiles
fill_colours_percentiles = c("darkgreen","chartreuse3", "darkolivegreen2", 
                 "darkorange", "firebrick")
# "chartreuse4", "chartreuse3", "darkolivegreen2",
# "khaki", "gold", "darkorange", "firebrick", "darkred"
fill_values_percentiles = c(0, .25, .5, .75, 1)


# Quantiles values
quantiles_val <- quantile(df_psi_metrics$psi.rmse, fill_values_percentiles, na.rm=T)
# Get percentile of mse to fill the tiles
df_psi_metrics$psi.rmse.prctl <- ecdf(df_psi_metrics$psi.rmse)(df_psi_metrics$psi.rmse)
# Percentile text: nth + value
percentile_txt = c('Minimum', 
                  paste0(fill_values_percentiles[2:(length(fill_values_percentiles)-1)]*100, "th percentile"), 
                  'Maximum')
# fill_txt = paste(percentile_txt, format(quantiles_val, digits = 3, scientific = T), sep = '\n')
fill_txt_percentiles = paste(percentile_txt, format(round(quantiles_val, 3), nsmall = 3), sep = '\n')



# Text label: RMSE
df_psi_metrics$psi.rmse.label <- formatC(df_psi_metrics$psi.rmse,
                                         digits = 3,
                                         format = 'f')


# Text labels: 95% confidence intervals of psi, with and without the number of simulations
df_psi_metrics$psi.rmse.CI.label <- paste0("[",
                                        format(round(df_psi_metrics$psi.pointestimate.CI.lower, 2), nsmall = 2),
                                        ", ",
                                        format(round(df_psi_metrics$psi.pointestimate.CI.upper, 2), nsmall = 2),
                                        "]"
)
df_psi_metrics$psi.rmse.CI.n.label <- paste0(
  "' '*widehat(psi) %in% '[",
  format(round(df_psi_metrics$psi.pointestimate.CI.lower, 2), nsmall = 2),
  ", ",
  format(round(df_psi_metrics$psi.pointestimate.CI.upper, 2), nsmall = 2),
  "] (n=",
  df_psi_metrics$NbSimul,
  ")'*' '"
)

## RMSE percentile -------------------------------------------------------------

### facet_top = model ----------------------------------------------------------
# Discrete plot
plot_RMSE.discrete = df_psi_metrics %>%
  filter(model %in% c("BP", "COP")) %>%
  ggplot(
    aes(x = param.SessionLength.Text, y = param.letter, fill = psi.rmse.prctl)
  ) +
  geom_tile() +
  geom_text(
    aes(
      label = psi.rmse.label
    ),
    color = "black",
    parse = F,
    size = 2.7
  ) +
  labs(
    title = "Discrete models",
    x = "Session length for discretisation",
    y = "Simulation parameters of rarity and detectability",
    fill = "RMSE"
  ) +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed,
    switch = "y"
  ) +
  theme_minimal() +
  theme(
    strip.background.y = element_rect(fill = "gray93", colour = "white"),
    strip.text.y = element_text(colour = "black", face = "bold"),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    axis.text.y = element_text(hjust=.5),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  scale_x_discrete(position = "top") +
  scale_fill_gradientn(
    colours = fill_colours_percentiles,
    breaks = fill_values_percentiles,
    labels = fill_txt_percentiles,
    limits = c(0, 1)
  ) +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 30,
      label.position = "bottom",
      title.position = "top"
    )) #; print(plot_RMSE.discrete)


plot_RMSE.continuous = df_psi_metrics %>%
  filter(model %in% c("2-MMPP", "IPP", "PP")) %>%
  mutate(param.SessionLength.Text = "") %>%
  ggplot(
    aes(x = param.SessionLength.Text, y = param.letter, fill = psi.rmse.prctl)
  ) +
  geom_tile() +
  geom_text(
    aes(
      label = psi.rmse.label
    ),
    color = "black",
    parse = F,
    size = 2.7
  ) +
  labs(
    title = "Continuous models",
    x = "Session length for discretisation",
    y = "Simulation parameters of rarity and detectability",
    fill = "RMSE"
  ) +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed,
    switch = "y"
  ) +
  theme_minimal() +
  theme(
    strip.background.y = element_blank(),
    strip.text.y = element_blank(),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    axis.text.y = element_blank(),
    axis.title.y = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  scale_x_discrete(position = "top") +
  scale_fill_gradientn(
    colours = fill_colours_percentiles,
    breaks = fill_values_percentiles,
    labels = fill_txt_percentiles,
    limits = c(0, 1)
  ) +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 30,
      label.position = "bottom",
      title.position = "top"
    )) #; print(plot_RMSE.continuous)


plot_RMSE = ggpubr::ggarrange(plot_RMSE.discrete, plot_RMSE.continuous,
                              common.legend = TRUE,
                              legend = "bottom",
                              ncol = 2,
                              widths = c(1, 0.7))+
  theme(plot.background = element_rect(fill = "white"))
print(plot_RMSE)

ggsave(
  paste0(
    path_figs_out,
    "PSI--RMSE_tileplot_percentile_topmod_", optim_method, ".pdf"
  ),
  plot = plot_RMSE,
  width =  18,
  height = 20,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--RMSE_tileplot_percentile_topmod_", optim_method, ".jpeg"),
  width = 18,
  height = 20,
  unit = "cm",
  res = 100
)
print(plot_RMSE)
dev.off()


### y = model ------------------------------------------------------------------

plot_RMSE_ymod = df_psi_metrics %>%
  ggplot(aes(
    x = 1,
    y = model.session,
    fill = psi.rmse.prctl
  )) +
  geom_tile(colour='white') +
  facet_grid(
    reorder(
      x = factor(
        param.avg_Nij_zi1,
        levels = sort(unique(param.avg_Nij_zi1)),
        labels = paste0("N[100] == ", sort(unique(param.detection))),
        ordered = T
      ),
      X = param.avg_Nij_zi1,
      decreasing = T
    )
    ~
      factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
    labeller = label_parsed
  ) +
  scale_fill_gradientn(
    colours = fill_colours_percentiles,
    breaks = fill_values_percentiles,
    labels = fill_txt_percentiles,
    limits = c(0, 1)
  ) +
  labs(
    fill = "RMSE"
  ) +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 20,
      label.position = "bottom",
      title.position = "top"
    )) +
  theme_minimal() +
  theme(
    strip.background.x = element_rect(fill = "gray93", color = "white"),
    strip.background.y = element_rect(fill = "gray80", color = "white"),
    strip.text = element_text(colour = "black", face = "bold"),
    axis.title.y = element_blank(),
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    plot.title = element_blank()#element_text(hjust = 0.5)
  )
print(plot_RMSE_ymod)


ggsave(
  paste0(
    path_figs_out,
    "PSI--RMSE_tileplot_percentile_ymod_", optim_method, ".pdf"
  ),
  plot = plot_RMSE_ymod,
  width =  13,
  height = 25,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--RMSE_tileplot_percentile_ymod_", optim_method, ".jpeg"),
  width = 13,
  height = 25,
  unit = "cm",
  res = 100
)
print(plot_RMSE_ymod)
dev.off()


## RMSE classic ----------------------------------------------------------------


### facet_top = model ----------------------------------------------------------
# Discrete plot
plot_RMSE.discrete = df_psi_metrics %>%
  filter(model %in% c("BP", "COP")) %>%
  ggplot(
    aes(x = param.SessionLength.Text, y = param.letter, fill = psi.rmse.norm)
  ) +
  geom_tile() +
  geom_text(
    aes(
      label = psi.rmse.label
    ),
    color = "black",
    parse = F,
    size = 2.7
  ) +
  labs(
    title = "Discrete models",
    x = "Session length for discretisation",
    y = "Simulation parameters of rarity and detectability",
    fill = "RMSE"
  ) +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed,
    switch = "y"
  ) +
  theme_minimal() +
  theme(
    strip.background.y = element_rect(fill = "gray93", colour = "white"),
    strip.text.y = element_text(colour = "black", face = "bold"),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    axis.text.y = element_text(hjust=.5),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  scale_x_discrete(position = "top") +
  scale_fill_gradientn(
    colours = fill_colours,
    breaks = fill_values * 1 / 0.8,
    labels = fill_txt,
    limits = c(0, 1)
  ) +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 30,
      label.position = "bottom",
      title.position = "top"
    )) #; print(plot_RMSE.discrete)


plot_RMSE.continuous = df_psi_metrics %>%
  filter(model %in% c("2-MMPP", "IPP", "PP")) %>%
  mutate(param.SessionLength.Text = "") %>%
  ggplot(
    aes(x = param.SessionLength.Text, y = param.letter, fill = psi.rmse.norm)
  ) +
  geom_tile() +
  geom_text(
    aes(
      label = psi.rmse.label
    ),
    color = "black",
    parse = F,
    size = 2.7
  ) +
  labs(
    title = "Continuous models",
    x = "Session length for discretisation",
    y = "Simulation parameters of rarity and detectability",
    fill = "RMSE"
  ) +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed,
    switch = "y"
  ) +
  theme_minimal() +
  theme(
    strip.background.y = element_blank(),
    strip.text.y = element_blank(),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    axis.text.y = element_blank(),
    axis.title.y = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  scale_x_discrete(position = "top") +
  scale_fill_gradientn(
    colours = fill_colours,
    breaks = fill_values * 1 / 0.8,
    labels = fill_txt,
    limits = c(0, 1)
  ) +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 30,
      label.position = "bottom",
      title.position = "top"
    )) #; print(plot_RMSE.continuous)


plot_RMSE = ggpubr::ggarrange(plot_RMSE.discrete, plot_RMSE.continuous,
                              common.legend = TRUE,
                              legend = "bottom",
                              ncol = 2,
                              widths = c(1, 0.7))+
  theme(plot.background = element_rect(fill = "white"))
print(plot_RMSE)

ggsave(
  paste0(
    path_figs_out,
    "PSI--RMSE_tileplot_topmod_", optim_method, ".pdf"
  ),
  plot = plot_RMSE,
  width =  18,
  height = 20,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--RMSE_tileplot_topmod_", optim_method, ".jpeg"),
  width = 18,
  height = 20,
  unit = "cm",
  res = 100
)
print(plot_RMSE)
dev.off()


### y = model ------------------------------------------------------------------

plot_RMSE_ymod = df_psi_metrics %>%
  ggplot(aes(
    x = 1,
    y = model.session,
    fill = psi.rmse.norm
  )) +
  geom_tile(colour='white') +
  facet_grid(
    reorder(
      x = factor(
        param.avg_Nij_zi1,
        levels = sort(unique(param.avg_Nij_zi1)),
        labels = paste0("N[100] == ", sort(unique(param.detection))),
        ordered = T
      ),
      X = param.avg_Nij_zi1,
      decreasing = T
    )
    ~
      factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
    labeller = label_parsed
  ) +
  scale_fill_gradientn(
    colours = fill_colours,
    breaks = fill_values * 1 / 0.8,
    labels = fill_txt,
    limits = c(0, 1)
  ) +
  labs(
    fill = "RMSE"
  ) +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 20,
      label.position = "bottom",
      title.position = "top"
    )) +
  theme_minimal() +
  theme(
    strip.background.x = element_rect(fill = "gray93", color = "white"),
    strip.background.y = element_rect(fill = "gray80", color = "white"),
    strip.text = element_text(colour = "black", face = "bold"),
    axis.title.y = element_blank(),
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    plot.title = element_blank()#element_text(hjust = 0.5)
  )
print(plot_RMSE_ymod)


ggsave(
  paste0(
    path_figs_out,
    "PSI--RMSE_tileplot_ymod_", optim_method, ".pdf"
  ),
  plot = plot_RMSE_ymod,
  width =  13,
  height = 25,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--RMSE_tileplot_ymod_", optim_method, ".jpeg"),
  width = 13,
  height = 25,
  unit = "cm",
  res = 100
)
print(plot_RMSE_ymod)
dev.off()





# ──────────────────────────────────────────────────────────────────────────────
# PSI COVERAGE                                                              ----
# ──────────────────────────────────────────────────────────────────────────────

results_for_visu %>%
  mutate(psi_UCI = ifelse(!is.na(psi_upper_95CI), Im(psi_upper_95CI) == 0, NA)) %>%
  pull(psi_UCI) %>% 
  table(useNA = "always")

results_for_visu %>%
  mutate(psi_LCI = ifelse(!is.na(psi_lower_95CI), Im(psi_lower_95CI) == 0, NA)) %>%
  pull(psi_LCI) %>% 
  table(useNA = "always")


### Colours --------------------------------------------------------------------

# viridis::plasma(40)
# fill_colours_all = c(
#   "1.00" = "#F0F921FF",
#   ".975" = "#FDAE32FF",
#   "0.95" = "#E4695EFF",
#   "0.90" = "#C03A51FF",
#   "0.85" = "#A22B62FF",
#   "0.70" = "#8D2369FF",
#   "0.60" = "#83206BFF",
#   "0.50" = "#781C6DFF",
#   "0.40" = "#6E186EFF",
#   "0.30" = "#64156EFF",
#   "0.20" = "#59106EFF",
#   "0.10" = "#4F0D6CFF",
#   "0.00" = "#390963FF"#,
#   #"0.00" = "#2D0B5AFF",
#   # "0.20" = "#230C4BFF",
#   # "0.15" = "#180C3CFF",
#   # "0.10" = "#0F092CFF",
#   # "0.05" = "#08051DFF",
#   # "0.02" = "#02020FFF",
#   # "0.00" = "#000004FF"
# )

# fill_colours = fill_colours_all[
#   1:which(as.numeric(names(fill_colours_all)) ==
#             (ceiling(min(df_psi_metrics$psi.coverage, na.rm = T) * 20) / 20))]
fill_colours = rev(
  c(
    "#DEF5E5FF",
    # "#54C9ADFF",
    "#38AAACFF",
    "#2fa0ac",
    "#3091a9",
    "#348AA6FF",
    "#357BA2FF",
    "#366A9FFF",
    "#36609b",
    "#3a5595",
    "#40498EFF",
    "#413D7BFF"
    # "#3B2F5EFF"
  )
)
fill_values = seq(0, 1, length.out = 5)
fill_txt = fill_values * 100

# a = 1 / (1 - min(fill_values))
# b = -(min(fill_values) * a)
# df_psi_metrics$psi.coverage.norm = (a * df_psi_metrics$psi.coverage + b)


### facet_top = model ----------------------------------------------------------

#### without n -----------------------------------------------------------------

# Discrete plot
plot_coverage.discrete = df_psi_metrics %>%
  filter(model %in% c("BP", "COP")) %>%
  ggplot(
    aes(x = param.SessionLength.Text, y = param.letter, fill = psi.coverage, colour="")
  ) +
  geom_tile() +
  geom_text(
    aes(
      label = paste(format(psi.coverage * 100, digits = 2, nsmall = 2), "%")
    ),
    color = "black",
    parse = F,
    size = 2.7
  ) +
  labs(
    title = "Discrete models",
    x = "Session length for discretisation",
    y = "Simulation parameters of rarity and detectability",
    fill = "Coverage (%)"
  ) +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed,
    switch = "y"
  ) +
  theme_minimal() +
  theme(
    strip.background.y = element_rect(fill = "gray93", colour = "white"),
    strip.text.y = element_text(colour = "black", face = "bold"),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    axis.text.y = element_text(hjust=.5),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  scale_x_discrete(position = "top") +
  scale_fill_gradientn(
    colours = (fill_colours),
    breaks = seq(from = 0,
                 to = 1,
                 length.out = length(fill_txt)),
    labels = (fill_txt),
    limits = c(0, 1),
    na.value = "gray3"
  ) +
  scale_colour_manual(values = "white") +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 20,
      label.position = "bottom",
      title.position = "top",
    ),
    colour = guide_legend(
      "< 100 CI",
      override.aes = list(colour = "black"),
      label.position = "bottom",
      title.position = "top",
      title.hjust = .5,
      label.hjust = .5,
      keywidth = 5,
    )
  ) #; print(plot_coverage.discrete)



# Continuous plot
plot_coverage.continuous <- df_psi_metrics %>%
  filter(model %in% c("PP", "IPP", "2-MMPP")) %>%
  mutate(param.SessionLength.Text = "") %>%
  ggplot(
    aes(x = param.SessionLength.Text, y = param.letter, fill = psi.coverage, colour="")
  ) +
  geom_tile() +
  geom_text(
    aes(
      label = paste(format(psi.coverage * 100, digits = 2, nsmall = 2), "%")
    ),
    color = "black",
    parse = F,
    size = 2.7
  ) +
  labs(
    title = "Continuous models",
    x = "Session length for discretisation",
    y = "Simulation parameters of rarity and detectability",
    fill = "Coverage (%)"
  ) +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed,
    switch = "y"
  ) +
  theme_minimal() +
  theme(
    strip.background.y = element_blank(),
    strip.text.y = element_blank(),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    axis.text.y = element_blank(),
    axis.title.y = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  scale_x_discrete(position = "top") +
  scale_fill_gradientn(
    colours = (fill_colours),
    breaks = seq(from = 0,
                 to = 1,
                 length.out = length(fill_txt)),
    labels = (fill_txt),
    limits = c(0, 1),
    na.value = "gray3"
  ) +
  scale_colour_manual(values = "white") +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 20,
      label.position = "bottom",
      title.position = "top",
    ),
    colour = guide_legend(
      "< 100 CI",
      override.aes = list(colour = "black"),
      label.position = "bottom",
      title.position = "top",
      title.hjust = .5,
      label.hjust = .5,
      keywidth = 5,
    )
  ) #; print(plot_coverage.continuous)




plot_coverage = ggpubr::ggarrange(plot_coverage.discrete, plot_coverage.continuous,
                                  common.legend = TRUE,
                                  legend = "bottom",
                                  ncol = 2,
                                  widths = c(1, 0.7))+
  theme(plot.background = element_rect(fill = "white"))
print(plot_coverage)

ggsave(
  paste0(
    path_figs_out,
    "PSI--coverage_tileplot_topmod_", optim_method, ".pdf"
  ),
  plot = plot_coverage,
  width =  18,
  height = 20,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--coverage_tileplot_topmod_", optim_method, ".jpeg"),
  width = 18,
  height = 20,
  unit = "cm",
  res = 100
)
print(plot_coverage)
dev.off()



#### with n -----------------------------------------------------------------

# Discrete plot
plot_coverage.discrete = df_psi_metrics %>%
  filter(model %in% c("BP", "COP")) %>%
  ggplot(
    aes(x = param.SessionLength.Text, y = param.letter, fill = psi.coverage, colour="")
  ) +
  geom_tile() +
  geom_text(
    aes(
      label = paste(format(psi.coverage * 100, digits = 2, nsmall = 2), "% (n=", psi.coverage.NbSimul,")")
    ),
    color = "black",
    parse = F,
    size = 2.7
  ) +
  labs(
    title = "Discrete models",
    x = "Session length for discretisation",
    y = "Simulation parameters of rarity and detectability",
    fill = "Coverage (%)"
  ) +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed,
    switch = "y"
  ) +
  theme_minimal() +
  theme(
    strip.background.y = element_rect(fill = "gray93", colour = "white"),
    strip.text.y = element_text(colour = "black", face = "bold"),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    axis.text.y = element_text(hjust=.5),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  scale_x_discrete(position = "top") +
  scale_fill_gradientn(
    colours = (fill_colours),
    breaks = seq(from = 0,
                 to = 1,
                 length.out = length(fill_txt)),
    labels = (fill_txt),
    limits = c(0, 1),
    na.value = "gray3"
  ) +
  scale_colour_manual(values = "white") +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 20,
      label.position = "bottom",
      title.position = "top",
    ),
    colour = guide_legend(
      "< 100 CI",
      override.aes = list(colour = "black"),
      label.position = "bottom",
      title.position = "top",
      title.hjust = .5,
      label.hjust = .5,
      keywidth = 5,
    )
  ) #; print(plot_coverage.discrete)



# Continuous plot
plot_coverage.continuous <- df_psi_metrics %>%
  filter(model %in% c("PP", "IPP", "2-MMPP")) %>%
  mutate(param.SessionLength.Text = "") %>%
  ggplot(
    aes(x = param.SessionLength.Text, y = param.letter, fill = psi.coverage, colour="")
  ) +
  geom_tile() +
  geom_text(
    aes(
      label = paste(format(psi.coverage * 100, digits = 2, nsmall = 2), "% (n=", psi.coverage.NbSimul,")")
    ),
    color = "black",
    parse = F,
    size = 2.7
  ) +
  labs(
    title = "Continuous models",
    x = "Session length for discretisation",
    y = "Simulation parameters of rarity and detectability",
    fill = "Coverage (%)"
  ) +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed,
    switch = "y"
  ) +
  theme_minimal() +
  theme(
    strip.background.y = element_blank(),
    strip.text.y = element_blank(),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    axis.text.y = element_blank(),
    axis.title.y = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  scale_x_discrete(position = "top") +
  scale_fill_gradientn(
    colours = (fill_colours),
    breaks = seq(from = 0,
                 to = 1,
                 length.out = length(fill_txt)),
    labels = (fill_txt),
    limits = c(0, 1),
    na.value = "gray3"
  ) +
  scale_colour_manual(values = "white") +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 20,
      label.position = "bottom",
      title.position = "top",
    ),
    colour = guide_legend(
      "< 100 CI",
      override.aes = list(colour = "black"),
      label.position = "bottom",
      title.position = "top",
      title.hjust = .5,
      label.hjust = .5,
      keywidth = 5,
    )
  ) #; print(plot_coverage.continuous)


plot_coverage = ggpubr::ggarrange(plot_coverage.discrete, plot_coverage.continuous,
                                  common.legend = TRUE,
                                  legend = "bottom",
                                  ncol = 2,
                                  widths = c(1, 0.7))+
  theme(plot.background = element_rect(fill = "white"))
print(plot_coverage)

ggsave(
  paste0(
    path_figs_out,
    "PSI--coverage_tileplot_topmod_", optim_method, "with_n.pdf"
  ),
  plot = plot_coverage,
  width =  29,
  height = 20,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--coverage_tileplot_topmod_", optim_method, "with_n.jpeg"),
  width = 29,
  height = 20,
  unit = "cm",
  res = 100
)
print(plot_coverage)
dev.off()


### y = model ------------------------------------------------------------------

psi_coverage_ymod = df_psi_metrics %>%
  ggplot(aes(
    x = 1,
    y = model.session,
    fill = psi.coverage
  )) +
  geom_tile(colour='white') +
  facet_grid(
    reorder(
      x = factor(
        param.avg_Nij_zi1,
        levels = sort(unique(param.avg_Nij_zi1)),
        labels = paste0("N[100] == ", sort(unique(param.detection))),
        ordered = T
      ),
      X = param.avg_Nij_zi1,
      decreasing = T
    )
    ~
      factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
    labeller = label_parsed
  ) +
  scale_fill_gradientn(
    colours = rev(fill_colours),
    breaks = seq(from = 0,
                 to = 1,
                 length.out = length(fill_txt)),
    labels = rev(fill_txt),
    limits = c(0, 1),
    na.value = "grey3"
  ) +
  labs(
    fill = "Coverage (%)"
  ) +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 20,
      label.position = "bottom",
      title.position = "top"
    )) +
  theme_minimal() +
  theme(
    strip.background.x = element_rect(fill = "gray93", color = "white"),
    strip.background.y = element_rect(fill = "gray80", color = "white"),
    strip.text = element_text(colour = "black", face = "bold"),
    axis.title.y = element_blank(),
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    plot.title = element_blank()#element_text(hjust = 0.5)
  )
print(psi_coverage_ymod)


ggsave(
  paste0(
    path_figs_out,
    "PSI--coverage_tileplot_ymod_", optim_method, ".pdf"
  ),
  plot = psi_coverage_ymod,
  width =  13,
  height = 25,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--coverage_tileplot_ymod_", optim_method, ".jpeg"),
  width = 13,
  height = 25,
  unit = "cm",
  res = 100
)
print(psi_coverage_ymod)
dev.off()



# ──────────────────────────────────────────────────────────────────────────────
# PSI IC RANGE                                                              ----
# ──────────────────────────────────────────────────────────────────────────────

### facet_top = model ----------------------------------------------------------

#### without n -----------------------------------------------------------------

# Discrete plot
plot_psi_ic_range.discrete = df_psi_metrics %>%
  filter(model %in% c("BP", "COP")) %>%
  ggplot(
    aes(x = param.SessionLength.Text, y = param.letter, fill = psi.CI.range.avg, colour="")
  ) +
  geom_tile() +
  geom_text(
    aes(
      label = format(round(psi.CI.range.avg, 2),
                     digits = 2,
                     nsmall = 2)
    ),
    color = "black",
    parse = F,
    size = 2.7
  ) +
  labs(
    title = "Discrete models",
    x = "Session length for discretisation",
    y = "Simulation parameters of rarity and detectability",
    fill = "Average range of the CI"
  ) +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed,
    switch = "y"
  ) +
  theme_minimal() +
  theme(
    strip.background.y = element_rect(fill = "gray93", colour = "white"),
    strip.text.y = element_text(colour = "black", face = "bold"),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    axis.text.y = element_text(hjust=.5),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  scale_x_discrete(position = "top") +
  scale_fill_gradientn(
    # colours = rev(viridis::magma(14)[-c(1,2,3,4)]),
    colours = c(
      "#F0F921FF",
      "#FDAE32FF",
      "#E4695EFF",
      "#A22B62FF",
      "#781C6DFF"
    ),
    breaks = seq(
      from = 0,
      to = 1,
      length.out = 5
    ),
    limits = c(0, 1),
    na.value = "gray3"
  ) +
  scale_colour_manual(values = "white") +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 20,
      label.position = "bottom",
      title.position = "top",
    ),
    colour = guide_legend(
      "< 100 CI",
      override.aes = list(colour = "black"),
      label.position = "bottom",
      title.position = "top",
      title.hjust = .5,
      label.hjust = .5,
      keywidth = 5,
    )
  ) ; print(plot_psi_ic_range.discrete)



# Continuous plot
plot_psi_ic_range.continuous <- df_psi_metrics %>%
  filter(model %in% c("PP", "IPP", "2-MMPP")) %>%
  mutate(param.SessionLength.Text = "") %>%
  ggplot(
    aes(x = param.SessionLength.Text, y = param.letter, fill = psi.CI.range.avg, colour="")
  ) +
  geom_tile() +
  geom_text(
    aes(
      label = format(round(psi.CI.range.avg, 2),
                     digits = 2,
                     nsmall = 2)
    ),
    color = "black",
    parse = F,
    size = 2.7
  ) +
  labs(
    title = "Continuous models",
    x = "Session length for discretisation",
    y = "Simulation parameters of rarity and detectability",
    fill = "Average range of the CI"
  ) +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed,
    switch = "y"
  ) +
  theme_minimal() +
  theme(
    strip.background.y = element_blank(),
    strip.text.y = element_blank(),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    axis.text.y = element_blank(),
    axis.title.y = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
  ) +
  scale_x_discrete(position = "top") +
  scale_fill_gradientn(
    # colours = rev(viridis::magma(14)[-c(1,2,3,4)]),
    colours=c("#F0F921FF", "#FDAE32FF", "#E4695EFF", "#A22B62FF", "#781C6DFF"),
    breaks = seq(from = 0,
                 to = 1,
                 length.out = 5),
    limits = c(0,1),
    na.value = "gray3"
  ) +
  scale_colour_manual(values = "white") +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = 20,
      label.position = "bottom",
      title.position = "top",
    ),
    colour = guide_legend(
      "< 100 CI",
      override.aes = list(colour = "black"),
      label.position = "bottom",
      title.position = "top",
      title.hjust = .5,
      label.hjust = .5,
      keywidth = 5,
    )
  ) ; print(plot_psi_ic_range.continuous)



plot_psi_ic_range = ggpubr::ggarrange(plot_psi_ic_range.discrete, plot_psi_ic_range.continuous,
                                      common.legend = TRUE,
                                      legend = "bottom",
                                      ncol = 2,
                                      widths = c(1, 0.7))+
  theme(plot.background = element_rect(fill = "white"))
print(plot_psi_ic_range)

ggsave(
  paste0(
    path_figs_out,
    "PSI--IC_range_tileplot_topmod_", optim_method, ".pdf"
  ),
  plot = plot_psi_ic_range,
  width =  18,
  height = 20,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--IC_range_tileplot_topmod_", optim_method, ".jpeg"),
  width = 18,
  height = 20,
  unit = "cm",
  res = 100
)
print(plot_psi_ic_range)
dev.off()






# ──────────────────────────────────────────────────────────────────────────────
# BOXPLOT PSI                                                               ----
# ──────────────────────────────────────────────────────────────────────────────

### facet_top = model ----------------------------------------------------------


# Plot discrete BP
boxplot_psi_BP <- results_for_visu %>%
  mutate(
    param.detection = factor(format(round(
      param.avg_Nij_zi1, 2
    ), nsmall = 2))
  ) %>%
  filter(model %in% c("BP")) %>%
  ggplot(aes(
    x = param.detection,
    y = psi,
    # fill = model.session#str_replace(string = model.session, pattern = " ", replacement = "\n")
  )) +
  geom_hline(aes(yintercept = param.psi), color = "grey10", linetype = "solid", linewidth = .5) +
  geom_boxplot(color = "black", linewidth=.2, outlier.size =.2, fill = "grey80") +
  coord_flip() +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ param.SessionLength.Text,
    labeller = label_parsed,
    switch = "x"
  ) + 
  labs(title = "BP",
       y = "Session length for discretisation",
       x = "Expected number of detections per deployment in occupied sites") +
  theme_light() +
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 1,
      size = 8
    ),
    strip.placement = "outside",
    strip.background.y = element_rect(fill = "gray93", color = "white"),
    strip.background.x = element_blank(),
    strip.text = element_text(
      colour = "black",
      face = "bold",
      size = 10
    ),
    plot.title = ggtext::element_textbox(
      hjust = .5,
      vjust = .5,
      halign = .5,
      valign = .5,
      box.colour="grey80",
      fill = 'grey80',
      width = 1,
      size = 10,
      linetype = 1,
      linewidth = 4
    ),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
  ) +
  scale_y_continuous(breaks = c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) #; print(boxplot_psi_BP)

# Plot discrete BP
boxplot_psi_COP <- results_for_visu %>%
  mutate(
    param.detection = factor(format(round(
      param.avg_Nij_zi1, 2
    ), nsmall = 2))
  ) %>%
  filter(model %in% c("COP")) %>%
  ggplot(aes(
    x = param.detection,
    y = psi,
    # fill = model.session#str_replace(string = model.session, pattern = " ", replacement = "\n")
  )) +
  geom_hline(aes(yintercept = param.psi), color = "grey10", linetype = "solid", linewidth = .5) +
  geom_boxplot(color = "black", linewidth=.2, outlier.size =.2, fill = "grey80") +
  coord_flip() +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ param.SessionLength.Text,
    labeller = label_parsed,
    switch = "x"
  ) + 
  labs(title = "COP",
       y = "Session length for discretisation",
       x = "Expected number of detections per deployment in occupied sites") +
  theme_light() +
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 1,
      size = 8
    ),
    strip.placement = "outside",
    strip.background.y = element_rect(fill = "gray93", color = "white"),
    strip.background.x = element_blank(),
    strip.text = element_text(
      colour = "black",
      face = "bold",
      size = 10
    ),
    plot.title = ggtext::element_textbox(
      hjust = .5,
      vjust = .5,
      halign = .5,
      valign = .5,
      box.colour="grey80",
      fill = 'grey80',
      width = 1,
      size = 10,
      linetype = 1,
      linewidth = 4
    ),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
  ) +
  scale_y_continuous(breaks = c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))# ; print(boxplot_psi_COP)




boxplot_psi_continuous = results_for_visu %>%
  filter(model %in% c("PP", "2-MMPP", "IPP")) %>%
  ggplot(aes(
    x = param.detection,
    y = psi,
    # fill = model.session#str_replace(string = model.session, pattern = " ", replacement = "\n")
  )) +
  geom_hline(aes(yintercept = param.psi), color = "grey10", linetype = "solid", linewidth = .5) +
  geom_boxplot(color = "black", linewidth=.2, outlier.size =.2, fill = "grey80") +
  coord_flip() +
  facet_grid(
    reorder(
      x = factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
      X = param.psi,
      decreasing = T
    ) ~ model,
    labeller = label_parsed
  ) + 
  labs(y = "Estimated psi") +
  scale_y_continuous(breaks = c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) +
  theme_light() +
  theme(
    # axis.text.y = element_blank(),
    axis.title.x = element_blank(),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 8),
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    legend.position = "none",
    legend.title = element_blank(),
    title = element_text(size = 10, face = "bold"),
    strip.background.y = element_rect(fill = "gray93", color = "white"),
    strip.background.x = element_rect(fill = "gray80", color = "white"),
    strip.text = element_text(colour = "black", face = "bold", size = 10),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    # panel.grid.major.y = element_blank(),
  ) #; print(boxplot_psi_continuous)



boxplot_psi_discrete = ggpubr::ggarrange(
  boxplot_psi_BP + theme(
    strip.background.y = element_blank(),
    strip.text.y = element_blank(),
    axis.title.x = element_blank()
  ),
  boxplot_psi_COP + theme(
    strip.background.y = element_blank(),
    strip.text.y = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  ),
  common.legend = TRUE,
  legend = "bottom",
  ncol = 2,
  widths = c(1, 1)
)

boxplot_psi = ggpubr::ggarrange(
  boxplot_psi_discrete + 
    theme(plot.margin = margin(t = 5, r = 0, b = 0, l = 0, unit = "pt")),
  boxplot_psi_continuous + 
    theme(plot.margin = margin(t = 5.5, r = 0, b = 28, l = 0, unit = "pt")),
  common.legend = TRUE,
  legend = "bottom",
  ncol = 2,
  widths = c(1, 0.6)
) +
  theme(plot.background = element_rect(fill = "white"))
print(boxplot_psi)


ggsave(
  paste0(
    path_figs_out,
    "PSI--boxplot_topmod_", optim_method, ".pdf"
  ),
  plot = boxplot_psi,
  width =  25,
  height = 20,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--boxplot_topmod_", optim_method, ".jpeg"),
  width = 25,
  height = 20,
  unit = "cm",
  res = 100
)
print(boxplot_psi)
dev.off()


### y = model ------------------------------------------------------------------

boxplot_psi_ymod = results_for_visu %>%
  ggplot(aes(
    x = model.session,
    y = psi,
    fill = model.session#str_replace(string = model.session, pattern = " ", replacement = "\n")
  )) +
  geom_hline(aes(yintercept = param.psi), color = "grey10", linetype = "solid", linewidth = .5) +
  geom_boxplot(color = "black", linewidth=.05, outlier.size =.2) +
  coord_flip() +
  facet_grid(
    reorder(
      x = factor(
        param.avg_Nij_zi1,
        levels = sort(unique(param.avg_Nij_zi1)),
        labels = paste0("N[100] == ", sort(unique(param.detection))),
        ordered = T
      ),
      X = param.avg_Nij_zi1,
      decreasing = T
    )
    ~
      factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
    labeller = label_parsed
  ) + 
  labs(y = "Estimated psi") +
  scale_y_continuous(breaks = c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) +
  theme_light() +
  theme(
    # axis.text.y = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 8),
    axis.text.y = element_text(size = 8),
    legend.position = "none",
    legend.title = element_blank(),
    title = element_text(size = 10, face = "bold"),
    strip.background.y = element_rect(fill = "gray93", color = "white"),
    strip.background.x = element_rect(fill = "gray80", color = "white"),
    strip.text = element_text(colour = "black", face = "bold", size = 10),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.y = element_blank(),
  ) +
  scale_fill_manual(values = c(
    "BP (Month)" = "darkslategray4",
    "BP (Week)" = "darkslategray3",
    "BP (Day)" = "darkslategray1",
    
    "COP (Month)" = "darkolivegreen4",
    "COP (Week)" = "darkolivegreen3",
    "COP (Day)" = "darkolivegreen1",
    
    "PP" = "goldenrod",
    "IPP" = "coral",
    "2-MMPP" = "firebrick"
  )) +
  guides(fill = guide_legend(nrow = 1))

print(boxplot_psi_ymod)

ggsave(
  paste0(
    path_figs_out,
    "PSI--boxplot_ymod_", optim_method, ".pdf"
  ),
  plot = boxplot_psi_ymod,
  width =  25,
  height = 25,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--boxplot_ymod_", optim_method, ".jpeg"),
  width = 25,
  height = 25,
  unit = "cm",
  res = 100
)
print(boxplot_psi_ymod)
dev.off()



# ──────────────────────────────────────────────────────────────────────────────
# VIOLIN PSI                                                                ----
# ──────────────────────────────────────────────────────────────────────────────

### y = model ------------------------------------------------------------------

violin_psi_ymod = results_for_visu %>%
  ggplot(aes(
    x = model.session,
    y = psi,
    fill = model.session#str_replace(string = model.session, pattern = " ", replacement = "\n")
  )) +
  geom_hline(aes(yintercept = param.psi), color = "grey10", linetype = "solid", linewidth = .5) +
  geom_violin(color = "black", linewidth = .1) +
  coord_flip() +
  facet_grid(
    param.letter
    ~
      factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
    labeller = label_parsed
  ) + 
  labs(y = "Occupancy probability point estimate") +
  scale_y_continuous(breaks = c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) +
  theme_light() +
  theme(
    # axis.text.y = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 8),
    axis.text.y = element_text(size = 8),
    legend.position = "bottom",
    legend.title = element_blank(),
    title = element_text(size = 10, face = "bold"),
    strip.background.y = element_rect(fill = "gray93", color = "white"),
    strip.background.x = element_rect(fill = "gray80", color = "white"),
    strip.text = element_text(colour = "black", face = "bold", size = 10),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.y = element_blank(),
  ) +
  scale_fill_manual(values = c(
    "BP (Month)" = "darkslategray4",
    "BP (Week)" = "darkslategray3",
    "BP (Day)" = "darkslategray1",
    
    "COP (Month)" = "darkolivegreen4",
    "COP (Week)" = "darkolivegreen3",
    "COP (Day)" = "darkolivegreen1",
    
    "PP" = "goldenrod",
    "IPP" = "coral",
    "2-MMPP" = "firebrick"
  )) +
  guides(fill = guide_legend(nrow = 1))

print(violin_psi_ymod)

ggsave(
  paste0(
    path_figs_out,
    "PSI--violin_ymod_", optim_method, ".pdf"
  ),
  plot = violin_psi_ymod,
  width =  25,
  height = 30,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI--violin_ymod_", optim_method, ".jpeg"),
  width = 25,
  height = 30,
  unit = "cm",
  res = 100
)
print(violin_psi_ymod)
dev.off()




# ──────────────────────────────────────────────────────────────────────────────
# BOXPLOT P_1                                                               ----
# ──────────────────────────────────────────────────────────────────────────────

### y = model ------------------------------------------------------------------

p_1_boxplot = results_for_visu %>%
  mutate(
    param.p_1 = round(param.p_1, 3)
  ) %>%
  ggplot(aes(
    x = model.session,
    y = p_1,
    fill = model.session
  )) +
  geom_hline(aes(yintercept = param.p_1), color = "grey10", linetype = "solid", linewidth = .5) +
  geom_boxplot(color = "black", linewidth=.05, outlier.size =.2) +
  # coord_flip() +
  facet_grid(
    reorder(
      x = factor(
        param.avg_Nij_zi1,
        levels = sort(unique(param.avg_Nij_zi1)),
        labels = paste0("N[100] == ", sort(unique(param.detection))),
        ordered = T
      ),
      X = param.avg_Nij_zi1,
      decreasing = T
    )
    ~
      factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
    labeller = label_parsed
  ) +
  labs(y = "Estimated probability of having at least one detection in 1 day") +
  scale_y_continuous(breaks = c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) +
  theme_light() +
  theme(
    # axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 8),
    axis.text.y = element_text(size = 8),
    legend.position = "none",
    legend.title = element_blank(),
    title = element_text(size = 10, face = "bold"),
    strip.background.y = element_rect(fill = "gray93", color = "white"),
    strip.background.x = element_rect(fill = "gray80", color = "white"),
    strip.text = element_text(colour = "black", face = "bold", size = 10),
    panel.grid.minor.y = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.x = element_blank(),
  ) +
  scale_fill_manual(values = c(
    "BP (Month)" = "darkslategray4",
    "BP (Week)" = "darkslategray3",
    "BP (Day)" = "darkslategray1",
    
    "COP (Month)" = "darkolivegreen4",
    "COP (Week)" = "darkolivegreen3",
    "COP (Day)" = "darkolivegreen1",
    
    "PP" = "goldenrod",
    "IPP" = "coral",
    "2-MMPP" = "firebrick"
  )) +
  guides(fill = guide_legend(nrow = 1))
plot(p_1_boxplot)

ggsave(
  paste0(path_figs_out,
         "DETEC--p_1_boxplot_ymod_", optim_method, ".pdf"),
  plot = p_1_boxplot,
  width = 21,
  height = 27,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "DETEC--p_1_boxplot_ymod_", optim_method, ".jpeg"),
  width = 21,
  height = 27,
  unit = "cm",
  res = 100
)
print(p_1_boxplot)
dev.off()

















# ──────────────────────────────────────────────────────────────────────────────
# COMPARISON TESTS                                                          ----
# ──────────────────────────────────────────────────────────────────────────────

library(rstatix)
library(ggpubr)

results_for_tests = results_for_visu %>%
  ungroup() %>%
  select(psi, param.letter, param.psi, model, model.session)

## Kruskal-Wallis & Wilcoxon ---------------------------------------------------

param.letter_ls = sort(unique(results_for_tests$param.letter))
param.psi_ls = as.character(sort(unique(results_for_tests$param.psi)))

res_tests <- vector(mode = 'list', length = length(param.letter_ls)) %>%
  setNames(param.letter_ls)
res_tests_df <- expand.grid(param.letter_ls, param.psi_ls) %>% 
  setNames(c("param.letter", "param.psi")) %>% 
  mutate("KW.p.value"=NA,"KW.statistic"=NA)
for (i in param.letter_ls ) {
  res_tests[[i]] <- vector(mode = 'list', length = length(param.psi_ls)) %>%
    setNames(param.psi_ls)
  for (j in param.psi_ls) {
    res_tests[[i]][[j]] <- vector(mode = 'list', length = 2) %>%
      setNames(c("Kruskal-Wallis", "Wilcoxon rank sum test"))
    
    dfsub <- results_for_tests %>%
      filter(param.letter == i,
             param.psi == as.numeric(j))
    
    res_tests_l_p <- kruskal.test(x = dfsub$psi, g = dfsub$model.session)
    res_tests[[i]][[j]][["Kruskal-Wallis"]] <- res_tests_l_p
    res_tests_df[which(res_tests_df$param.letter == i &
                         res_tests_df$param.psi == j),
                 "KW.p.value"] <- res_tests_l_p$p.value
    res_tests_df[which(res_tests_df$param.letter == i &
                         res_tests_df$param.psi == j),
                 "KW.statistic"] <- res_tests_l_p$statistic
    
    res_pww_l_p <- pairwise.wilcox.test(x = dfsub$psi, g = dfsub$model.session, p.adjust.method = "bonferroni")
    res_tests[[i]][[j]][["Wilcoxon rank sum test (bonferroni)"]] <- res_pww_l_p
    
    }
}

### Kruskal-Wallis -------------------------------------------------------------

#### Table ----


df_KW.statistic = res_tests_df %>%
  mutate(KW.res = format(
    KW.statistic,
    digits = 2,
    nsmall = 2,
    scientific = F
  )) %>%
  select(!KW.p.value:KW.statistic) %>%
  pivot_wider(names_from = param.psi,
              values_from = "KW.res") %>% 
  arrange(desc(param.letter))

df_KW.p.value = res_tests_df %>%
  mutate(KW.res = paste0(
    ifelse(
      substr(format.pval(KW.p.value, digits = 2), 1, 1) == "<",
      paste0("p ", format.pval(KW.p.value, digits = 2)),
      paste0("p = ", format.pval(KW.p.value, digits = 2))
    ),
    ifelse(stars.pval(KW.p.value) == " ",
           "",
           paste0(" (", stars.pval(KW.p.value), ")"))
  )) %>%
  select(!KW.p.value:KW.statistic) %>%
  pivot_wider(names_from = param.psi,
              values_from = "KW.res") %>% 
  arrange(desc(param.letter))



sink(paste0(path_figs_out, "PSI-TEST--Kruskal-Wallis.tex"))
for (i in 1:nrow(df_KW.statistic)){
  cat(paste0(
    "\\multirow{2}{*}{", df_KW.statistic[i, ]$param.letter, "} & ",
    df_KW.statistic[i,] %>% select(!param.letter)%>% c(., recursive=T) %>% paste(collapse=" & "),
    " \\\\", "\n",
    " & ", df_KW.p.value[i,] %>% select(!param.letter)%>% c(., recursive=T) %>% paste(collapse=" & "),
    " \\\\", "\n",
    "\\hlineB{1}", "\n"
  ))
}
sink()

#### Plot ----
# Which combinations have significant diffences?
kw_plot = res_tests_df %>%
  mutate(KW.p.value.stars = stars.pval(p.value = KW.p.value)) %>% 
  ggplot(aes(x = 1, y = 1, fill = KW.p.value.stars)) +
  geom_tile(colour = 'white') +
  geom_text(aes(label = format.pval(res_tests_df$KW.p.value, digits = 1)))+
  facet_grid(
    param.letter
    ~
      factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
    labeller = label_parsed
  ) +
  labs(fill = "Kruskal-Wallis p-value",
       x = "Detection parameters",
       y = "Occupancy probability") %>%
  scale_fill_manual(
    values = c("#D7191C", "#FF7417", "#fedc56", "#ABDDA4", "#2B83BA"),
    limits = c("***", "**", "*", ".", " "),
    labels = c("≤ 0.001 (***)", "≤ 0.01 (**)", "≤ 0.05 (*)", "≤ 0.1 (.)", "> 0.1")
  ) +
  theme_light() +
  theme(
    axis.title.y = element_blank(),
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    legend.position = "bottom",
    # legend.title = element_blank(),
    title = element_text(size = 10, face = "bold"),
    strip.background.y = element_rect(fill = "gray93", color = "white"),
    strip.background.x = element_rect(fill = "gray80", color = "white"),
    strip.text = element_text(colour = "black", face = "bold", size = 10),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.y = element_blank(),
  ) +
  guides(fill = guide_legend(title.position = "top", title.hjust = 0.5));print(kw_plot)


ggsave(
  paste0(path_figs_out,
         "PSI-TEST--Kruskal-Wallis_p-value_", optim_method, ".pdf"),
  plot = kw_plot,
  width = 15,
  height = 15,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI-TEST--Kruskal-Wallis_p-value_", optim_method, ".jpeg"),
  width = 15,
  height = 15,
  unit = "cm",
  res = 100
)
print(kw_plot)
dev.off()





### Wilcoxon -------------------------------------------------------------------



for (i in 1:nrow(res_tests_df)) {
  df_wilc_i = res_tests[[res_tests_df[i, 'param.letter']]][[res_tests_df[i, 'param.psi']]][["Wilcoxon rank sum test (bonferroni)"]]$p.value %>%
    as.data.frame() %>%
    rownames_to_column("Mod1") %>%
    pivot_longer(cols = !Mod1,
                 names_to = "Mod2",
                 values_to = "p.value") %>%
    filter(!is.na(p.value)) %>%
    mutate(
      param.letter = res_tests_df[i, 'param.letter'],
      param.psi = res_tests_df[i, 'param.psi'],
      Mod1 = factor(Mod1,
                    levels = rev(
                      levels(results_for_visu$model.session)
                    ),
                    ordered = T),
      Mod2 = factor(Mod2, levels = (
        levels(results_for_visu$model.session)
      ), ordered = T),
    )
  if (i == 1) {
    df_wilc = df_wilc_i
  } else{
    df_wilc = bind_rows(df_wilc, df_wilc_i)
  }
}


wilcoxon_plot = df_wilc %>%
  mutate(p.value.stars = stars.pval(p.value = p.value))%>% 
  ggplot(aes(x = Mod2, y = Mod1, fill = p.value.stars)) +
  geom_tile(colour = 'white')+
  # geom_text(aes(label = format.pval(df_wilc$p.value, digits = 1))) +
  facet_grid(
    param.letter
    ~
      factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
    labeller = label_parsed
  ) +
  labs(fill = "Wilcoxon rank sum test p-value") %>%
  scale_fill_manual(
    values = c("#D7191C", "#FF7417", "#fedc56", "#ABDDA4", "#2B83BA"),
    limits = c("***", "**", "*", ".", " "),
    labels = c("≤ 0.001 (***)", "≤ 0.01 (**)", "≤ 0.05 (*)", "≤ 0.1 (.)", "> 0.1")
  ) +
  theme_light() +
  theme(
    axis.title.y = element_blank(),
    axis.title.x = element_blank(),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 8),
    axis.text.y = element_text(size = 8),
    legend.position = "bottom",
    title = element_text(size = 10, face = "bold"),
    strip.background.y = element_rect(fill = "gray93", color = "white"),
    strip.background.x = element_rect(fill = "gray80", color = "white"),
    strip.text = element_text(colour = "black", face = "bold", size = 10),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank(),
  ) +
  guides(fill = guide_legend(title.position = "top", title.hjust = 0.5));print(wilcoxon_plot)

ggsave(
  paste0(path_figs_out,
         "PSI-TEST--Wilcoxon_p-value_", optim_method, ".pdf"),
  plot = wilcoxon_plot,
  width = 20,
  height = 25,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI-TEST--Wilcoxon_p-value_", optim_method, ".jpeg"),
  width = 20,
  height = 25,
  unit = "cm",
  res = 100
)
print(wilcoxon_plot)
dev.off()




ggsave(
  paste0(path_figs_out,
         "PSI-TEST--Wilcoxon_p-value_", optim_method, "_with_txt.pdf"),
  plot = wilcoxon_plot +
    geom_text(aes(label = format.pval(df_wilc$p.value, digits = 1)), size = 2),
  width = 30,
  height = 25,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "PSI-TEST--Wilcoxon_p-value_", optim_method, "_with_txt.jpeg"),
  width = 30,
  height = 25,
  unit = "cm",
  res = 100
)
print(wilcoxon_plot +
        geom_text(aes(label = format.pval(df_wilc$p.value, digits = 1)), size = 2)
)
dev.off()





# ──────────────────────────────────────────────────────────────────────────────
# CALCULATION TIME                                                          ----
# ──────────────────────────────────────────────────────────────────────────────

### y = model ------------------------------------------------------------------

calculation_time = results_for_visu %>%
  ggplot(aes(
    x = model.session,
    y = log(lubridate::dseconds(fitting_time)),
    fill = model.session,
    color = model.session
  )) +
  geom_violin(alpha = .8, linewidth = .7)+#color = "black", linewidth = .1) +
  # geom_boxplot(color = "black", linewidth = .1) +
  coord_flip() +
  facet_grid(
    param.letter
    ~
      factor(
        param.psi,
        levels = unique(param.psi),
        labels = paste0("psi == ", unique(param.psi)),
        ordered = T
      ),
    labeller = label_parsed,
    # scales="free"
  ) + 
  labs(y = "Fitting time (seconds)") +
  scale_y_continuous(
    limits = c(0, log(ceiling (max(results_for_visu$fitting_time) / 100) * 100)),
    # expand = c(0, 0),
    breaks = c(0,log(c(round(exp(seq(
      from = log(1),
      to = log(ceiling (max(
        results_for_visu$fitting_time
      ) / 100) * 100),
      length.out = 5
    ))))[-1])),
    labels = c(0, round(exp(seq(
      from = log(1),
      to = log(ceiling (max(
        results_for_visu$fitting_time
      ) / 100) * 100),
      length.out = 5
    ))))[-2]
  ) +
  # coord_trans(y="log2")+
  theme_light() +
  theme(
    # axis.text.y = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 8),
    axis.text.y = element_text(size = 8),
    legend.position = "bottom",
    legend.title = element_blank(),
    title = element_text(size = 10, face = "bold"),
    strip.background.y = element_rect(fill = "gray93", color = "white"),
    strip.background.x = element_rect(fill = "gray80", color = "white"),
    strip.text = element_text(colour = "black", face = "bold", size = 10),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.y = element_blank(),
  ) +
  scale_fill_manual(values = c(
    "BP (Month)" = "darkslategray4",
    "BP (Week)" = "darkslategray3",
    "BP (Day)" = "darkslategray1",

    "COP (Month)" = "darkolivegreen4",
    "COP (Week)" = "darkolivegreen3",
    "COP (Day)" = "darkolivegreen1",

    "PP" = "goldenrod",
    "IPP" = "coral",
    "2-MMPP" = "firebrick"
  ), aesthetics = c("fill", "colour")) +
  guides(fill = guide_legend(nrow = 1))

print(calculation_time)

ggsave(
  paste0(
    path_figs_out,
    "TIME--fitting_ymod_", optim_method, ".pdf"
  ),
  plot = calculation_time,
  width =  25,
  height = 30,
  unit = "cm"
)
jpeg(
  filename = paste0(path_figs_out, "TIME--fitting_ymod_", optim_method, ".jpeg"),
  width = 25,
  height = 30,
  unit = "cm",
  res = 100
)
print(calculation_time)
dev.off()





### Tables ---------------------------------------------------------------------

calculation_df_all = results_for_visu %>%
  group_by(model.session, param.letter, param.psi) %>%
  summarise(
    fitting_time_recap = paste0(
      format(round(mean(fitting_time), 2), nsmall = 2),
      ' [',
      format(round(min(fitting_time), 2), nsmall = 2),
      ' - ',
      format(round(max(fitting_time), 2), nsmall = 2),
      ']'
    ),
    
    fitting_time_avg = mean(fitting_time),
    fitting_time_min = min(fitting_time),
    fitting_time_max = max(fitting_time),
    fitting_time_lower_CI = quantile(fitting_time, 0.025),
    fitting_time_upper_CI = quantile(fitting_time, 0.975)
  ) %>%
  arrange(fitting_time_avg)
print(calculation_df_all)
write.csv(calculation_df_all,
          file = paste0(path_figs_out, "TIME--fitting_all_", optim_method, ".csv"))

calculation_df=results_for_visu %>% 
  group_by(model.session) %>% 
  summarise(
    fitting_time_recap = paste0(
      format(round(mean(fitting_time),2), nsmall = 2),
      ' [',
      format(round(min(fitting_time),2), nsmall = 2),
      ' - ',
      format(round(max(fitting_time),2), nsmall = 2),
      ']'
    ),
    
    fitting_time_avg = mean(fitting_time),
    fitting_time_min = min(fitting_time),
    fitting_time_max = max(fitting_time),
    fitting_time_lower_CI = quantile(fitting_time, 0.025),
    fitting_time_upper_CI = quantile(fitting_time, 0.975)
  )
print(calculation_df)
write.csv(calculation_df,
          file = paste0(path_figs_out, "TIME--fitting_", optim_method, ".csv"))


calculation_df=results_for_visu %>% 
  group_by(model) %>% 
  summarise(
    fitting_time_recap = paste0(
      format(round(mean(fitting_time),2), nsmall = 2),
      ' [',
      format(round(min(fitting_time),2), nsmall = 2),
      ' - ',
      format(round(max(fitting_time),2), nsmall = 2),
      ']'
    ),
    
    fitting_time_avg = mean(fitting_time),
    fitting_time_min = min(fitting_time),
    fitting_time_max = max(fitting_time),
    fitting_time_lower_CI = quantile(fitting_time, 0.025),
    fitting_time_upper_CI = quantile(fitting_time, 0.975)
  )
print(calculation_df)
write.csv(calculation_df,
          file = paste0(path_figs_out, "TIME--fitting_recap_", optim_method, ".csv"))



# END --------------------------------------------------------------------------

