# ╭─────────────────────────────────────────────────────────────────────────────╮
# │                                                                             │
# │ Run comparisons for all simulation scenarios                                │
# │ Léa Pautrel, TerrOïko | CEFE | IRMAR                                        │
# │ Last update: October 2023                                                   │
# │                                                                             │
# ╰─────────────────────────────────────────────────────────────────────────────╯


# ──────────────────────────────────────────────────────────────────────────────
# READING ARGUMENTS                                                         ----
# ──────────────────────────────────────────────────────────────────────────────


if (interactive()) {
  # Note: this script was formatted to be called with a command-line.
  # If you want to run it in an IDE like RStudio, the values below will be
  # initialised to produce the same results as in the article.
  try_seed_min = 0
  try_seed_max = 500
  output_path = './output/'
  run_suffix = Sys.Date()
  optim_method = 'Nelder-Mead'
} else {
  args = commandArgs(trailingOnly = TRUE)
  if (length(args) == 0) {
    cat(
      " Usage: Rscript run_comparisons.R <try_seed_min> <try_seed_max> <output_path> <run_suffix> <optim_method>\n",
      "<!> No arguments provided\n",
      "\tUsing default seeds: 1:500\n",
      "\tUsing default output path: './output/'\n",
      "\tUsing default run suffix: today's date\n",
      "\tUsing default optimisation algorithm: Nelder-Mead\n"
    )
    1 -> try_seed_min
    500 -> try_seed_max
    "./output/" -> output_path
    Sys.Date() -> run_suffix
    "Nelder-Mead" -> optim_method
  } else if (length(args) != 5) {
    cat(
      "Usage: Rscript run_comparisons.R <try_seed_min> <try_seed_max> <run_suffix> <optim_method>\n"
    )
    stop("Arguments not provided correctly")
  } else {
    try_seed_min <- as.integer(args[1])
    try_seed_max <- as.integer(args[2])
    output_path <- as.character(args[3])
    run_suffix <- as.character(args[4])
    optim_method <- as.character(args[5])
  }
}

# ──────────────────────────────────────────────────────────────────────────────
# LIBRARIES                                                                 ----
# ──────────────────────────────────────────────────────────────────────────────

library(plyr)       # Data formatting
library(tidyverse)  # Data formatting
library(gridExtra)  # Plot
library(glue)       # Python equivalent of f"text {variable}"
library(expm)       # Exponential of a matrix
library(jsonlite)   # Manage json files
library(unmarked)   # BP model
library(progress)   # Progress bar
library(latex2exp)  # LaTeX to R expression for plots

# ──────────────────────────────────────────────────────────────────────────────
# FUNCTIONS                                                                 ----
# ──────────────────────────────────────────────────────────────────────────────

# Source all other functions
source("./utils/comparison_of_occupancy_models.R")


# ──────────────────────────────────────────────────────────────────────────────
# COMPARISONS TO DO                                                         ----
# ──────────────────────────────────────────────────────────────────────────────

# Fixed values
NbSites = 100
NbDeployPerSite = 1
DeployementTimeValues = 100

# Seeds
if (!is.na(try_seed_min) & !is.na(try_seed_max)) {
  try_seed = try_seed_min:try_seed_max
}

# List the occupancy probabilities of the different simulation scenario
try_psi = c(0.1, 0.25, 0.5, 0.75, 0.9)

# List the detection parameters of the different simulation scenarios
# With lambda (detection rates) and mu (switching rates)
# 'lambda_1, lambda_2 ; mu_12, mu_21'
try_lambda_mu = c(
  "0, 1 ; 0.0667, 24",
  "0, 5 ; 0.0667, 24",
  "0, 1 ; 0.0667, 1",
  "0.25, 0.25 ; 0.0667, 0.1",
  "0, 5 ; 0.0667, 1",
  "0, 1 ; 0.0667, 0.1",
  "0, 5 ; 0.0667, 0.1"
)

# List the discretisations to try with each simulation scenario
try_SessionLength = c(
  30, # 1 month
  7,  # 1 week
  1  # 1 day
  # 1/24 # 1 hour
)


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT FILE                                                               ----
# ──────────────────────────────────────────────────────────────────────────────

# Result file
result_file = paste0(
  output_path,
  "/",
  "OccModComp",
  "_S", NbSites, 
  "_R", NbDeployPerSite, 
  "_T", DeployementTimeValues, 
  "_seed", min(try_seed), "-", max(try_seed),
  "_", optim_method,
  "_", run_suffix,
  ".json"
) %>% str_replace(., "//", "/")

# Check if output directory exists
if (!dir.exists(output_path)) {
  if (output_path == './output/') {
    dir.create(output_path)
  } else {
    stop(glue::glue('The ouput path {output_path} does not exist.'))
  }
}

# ──────────────────────────────────────────────────────────────────────────────
# experimental_plan                                                         ----
# ──────────────────────────────────────────────────────────────────────────────

# List all the simulation scenarios to run
experimental_plan <- expand.grid(try_psi, try_lambda_mu, try_SessionLength, try_seed) %>%
  setNames(c('psi', 'lambdamu', 'SessionLength', 'seed')) %>%
  rowwise() %>%
  tidyr::separate_wider_delim(
    cols = lambdamu,
    delim = ' ; ',
    names = c('lambda', 'mu'),
    cols_remove = TRUE
  ) %>% 
  tidyr::separate_wider_delim(
    cols = lambda,
    delim = ', ',
    names = c('lambda_1', 'lambda_2'),
    cols_remove = TRUE
  ) %>% 
  tidyr::separate_wider_delim(
    cols = mu,
    delim = ', ',
    names = c('mu_12', 'mu_21'),
    cols_remove = TRUE
  ) %>% 
  mutate_at(vars(lambda_1, lambda_2, mu_12, mu_21), as.numeric) %>%
  mutate(SessionLength = format(
    round(SessionLength, 4),
    nsmall = 4,
    scientific = F
  )) 


# Run continuous?
# Because there are three discretisations tested for the discrete models,
# we can reduce the number of simulations ran by removing the  simulations
# that would produce equivalent analysis for continuous models.
# Example of what we want:
# psi   lambda_1    lambda_2    mu_12     mu_21   SessionLength     seed    run_continuous
# 0.1   0           1           0.0667    24      30.0000           0       TRUE
# 0.1   0           1           0.0667    24      7.0000            0       FALSE
# 0.1   0           1           0.0667    24      1.0000            0       FALSE
experimental_plan$run_continuous <- FALSE
experimental_plan[
  experimental_plan$SessionLength == format(
    round(try_SessionLength[1], 4), nsmall = 4, scientific = F),
  "run_continuous"] <- TRUE


# Calculation of the detection probability corresponding to the parameters
unique_experimental_plan = unique(experimental_plan %>% select(lambda_1, lambda_2, mu_12, mu_21))
unique_experimental_plan$param.p100 = NA
unique_experimental_plan$pi_1 = NA
unique_experimental_plan$pi_2 = NA
unique_experimental_plan$avg_Nij_zi1_per_day = NA
for (i in 1:nrow(unique_experimental_plan)) {
  
  # p100 = probability of having at least one detection during the 100 days of
  # the deployment
  unique_experimental_plan$param.p100[i] = get_p_from_2MMPP_param(
    lambda = c(unique_experimental_plan[i, ]$lambda_1, unique_experimental_plan[i, ]$lambda_2),
    mu = c(unique_experimental_plan[i, ]$mu_12, unique_experimental_plan[i, ]$mu_21),
    pT = DeployementTimeValues
  )
  
  unique_experimental_plan$pi_1[i] <- unique_experimental_plan[i,]$mu_21 / 
    (unique_experimental_plan[i,]$mu_12 + unique_experimental_plan[i,]$mu_21)
  
  unique_experimental_plan$pi_2[i] <- unique_experimental_plan[i,]$mu_12 / 
    (unique_experimental_plan[i,]$mu_12 + unique_experimental_plan[i,]$mu_21)
  
  unique_experimental_plan$avg_Nij_zi1_per_day[i] <-
    unique_experimental_plan[i,]$pi_1 * unique_experimental_plan[i,]$lambda_1 + unique_experimental_plan[i,]$pi_2 * unique_experimental_plan[i,]$lambda_2
}

unique_experimental_plan$avg_Nij_zi1 <- unique_experimental_plan$avg_Nij_zi1_per_day * DeployementTimeValues
if (interactive()) {
  print(tibble(unique_experimental_plan) %>% arrange(avg_Nij_zi1))
}

experimental_plan <- left_join(
  experimental_plan,
  unique_experimental_plan,
  by = c("lambda_1", "lambda_2", "mu_12", "mu_21")
)


# ──────────────────────────────────────────────────────────────────────────────
# COMPARISONS ALREADY DONE                                                  ----
# ──────────────────────────────────────────────────────────────────────────────

if (file.exists(result_file)) {
  results = read_json(result_file,
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
}

if (exists("results")) {
  comparison_already_done = results %>%
    mutate(
      param.seed = format(round(param.seed, 4),
                         nsmall = 0,
                         scientific = F),
      
      param.psi = format(round(param.psi, 4),
                         nsmall = 4,
                         scientific = F),
      
      param.lambda_1 = format(round(param.lambda_1, 4),
                              nsmall = 4,
                              scientific = F),
      param.lambda_2 = format(round(param.lambda_2, 4),
                              nsmall = 4,
                              scientific = F),
      
      param.mu_12 = format(round(param.mu_12, 4),
                           nsmall = 4,
                           scientific = F),
      param.mu_21 = format(round(param.mu_21, 4),
                           nsmall = 4,
                           scientific = F),
      
      param.SessionLength = format(round(param.SessionLength, 4),
                                   nsmall = 4,
                                   scientific = F)
      
    ) %>%
    select(
      param.seed,
      param.psi,
      param.lambda_1,
      param.lambda_2,
      param.mu_12,
      param.mu_21,
      param.SessionLength
    ) %>%
    apply(MARGIN = 1,
          FUN = paste,
          collapse = ", ") %>% 
    str_replace(pattern = "  ", replacement = " ") %>% 
    str_replace(pattern = "  ", replacement = " ") %>% 
    str_trim()
  cat("\n\n",length(comparison_already_done), "comparisons already done\n\n")
} else {
  comparison_already_done = c()
}


cat(glue::glue(
  "{nrow(experimental_plan)} runs to do in total\n",
  "--> {experimental_plan %>% select(-seed) %>% unique() %>% nrow()} different sets of parameters\n",
  "--> {sum(experimental_plan$run_continuous)} with continuous models\n",
  "\n",
  "Listing comparisons not already done...\n",
  "\n"
))


## Checking all comparisons that are already done ------------------------------
# To remove them and only keep the comparisons that have to run for the next step
if (length(comparison_already_done) == 0) {
  comparisons_to_run = 1:nrow(experimental_plan)
} else {
  comparisons_to_run = rep(NA,
                           nrow(experimental_plan) - length(comparison_already_done))
  cpt = 1
  for (comparison in 1:nrow(experimental_plan)) {
    param_for_comparison = experimental_plan[comparison,] %>%
      select(seed, psi, lambda_1, lambda_2, mu_12, mu_21, SessionLength) %>%
      mutate(
        psi = format(round(psi, 4),
                     nsmall = 4,
                     scientific = F),
        
        lambda_1 = format(
          round(lambda_1, 4),
          nsmall = 4,
          scientific = F
        ),
        lambda_2 = format(
          round(lambda_2, 4),
          nsmall = 4,
          scientific = F
        ),
        
        mu_12 = format(round(mu_12, 4),
                       nsmall = 4,
                       scientific = F),
        mu_21 = format(round(mu_21, 4),
                       nsmall = 4,
                       scientific = F)
      ) %>%
      apply(MARGIN = 1,
            FUN = paste,
            collapse = ", ") %>%
      str_replace(pattern = "  ", replacement = " ")
    
    
    do_comparison = (param_for_comparison %in% comparison_already_done) == FALSE |
      experimental_plan[comparison, "seed"] == "NULL"
    # Manual exploration of comparison_already_done:
    # as.data.frame(do.call(rbind, comparison_already_done %>% str_split(", "))) %>% mutate(txt=comparison_already_done) %>% View()
    
    if (do_comparison) {
      comparisons_to_run[cpt] <- comparison
      cpt = cpt + 1
    }
  }
  
  comparisons_to_run = comparisons_to_run[!is.na(comparisons_to_run)]
}

# cat(
#   glue::glue(
#     "Result file: {result_file}\n",
#     "--> {ifelse(exists('results'), 'The result file already exists.', 'The result file does not exist yet.')}\n",
#     "--> {length(comparison_already_done)} comparisons already done\n",
#     "--> {length(comparisons_to_run)} comparisons left to run\n\n\n",
#   )
# )

# ──────────────────────────────────────────────────────────────────────────────
# RUN COMPARISONS                                                           ----
# ──────────────────────────────────────────────────────────────────────────────

pb = newpb(total = length(comparisons_to_run), txt = "Comparison")
invisible(pb$tick(0))
for (comparison in comparisons_to_run) {
  cat("\n", format(Sys.time(), "%d/%m/%Y at %H:%M:%S"), "\n", sep = "")
  
  run_continuous_models = experimental_plan[comparison, ]$run_continuous
  
  # Retrieve parameters
  psi = experimental_plan$psi[comparison]
  lambda = c("lambda_1" = experimental_plan$lambda_1[comparison],
             "lambda_2" = experimental_plan$lambda_2[comparison])
  mu = c("mu_12" = experimental_plan$mu_12[comparison],
         "mu_21" = experimental_plan$mu_21[comparison])
  SessionLength = as.numeric(experimental_plan$SessionLength[comparison])
  seed = unlist(ifelse(
    experimental_plan$seed[comparison] != "NULL",
    as.integer(as.character(experimental_plan$seed[comparison])),
    list(NULL)
  ))
  
  # Print
  cat(
    paste0(
      "      psi = ",
      psi,
      " ; SessionLength = ",
      SessionLength,
      " ; seed = ",
      seed,
      "\n",
      "      lambda = ",
      deparse(unname(lambda)),
      " ; mu = ",
      deparse(unname(mu)),
      "\n"
    )
  )
  
  # Run
  res = run_one_comparison(
    NbSites = NbSites,
    NbDeployPerSite = NbDeployPerSite,
    DeployementTimeValues = DeployementTimeValues,
    psi = psi,
    lambda = lambda,
    mu = mu,
    SessionLength = SessionLength,
    ComparisonResult_ExportPath = result_file,
    optim_method = optim_method,
    quiet = TRUE,
    seed = seed,
    run_continuous_models = run_continuous_models,
    run_discrete_models = TRUE
  )
  
  comparison_already_done = c(
    comparison_already_done,
    experimental_plan[comparison,] %>%
      select(seed, psi, lambda_1, lambda_2, mu_12, mu_21, SessionLength) %>%
      mutate(
        psi = format(round(psi, 4),
                     nsmall = 4,
                     scientific = F),
        
        lambda_1 = format(round(lambda_1, 4),
                          nsmall = 4,
                          scientific = F),
        lambda_2 = format(round(lambda_2, 4),
                          nsmall = 4,
                          scientific = F),
        
        mu_12 = format(round(mu_12, 4),
                       nsmall = 4,
                       scientific = F),
        mu_21 = format(round(mu_21, 4),
                       nsmall = 4,
                       scientific = F)
      ) %>%
      apply(
        MARGIN = 1,
        FUN = paste,
        collapse = ", "
      ) %>%
      str_replace(pattern = "  ", replacement = " ")
  )
  comparisons_to_run = comparisons_to_run[-which(comparisons_to_run == comparison)]
  
  pb$tick()
}
