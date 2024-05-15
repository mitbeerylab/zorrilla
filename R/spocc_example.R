library(spOccupancy)

inv_logit <- function(x) {
  exp(x)/(1+exp(x))
}

# Generate site occupancy
nsites <- 200 # number of sites
occ_covar <- rnorm(n = nsites, mean = 0, sd = 1)
occ_beta0 <- -1
occ_beta1 <- 1
lin_occ_function <- occ_beta0 + occ_beta1*occ_covar
prop_occupancy <- inv_logit(lin_occ_function)
z <- rbinom(nsites,1,prop_occupancy) # vector of latent occupancy status for each site

prop_images_present <- 0.025 # given an occupied site, proportion of images where species is present
images_per_day <- 100 # number of camera trap images per day
days_per_year <- 365 # number of days of operation per year
false_positive_lin_function <- -10 # can modify to be a linear model of some sort, for now this gives a very small false positive rate 
false_positive_rate <- inv_logit(false_positive_lin_function) 

detections <- array(numeric(),c(nsites,days_per_year,images_per_day)) 

# Generate detections
for (i in 1:nsites) {
  for (j in 1:days_per_year) {
    for (k in 1:images_per_day) {
      if (z[i] == 0) { # Unoccupied sites will only have false positives
        detections[i,j,k] <- rbinom(1, size = 1, false_positive_rate)
      } else { # Occupied sites
        detections[i,j,k] <- rbinom(1, size = 1, p = false_positive_rate + prop_images_present)
      }
    }
  }
}

site_daily_detections_sum <- apply(detections, c(1,2), sum)
site_daily_detections_binary <- apply(detections, c(1,2), max)

# inputs needed for spoccupancy
occ.formula <- ~occ_covar
det.formula <- ~1
inits <- list(alpha = 0, 
              beta = 0, 
              z = apply(site_daily_detections_binary, 1, max, na.rm = TRUE))
priors <- list(alpha.normal = list(mean = 0, var = 3), 
               beta.normal = list(mean = 0, var = 3))
n.samples <- 5000
n.burn <- 3000
n.thin <- 2
n.chains <- 3
input_data = list( y = site_daily_detections_binary, occ.covs = as.data.frame(occ_covar))

# fit a basic model
out <- PGOcc(occ.formula = occ.formula, 
             det.formula = det.formula, 
             data = input_data, 
             inits = inits, 
             n.samples = n.samples, 
             priors = priors, 
             n.omp.threads = 1, 
             verbose = TRUE, 
             n.report = 1000, 
             n.burn = n.burn, 
             n.thin = n.thin, 
             n.chains = n.chains)

summary(out)

#Posterior predictive check
ppc.out <- ppcOcc(out, fit.stat = 'freeman-tukey', group = 1)
summary(ppc.out)

# Posterior predictive check
ppc.out <- ppcOcc(out, fit.stat = 'freeman-tukey', group = 1)
summary(ppc.out)
