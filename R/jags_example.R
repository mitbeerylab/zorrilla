library(jagsUI)

# Generate occupancy and site-level covariates
n_sites <- 100 # number of sites
site_cov <- rnorm(n_sites)
beta1 <- 1 # slope for occupancy logistic regression
beta0 <- -1 # intercept for occupancy logistic regression
psi <- 0.2 # proportion of sites occupied (only used if not using logistic regression to generate psi)
psi_cov <- 1/(1+exp(-(beta0 + beta1*site_cov)))
z <- rbinom(n=n_sites,1,psi_cov) # vector of latent occupancy status for each site

# Generate detection data
deployment_days_per_site = 120
prob_detection <- 0.3 # probability of detecting species of interest 
prob_fp <- 0.01 # propbability of a false positive for a given time point
session_duration <- 7 # 1, 7, or 30
time_periods = round(deployment_days_per_site/session_duration)

# Create matrix of detections
dfa = matrix(,n_sites,time_periods)

for (i in 1:n_sites){
  # According to the Royle model in unmarked, false positives are generated only if the site is unoccupied
  # Note this is different than how we think about false positives being a random occurrence per image.
  # For now, this is generating positive/negative per time period, which is different than per image. 
  dfa[i,] = rbinom(n = time_periods, size = 1, prob = (prob_detection*z[i] + prob_fp*(1-z[i])))
}


y <- (as.matrix(dfa) >= 1) * 1
site_covs <- data.frame(site_cov = site_cov)


modfile_fp <- tempfile()
writeLines(
  "
model {

  # Priors
  p11 ~ dbeta(2, 2) # Probability of detection p11 = Pr(y = 1 | z = 1)  
  p01 ~ dbeta(4,1)  # Probability of a false positive
  beta0 ~ dunif(-5, 5) 
  beta1 ~ dunif(-5, 5)

  # Likelihood 
  for (i in 1:n_sites) { # Loop over sites
    # Occupancy process
    logit(psi[i]) <- beta0 + beta1*site_covar[i] 
    z[i] ~ dbern(psi[i]) 

    # Detection process
    p_det[i] <- z[i]*p11 + p01 
    for(j in 1:time_periods) {
      y[i,j] ~ dbern(p_det[i]) # Observed occ. data (if available)
    }
  }
  # Estimate proportion of occupied sites
  NOcc <- sum(z[])
  PropOcc <- NOcc/n_sites
}
",con=modfile_fp
)


modfile_nofp <- tempfile()
writeLines(
  "
model {

  # Priors
  p11 ~ dbeta(2, 2) # p11 = Pr(y = 1 | z = 1) 
  beta0 ~ dunif(-5, 5) 
  beta1 ~ dunif(-5, 5)

  # Likelihood
  for (i in 1:n_sites) { # Loop over sites
    # Occupancy process
    logit(psi[i]) <- beta0 + beta1*site_covar[i] 
    z[i] ~ dbern(psi[i]) # Latent occupancy states

    # Detection process
    p_det[i] <- z[i]*p11 # Detection probability does not include false positives in this model
    for(j in 1:time_periods) {
      y[i,j] ~ dbern(p_det[i]) 
    }
  }
  # Estimate proportion of occupied sites
  NOcc <- sum(z[])
  PropOcc <- NOcc/n_sites
}
",con=modfile_nofp
)

# Get initial values for parameters/occupancy
zst <- rep(1, n_sites)
inits <- function() {
  list(
    z = zst,
    p11 = runif(1, 0.1, 0.5),
    p01 = runif(1, 0.001, 0.1),
    beta0 = 0,
    beta1 = 1
  )
}

# List of parameters we want to have in the results
monitored <- c(
  "PropOcc",
  "beta0",
  "beta1",
  "p11",
  "p01"
)

na <- 100
ni <- 500
nt <- 10
nb <- 100
nc <- 5

# Jags takes a list of named numeric arrays as inputs - must include any data that is input into the model file
jagsData = list(y=y, site_covar=site_covs$site_cov, n_sites=n_sites, time_periods=time_periods)
set.seed(123)

jagsResult_nofp <- jagsUI::jags(
  jagsData,
  inits,
  monitored,
  modfile_nofp,
  n.adapt = na,
  n.chains = nc,
  n.thin = nt,
  n.iter = ni,
  n.burnin = nb,
  parallel = TRUE,
)
print(jagsResult_nofp)
mean(z) # average occupancy to compare the estimated proportion of occupied sites to

jagsResult_fp <- jagsUI::jags(
  jagsData,
  inits,
  monitored,
  modfile_fp,
  n.adapt = na,
  n.chains = nc,
  n.thin = nt,
  n.iter = ni,
  n.burnin = nb,
  parallel = TRUE,
)

print(jagsResult_fp)

# Compare to unmarked
library(unmarked)
umf <- unmarkedFrameOccuFP(y=y, type=c(0,dim(y)[2],0), siteCovs=site_covs)

# Model with false positives, but no covariates
m1 <- occuFP(detformula = ~ 1, FPformula = ~ 1, stateformula = ~ 1, data=umf, method="Nelder-Mead", se = TRUE)
summary(m1)
backTransform(m1, 'state')
backTransform(m1, 'det')
backTransform(m1, 'fp')


#Model with false positives and occupancy covariates
m1b <- occuFP(detformula = ~ 1, FPformula = ~ 1, stateformula = ~ site_cov, data=umf, method="Nelder-Mead", se = TRUE)
summary(m1b)
lc1b<- linearComb(m1b, c(1, mean(site_cov)), type = "state")
backTransform(lc1b)
backTransform(m1b, 'det')
backTransform(m1b, 'fp')

# No false positives
umf2 <- unmarkedFrameOccu(y=y, siteCovs=site_covs)

m2 <- occu(~1 ~ 1, data=umf2, method="Nelder-Mead", se = TRUE, linkPsi="logit")
summary(m2)
backTransform(m2, 'state')
backTransform(m2, 'det')

m2b <- occu(~1 ~site_cov, data=umf2, method="Nelder-Mead", se = TRUE, linkPsi="logit")
summary(m2b)
lc2b<- linearComb(m2b, c(1, mean(site_cov)), type = "state")
backTransform(lc2b)
