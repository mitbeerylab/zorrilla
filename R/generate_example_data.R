nsites <- 200 # number of sites
prop_occupancy <- 0.5 # proportion of sites occupied
z <- rbinom(nsites,1,0.5) # vector of latent occupancy status for each site
prop_images_present <- 0.1 # given an occupied site, proportion of images where species is present
images_per_day <- 100 # number of camera trap images per day
days_per_year <- 364 # number of days of operation per year
false_positive_rate <- 0 # Not useful now, but if we want this later
detection_threshold <- 0 # Threshold for logit scores for generating detections
pos_logit_score_mean <- 1 # Setting a mean for the normal distribution for logit scores for positives
pos_logit_score_sd <- 1 # Setting the SD for the normal distribution for logit scores for positives

class_logit_score <- array(numeric(),c(nsites,days_per_year,images_per_day)) 
class_p_score <- array(numeric(),c(nsites,days_per_year,images_per_day)) 
detections <- array(numeric(),c(nsites,days_per_year,images_per_day)) 


for (i in 1:nsites) {
  for (j in 1:days_per_year) {
    for (k in 1:images_per_day) {
      if (z[i] == 0) {
        class_p_score[i,j,k] <- runif(1, min = false_positive_rate, max = min(false_positive_rate + 0.1, 0.5))
        class_logit_score[i,j,k] <- log(class_p_score[i,j,k]/(1 - class_p_score[i,j,k]))
        detections[i,j,k] <- ifelse(class_logit_score[i,j,k]  >= detection_threshold, 1, 0) 
      } else {
        if (rbinom(1,1,prop_images_present) == 1) {
          class_logit_score[i,j,k] <- rnorm(1, mean = pos_logit_score_mean, sd = pos_logit_score_sd)
          class_p_score[i,j,k] <- 1/(1+exp(-class_logit_score[i,j,k]))
          detections[i,j,k] <- ifelse(class_logit_score[i,j,k]  >= detection_threshold, 1, 0) 
        } else {
          class_p_score[i,j,k] <- runif(1, min = false_positive_rate, max = min(false_positive_rate + 0.1, 0.5))
          class_logit_score[i,j,k] <- log(class_p_score[i,j,k]/(1 - class_p_score[i,j,k]))
          detections[i,j,k] <- ifelse(class_logit_score[i,j,k]  >= detection_threshold, 1, 0) 
        }
      }
    }
  }
}

# Example aggregations
site_daily_detections_sum <- apply(detections, c(1,2), sum)
site_daily_detections_binary <- apply(detections, c(1,2), max)

site_daily_max_logit <- apply(class_logit_score, c(1,2), max)
site_daily_mean_logit <- apply(class_logit_score, c(1,2), mean)
