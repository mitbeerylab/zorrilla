## ----setup, include=FALSE-----------------------------------------------------
# Rmd chunks options
knitr::opts_chunk$set(
  echo = TRUE, # show code of the chunk and chunk output
  warning = FALSE, message = FALSE, # do not display warnings and messages as chunk outputs in HTML
  collapse = FALSE,
  comment = "", # no comment character for the chunk text outputs
  fig.height = 8,
  fig.width = 12,
  fig.align = "center",
  out.width = "100%", # responsive width for chunk outputs (figures, ...)
  class.source = 'fold-hide'
)

# Interactive HTML Dataframe
myDT <- function(df, rownames = FALSE, pageLength = 10, caption = NULL, autoWidth = FALSE, fixedColumns = list(leftColumns=2), ...) {
  if (!is.null(caption)) {
    dt = DT::datatable(
      df,
      filter = 'top',
      rownames = rownames,
      extensions = 'Buttons',
      escape = FALSE,
      options = list(
        autoWidth = autoWidth,
        scrollX = TRUE,
        pageLength = pageLength,
        dom = 'Blfrtip',
        buttons = c('copy', 'csv', 'excel', 'pdf'),
        lengthMenu = list(c(10, 25, 50, -1),
                          c(10, 25, 50, "All")),
        fixedColumns = fixedColumns
      ),
      caption = htmltools::tags$caption(
          style = "
            text-align: left;
            font-size: 1.5em; font-weight: bold;
            background-color:#a5c73dff;  color:white;
            padding: 0.3em; padding-left: 1em;
          ",
          caption
       ),
      ...
    )
  } else {
    dt = DT::datatable(
      df,
      filter = 'top',
      rownames = rownames,
      extensions = 'Buttons',
      escape = FALSE,
      options = list(
        autoWidth = autoWidth,
        scrollX=TRUE,
        pageLength = pageLength,
        dom = 'Blfrtip',
        buttons = c('copy', 'csv', 'excel', 'pdf'),
        lengthMenu = list(c(10, 25, 50, -1),
                          c(10, 25, 50, "All")),
        fixedColumns = fixedColumns
      ),
      ...
    )
  }

  return(dt)
}



## -----------------------------------------------------------------------------
# Data visualisation
library(ggplot2)
library(ggpubr)
library(scales)
library(ggnewscale)
theme_set(theme_minimal(base_size = 13))

# Modelling packages
library(unmarked)

# Data management
library(tidyverse)
library(lubridate)

# Source all custom functions
source("./utils/comparison_of_occupancy_models.R")

# Locale
Sys.setenv(LANG = "en_US.UTF-8")
Sys.setenv("LANGUAGE" = "EN")


## -----------------------------------------------------------------------------
if (!dir.exists("./data/")){
  dir.create("./data/")
}

# Download the data from the Ain county
if (!file.exists("./data/metadata_Ain.RData")) {
  download.file(
    "https://github.com/oliviergimenez/computo-deeplearning-occupany-lynx/raw/master/dat/metadata_Ain.RData",
    "./data/metadata_Ain.RData"
  )
}

# Load data from Ain
# load('./data/metadata_Ain.RData')
# load('./data/metadata_Ain_thres_0.5.RData')
# load('./data/metadata_iwildcam_2022_thres_0.001.RData')
load('./data/metadata_iwildcam_2022_tmp_v3.RData')
allpic <- allfiles %>%
  mutate(Keywords = as.character(observed)) %>% # pick manual tags
  mutate(DateTime = ymd_hms(str_replace(DateTimeOriginal, "2019:02:29", "2019:03:01"))) %>%
  mutate(FileName = pix) %>% 
  select(FileName, Keywords, DateTime)
rm(allfiles)

# Note: 4 rows failed to parse when we used "ymd_hms(DateTimeOriginal)"
#       because date in DateTimeOriginal are 2019:02:29 (format %y:%m:%d)
#       and there February only had 28 days in 2019...
# Since we will not use them, because they are not lynx detections,
# we don't have to investigate this further. 
# We just replaced 2019:02:29 by 2019:03:01 because it's the most likely error.

# Add SiteID
allpic$SiteID = gsub("_.*$", "", allpic$FileName)

# Remove columns that we will not use
allpic = allpic %>% 
  select(SiteID, FileName, Keywords, DateTime)


## -----------------------------------------------------------------------------
## We list all the labels that are used
(all_labels <- sort(unique(c(
 allpic$Keywords[!grepl('"', allpic$Keywords)], unname(unlist(sapply(unique(allpic$Keywords), function(x) {
   gsub('"', '', regmatches(x, gregexpr('"([^"]*)"', x))[[1]])
 })))
))))

# Grouping some labels
human_labels = c("cavalier", "chasseur", "chien", "Fréquentation humaine", "humain", "vehicule", "véhicule")
squirrel_labels = c("ecureuil", "écureuil", "Sciuridae")
small_mustelids_labels = c("fouine", "martre")
lagomorph_labels = c("laporidés", "lievre", "lièvre")

## All the sites and deployment times
# + I added the number of detection events per species
allsites = allpic %>%
  mutate(
    NbLynx = grepl(pattern = "lynx", x = Keywords),
    NbHuman = sapply(Keywords, function(lab) {any(grepl(paste(human_labels, collapse = "|"), lab))}),
    NbBadger = grepl(pattern = "blaireaux", x = Keywords),
    NbRedDeer = grepl(pattern = "cerf", x = Keywords),
    NbChamois = grepl(pattern = "chamois", x = Keywords),
    NbRoeDeer = grepl(pattern = "chevreuil", x = Keywords), 
    NbSquirrel = sapply(Keywords, function(lab) {any(grepl(paste(squirrel_labels, collapse = "|"), lab))}),
    NbSmallMustelid = sapply(Keywords, function(lab) {any(grepl(paste(small_mustelids_labels, collapse = "|"), lab))}),
    NbLagomorph = sapply(Keywords, function(lab) {any(grepl(paste(lagomorph_labels, collapse = "|"), lab))}),
    NbFox = grepl(pattern = "renard", x = Keywords), 
    NbWildBoar = grepl(pattern = "sangliers", x = Keywords),
    NbWildcat = grepl(pattern = "chat forestier", x = Keywords)
  ) %>%
  group_by(SiteID) %>%
  dplyr::summarise(
    FirstPic = min(DateTime),
    LastPic = max(DateTime),
    across(starts_with("Nb"), ~ sum(.x)),
    .groups = "keep"
  ) %>% 
  mutate(DeploymentDuration = LastPic - FirstPic) %>% 
  relocate(DeploymentDuration, .before = FirstPic)

allsites %>% 
  mutate(DeploymentDuration = round(DeploymentDuration, 2)) %>%
  mutate(across(where(is.character), \(x) as.factor(x))) %>% 
  mutate(across(where(is.numeric), \(x) round(x, 2))) %>%
  myDT(caption = "Monitoring periods and detections per site", pageLength=11)


## ----plot_nb_detec_per_site_and_species_all-----------------------------------
allsites %>% 
  tidyr::pivot_longer(cols = starts_with("Nb"),
               names_to = c("Species"),
               names_prefix = "Nb",
               values_to = "NbDetections") %>% 
  dplyr::group_by(Species) %>% 
  dplyr::mutate(SpeciesTxt = paste0(
    str_replace(gsub("(?<!^)(?=[A-Z])", " ", Species, perl=TRUE), "\\s(\\w+)", function(x) {tolower(x)}),
    "\n",
    sum(NbDetections == 0), "/", length(unique(allsites$SiteID)),
    " site", ifelse(sum(NbDetections == 0) > 1, "s", ""),
    " with no detection"
  )) %>% 
  ggplot() +
  geom_histogram(aes(x = NbDetections, y = after_stat(count), fill = NbDetections > 0), binwidth = 1) +
  geom_density(aes(x = NbDetections, y = after_stat(count))) +
  facet_wrap(SpeciesTxt ~ ., scales = "free") +
  labs(x = "Number of images with the species",
       y = "Number of sites")+
  theme(legend.position = "none")


## -----------------------------------------------------------------------------
LynxPic = allpic %>% 
  filter(grepl(pattern = "lynx", x = Keywords))


## -----------------------------------------------------------------------------
# Beginning of the sessions
BeginDateTime = min(allsites$FirstPic)
EndDateTime = max(allsites$LastPic)

# Month sessions
MonthSessions = seq(
  floor_date(BeginDateTime, unit = "month"),
  ceiling_date(EndDateTime, unit = "month"),
  by = "months"
)
MonthSessionsLabels = format(MonthSessions, "%b %y", locale = "en_GB")
LynxPic$MonthSession = cut(
  LynxPic$DateTime,
  breaks = MonthSessions,
  labels = MonthSessionsLabels[-length(MonthSessionsLabels)],
  include.lowest = FALSE
)

# Week sessions
WeekSessions = seq(
  floor_date(BeginDateTime, unit = "month"),
  ceiling_date(EndDateTime, unit = "month"),
  by = "week"
)
WeekSessionsLabels = format(WeekSessions, "%Y week %U", locale = "en_GB")
LynxPic$WeekSession = cut(
  LynxPic$DateTime,
  breaks = WeekSessions,
  labels = WeekSessionsLabels[-length(WeekSessionsLabels)],
  include.lowest = FALSE
)

# Day sessions
DaySessions = seq(
  floor_date(BeginDateTime, unit = "month"),
  ceiling_date(EndDateTime, unit = "month"),
  by = "day"
)
DaySessionsLabels = format(DaySessions, "%d %B %Y", locale = "en_GB")
LynxPic$DaySession = cut(
  LynxPic$DateTime,
  breaks = DaySessions,
  labels = DaySessionsLabels[-length(DaySessionsLabels)],
  include.lowest = FALSE
)

# We now have the session for each photo of a lynx
LynxPic %>% 
  select(-FileName, -Keywords) %>% 
  print()


## ----data_formatting_continuous-----------------------------------------------
# Number of sites
NbSites = length(allsites$SiteID)

print("Number of sites:")
print(NbSites)

# List of the detection times
# format: detection_times[[site i]][[deployment j]] -> vector of detection times
# Because we only have one deployment per site, the format is:
#  detection_times[[site i]][[1]] -> vector of detection times
#  
#  Detection times are numerics, between 0 and the latest detection, 
#  in the chosen time-unit

# list_T_ij
# Duration of deployment j at site i
# list_T_ij[[site i]] -> vector of the R_i duration of deployment(s) at site i

# Initialisation detection_times
detection_times <- vector(mode = "list", length = NbSites)
names(detection_times) <- allsites$SiteID

# Initialisation list_T_ij
list_T_ij <- vector(mode = "list", length = NbSites)
names(list_T_ij) <- allsites$SiteID

for (i in 1:NbSites) {
  i_siteID = names(detection_times)[i]
  
  # Date and time of the beginning and end of the deployment
  i_BeginTime = allsites %>% filter(SiteID == i_siteID) %>% dplyr::pull(FirstPic)
  i_EndTime = allsites %>% filter(SiteID == i_siteID) %>% dplyr::pull(LastPic)
  
  # Date and time of detections
  i_DetecDateTime = sort(LynxPic %>%
         filter(SiteID == i_siteID) %>%
         dplyr::pull(DateTime))
  
  # Days between the beginning of the deployment and each detection event
  i_DetecTimeDays = as.numeric(
    difftime(i_DetecDateTime, i_BeginTime, units = "days")
  )
  
  # Adding the time of detection events
  detection_times[[i]] <- list("1" = i_DetecTimeDays)
  
  # Duration of the deployment (in days too)
  list_T_ij[[i]]<-as.numeric(
    difftime(i_EndTime, i_BeginTime, units = "days")
  )
}

# list_R_i is the number of deployments at sites
# (here 1 deployment only in all sites)
list_R_i = rep(1, NbSites)


## -----------------------------------------------------------------------------
print(detection_times[1:3]) # Time of detection in days since the deployment began


## -----------------------------------------------------------------------------
print(list_T_ij[1:3]) # Deployment duration in days


## -----------------------------------------------------------------------------
# For the paper, we rename sites.
SiteID_2_SiteIDNew = left_join(
  (
    allsites %>%
      tidyr::separate(
        SiteID,
        into = c("SiteIDleft", "SiteIDright"),
        sep = "\\.",
        remove = FALSE
      )
  ),
  (
    allsites %>%
      tidyr::separate(
        SiteID,
        into = c("SiteIDleft", "SiteIDright"),
        sep = "\\.",
        remove = FALSE
      ) %>%
      dplyr::group_by(SiteIDleft) %>%
      dplyr::summarise(NbLynxTot = sum(NbLynx)) %>%
      dplyr::arrange(NbLynxTot) %>%
      dplyr::mutate(SiteIDleftNew = LETTERS[1:nrow(.)])
  ),
  by = "SiteIDleft"
) %>%
  mutate(SiteIDNew = paste0(SiteIDleftNew, SiteIDright)) %>%
  select(SiteID, SiteIDNew)


## ----class.source = 'fold-hide'-----------------------------------------------
## tryCatch({
LynxDetecHistory_continuous = ggplot(data = (
  allsites %>%
    left_join(LynxPic, by = "SiteID") %>%
    left_join(SiteID_2_SiteIDNew, by = "SiteID")
),
aes(x = DateTime, y = SiteIDNew)) +
  geom_segment(aes(
    x = FirstPic,
    xend = LastPic,
    yend = SiteIDNew,
    color = "Monitoring period"
  ), linewidth = 2.5) +
  geom_point(aes(fill = "Detection event"), shape = 4) +
  scale_x_datetime(date_breaks = "3 month" , date_labels = "%b %y") +
  scale_colour_manual(values = c("Monitoring period" = "grey"), name = "") +
  scale_fill_manual(values = c("Detection event" = "black"), name = "") +
  labs(title = "Lynx detection history") +
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 0.3
    ),
    axis.title = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )
print(LynxDetecHistory_continuous)
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


# Export
if (!dir.exists('./output/')) {
  dir.create('./output/')
}
## tryCatch({
ggsave(
  "./output/lynx_detection_history_continuous.pdf",
  plot = (
    LynxDetecHistory_continuous +
      theme(plot.title = element_blank())
  ),
  width = 23,
  height = 10,
  unit = "cm"
)
jpeg(
  filename = "./output/lynx_detection_history_continuous.jpeg",
  width = 23,
  height = 10,
  unit = "cm",
  res = 100
)
print(
  LynxDetecHistory_continuous +
    theme(plot.title = element_blank())
)
dev.off()
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


## ----class.source = 'fold-hide'-----------------------------------------------
DayTable = as_tibble(data.frame("DaySession" = cut(
    DaySessions,
    breaks = DaySessions,
    labels = DaySessionsLabels[-length(DaySessionsLabels)],
    include.lowest = FALSE
  )[-length(DaySessions)],
  "DaySessionBegin" = DaySessions[-length(DaySessions)] + seconds(0),
  "DaySessionEnd" = DaySessions[-1] - seconds(1e-16)
))

DayLongDF <- expand.grid(
  "SiteID" = unique(allsites$SiteID),
  "DaySession" = DayTable$DaySession) %>%
  as_tibble() %>% 
  left_join(allsites,
            by = "SiteID") %>% 
  left_join(DayTable, by = "DaySession") %>% 
  mutate(
    MonitoringTime = replace_na(as.duration(
      lubridate::intersect(
        lubridate::interval(DaySessionBegin, DaySessionEnd),
        lubridate::interval(FirstPic, LastPic)
      )
    ), as.duration(seconds(0))),
    FullyMonitored = (FirstPic <= DaySessionBegin &
                        LastPic >= DaySessionEnd),
    PartiallyMonitored = !FullyMonitored & (MonitoringTime > 0) &
      (MonitoringTime < max(MonitoringTime, na.rm = T))
  ) %>% 
  left_join((
    LynxPic %>%
      group_by(SiteID, DaySession) %>%
      dplyr::summarise(n = n(), .groups = "keep")
  ),
  by = c("SiteID", "DaySession")) %>% 
  mutate(NbDetec = ifelse(FullyMonitored, ifelse(is.na(n), 0, n), NA))


DayCountMatrix <- DayLongDF %>% 
  select(SiteID, DaySession, NbDetec) %>% 
  pivot_wider(names_from = "DaySession", 
              values_from = "NbDetec") %>% 
  column_to_rownames("SiteID")

## tryCatch({
LynxDetecHistory_day = ggplot() +
  geom_tile(
    data = DayLongDF,
    aes(
      x = as.Date(DaySessionBegin),
      y = SiteID,
      fill = ifelse(NbDetec == 0, NA, NbDetec)
    )
  ) +
  scale_fill_gradient(
    low = "#93c47d",
    high = "#274e13",
    name = "Detections",
    breaks = round(seq(
      from = 1,
      to = max(DayLongDF$NbDetec, na.rm = T),
      length.out = 4
    )),
    na.value = "transparent"
  ) +
  new_scale_fill() +
  geom_tile(
    data = (
      DayLongDF %>%
        filter(is.na(NbDetec) | NbDetec == 0) %>%
        mutate(NbDetec0NA = ifelse(
          is.na(NbDetec), "Not monitored", "No detection"
        ))
    ),
    aes(
      x = as.Date(DaySessionBegin),
      y = SiteID,
      fill = NbDetec0NA
    )
  ) +
  scale_fill_manual(
    values = c(
      "No detection" = "#ba3c3c",
      "Not monitored" = "white"
    ),
    name = ""
  ) +
  scale_x_date(date_breaks = "3 month" , date_labels = "%B %Y") +
  labs(title = "Lynx daily detection history") +
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 1
    ),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    legend.position = "bottom"
  )
print(LynxDetecHistory_day)
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})



## -----------------------------------------------------------------------------
DayLongDF %>% 
  dplyr::group_by(SiteID) %>% 
  dplyr::filter(PartiallyMonitored) %>% 
  dplyr::summarise(`Discarded (in days)` = as.character(as.period(as.duration(
    sum(MonitoringTime)
  )))) %>%
  myDT(caption = "Monitored but discarded time per site with daily sessions", pageLength=11)


## ----class.source = 'fold-hide'-----------------------------------------------
# For each monthly session, we check if each site was monitored during the
# entire session ("Monitored = TRUE") or not ("Monitored = FALSE").
# If the site was not monitored during the entire session, observation
# will be NAs.

WeekTable = data.frame("WeekSession" = cut(
    WeekSessions,
    breaks = WeekSessions,
    labels = WeekSessionsLabels[-length(WeekSessionsLabels)],
    include.lowest = FALSE
  )[-length(WeekSessions)],
  "WeekSessionBegin" = WeekSessions[-length(WeekSessions)],
  "WeekSessionEnd" = WeekSessions[-1] - days(1)
)

WeekLongDF <- expand.grid(
  "SiteID" = unique(allsites$SiteID),
  "WeekSession" = WeekTable$WeekSession) %>%
  as_tibble() %>% 
  left_join(allsites,
            by = "SiteID") %>% 
  left_join(WeekTable, by = "WeekSession") %>% 
  mutate(
    MonitoringTime = replace_na(as.duration(
      lubridate::intersect(
        lubridate::interval(WeekSessionBegin, WeekSessionEnd),
        lubridate::interval(FirstPic, LastPic)
      )
    ), as.duration(seconds(0))),
    FullyMonitored = (FirstPic <= WeekSessionBegin &
                        LastPic >= WeekSessionEnd),
    PartiallyMonitored = !FullyMonitored & (MonitoringTime > 0) &
      (MonitoringTime < max(MonitoringTime, na.rm = T))
  ) %>% 
  left_join((
    LynxPic %>%
      group_by(SiteID, WeekSession) %>%
      dplyr::summarise(n = n(), .groups = "keep")
  ),
  by = c("SiteID", "WeekSession")) %>% 
  mutate(NbDetec = ifelse(FullyMonitored, ifelse(is.na(n), 0, n), NA))

# print(WeekLongDF, n=40, width=Inf)
# quit()

WeekCountMatrix <- WeekLongDF %>% 
  select(SiteID, WeekSession, NbDetec) %>% 
  pivot_wider(names_from = "WeekSession", 
              values_from = "NbDetec") %>% 
  column_to_rownames("SiteID")

## tryCatch({
LynxDetecHistory_week = ggplot() +
  geom_tile(
    data = WeekLongDF,
    aes(
      x = as.Date(WeekSessionBegin),
      y = SiteID,
      fill = ifelse(NbDetec == 0, NA, NbDetec)
    ), colour = "grey"
  ) +
  scale_fill_gradient(
    low = "#93c47d",
    high = "#274e13",
    name = "Detections",
    breaks = round(seq(
      from = 1,
      to = max(WeekLongDF$NbDetec, na.rm = T),
      length.out = 4
    )),
    na.value = "transparent"
  ) +
  new_scale_fill() +
  geom_tile(
    data = (
      WeekLongDF %>%
        filter(is.na(NbDetec) | NbDetec == 0) %>%
        mutate(NbDetec0NA = ifelse(
          is.na(NbDetec), "Not monitored", "No detection"
        ))
    ),
    aes(
      x = as.Date(WeekSessionBegin),
      y = SiteID,
      fill = NbDetec0NA
    ), colour = "grey"
  ) +
  scale_fill_manual(
    values = c(
      "No detection" = "#ba3c3c",
      "Not monitored" = "white"
    ),
    name = ""
  ) +
  scale_x_date(date_breaks = "3 month" , date_labels = "%B %Y") +
  labs(title="Lynx weekly detection history")+
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 1
    ),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    legend.position = "bottom"
  )
print(LynxDetecHistory_week)
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})



## -----------------------------------------------------------------------------
WeekLongDF %>% 
  dplyr::group_by(SiteID) %>% 
  dplyr::filter(PartiallyMonitored) %>% 
  dplyr::summarise(`Discarded (in days)` = as.character(as.period(as.duration(
    sum(MonitoringTime)
  )))) %>%
  myDT(caption = "Monitored but discarded time per site with weekly sessions", pageLength=11)


## ----class.source = 'fold-hide'-----------------------------------------------
# For each monthly session, we check if each site was monitored during the
# entire session ("Monitored = TRUE") or not ("Monitored = FALSE").
# If the site was not monitored during the entire session, observation
# will be NAs.

MonthTable = data.frame("MonthSession" = cut(
    MonthSessions,
    breaks = MonthSessions,
    labels = MonthSessionsLabels[-length(MonthSessionsLabels)],
    include.lowest = FALSE
  )[-length(MonthSessions)],
  "MonthSessionBegin" = MonthSessions[-length(MonthSessions)],
  "MonthSessionEnd" = MonthSessions[-1] - days(1)
)

MonthLongDF <- expand.grid(
  "SiteID" = unique(allsites$SiteID),
  "MonthSession" = MonthTable$MonthSession) %>%
  as_tibble() %>% 
  left_join(allsites,
            by = "SiteID") %>% 
  left_join(MonthTable, by = "MonthSession") %>% 
  mutate(
    MonitoringTime = replace_na(as.duration(
      lubridate::intersect(
        lubridate::interval(MonthSessionBegin, MonthSessionEnd),
        lubridate::interval(FirstPic, LastPic)
      )
    ), as.duration(seconds(0))),
    FullyMonitored = (FirstPic <= MonthSessionBegin &
                        LastPic >= MonthSessionEnd),
    PartiallyMonitored = !FullyMonitored & (MonitoringTime > 0) &
      (MonitoringTime < max(MonitoringTime, na.rm = T))
  ) %>% 
  left_join((
    LynxPic %>%
      group_by(SiteID, MonthSession) %>%
      dplyr::summarise(n = n(), .groups = "keep")
  ),
  by = c("SiteID", "MonthSession")) %>% 
  mutate(NbDetec = ifelse(FullyMonitored, ifelse(is.na(n), 0, n), NA))

MonthCountMatrix <- MonthLongDF %>% 
  select(SiteID, MonthSession, NbDetec) %>% 
  pivot_wider(names_from = "MonthSession", 
              values_from = "NbDetec") %>% 
  column_to_rownames("SiteID")

## tryCatch({
LynxDetecHistory_month = ggplot() +
  geom_tile(
    data = MonthLongDF,
    aes(
      x = as.Date(MonthSessionBegin),
      y = SiteID,
      fill = ifelse(NbDetec == 0, NA, NbDetec)
    ), colour = "grey"
  ) +
  scale_fill_gradient(
    low = "#93c47d",
    high = "#274e13",
    name = "Detections",
    breaks = round(seq(
      from = 1,
      to = max(MonthLongDF$NbDetec, na.rm = T),
      length.out = 4
    )),
    na.value = "transparent"
  ) +
  new_scale_fill() +
  geom_tile(
    data = (
      MonthLongDF %>%
        filter(is.na(NbDetec) | NbDetec == 0) %>%
        mutate(NbDetec0NA = ifelse(
          is.na(NbDetec), "Not monitored", "No detection"
        ))
    ),
    aes(
      x = as.Date(MonthSessionBegin),
      y = SiteID,
      fill = NbDetec0NA
    ), colour = "grey"
  ) +
  scale_fill_manual(
    values = c(
      "No detection" = "#ba3c3c",
      "Not monitored" = "white"
    ),
    name = ""
  ) +
  scale_x_date(date_breaks = "3 month" , date_labels = "%B %Y") +
  labs(title = "Lynx monthly detection history") +
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 1
    ),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    legend.position = "bottom"
  )
print(LynxDetecHistory_month)
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


## -----------------------------------------------------------------------------
MonthLongDF %>% 
  dplyr::group_by(SiteID) %>% 
  dplyr::filter(PartiallyMonitored) %>% 
  dplyr::summarise(`Discarded (in days)` = as.character(as.period(as.duration(
    sum(MonitoringTime)
  )))) %>%
  myDT(caption = "Monitored but discarded time per site with monthly sessions", pageLength=11)


## -----------------------------------------------------------------------------

MaxNbDetec_AllDiscretisations = max(DayLongDF$NbDetec,
                                    WeekLongDF$NbDetec,
                                    MonthLongDF$NbDetec,
                                    na.rm = T)


# Month ----

MonthSessionsLabelsTrimester = MonthSessionsLabels
MonthSessionsLabelsTrimester[
  sort(c(seq(from = 2, to = length(MonthSessionsLabels), by = 3),
         seq(from = 3, to = length(MonthSessionsLabels), by = 3)))] <- ""

## tryCatch({
gg_Month = ggplot() +
  geom_tile(
    data = (
      MonthLongDF %>%
        left_join(SiteID_2_SiteIDNew, by = "SiteID") %>%
        filter(is.na(NbDetec) | NbDetec == 0) %>%
        mutate(NbDetec0NA = ifelse(is.na(NbDetec), NA, " "))
    ),
    aes(
      x = reorder(as.factor(format(
        as_date(MonthSessionBegin), "%b %y"
      )),
      as_date(MonthSessionBegin)),
      y = SiteIDNew,
      fill = NbDetec0NA
    )
  ) +
  scale_fill_manual(values = c(" " = "#ba3c3c"),
                    na.value = "transparent",
                    name = "No detection") +
  guides(fill = guide_legend(
    title.position = 'top',
    title.hjust = 0,
    title.vjust = -1
  )) +
  new_scale_fill() +
  geom_tile(data = (MonthLongDF %>% left_join(SiteID_2_SiteIDNew, by = "SiteID")),
            aes(
              x = reorder(as.factor(format(
                as_date(MonthSessionBegin), "%b %y"
              )),
              as_date(MonthSessionBegin)),
              y = SiteIDNew,
              fill = ifelse(NbDetec == 0, NA, NbDetec)
            )) +
  scale_fill_gradient(
    low = "#93c47d",
    high = "#274e13",
    name = "Detection(s)",
    breaks = round(seq(
      from = 1,
      to = MaxNbDetec_AllDiscretisations,
      length.out = 4
    )),
    limits = c(1, MaxNbDetec_AllDiscretisations),
    na.value = "transparent"
  ) +
  guides(fill = guide_colourbar(
    title.position = 'top',
    title.hjust = 0,
    title.vjust = -1,
    barheight = .5
  )) +
  scale_x_discrete(breaks = MonthSessionsLabels, labels = MonthSessionsLabelsTrimester) +
  labs(title = "(a) Monthly discretisation") +
  theme(    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 1
    ),
    axis.title = element_blank(),
    plot.title = element_text(size = 12, hjust = .5),
    panel.grid.major.x = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(size = 12),
    legend.title.align = 0.5
  )
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})

# Week ----
WeekSessionsLabelsTrimester = format(WeekSessions, "%b %y")
WeekSessionsLabelsTrimester[duplicated(WeekSessionsLabelsTrimester)] = ""
WeekSessionsLabelsTrimester[
  !WeekSessionsLabelsTrimester %in% MonthSessionsLabelsTrimester
  ] = ""

## tryCatch({
gg_Week = ggplot() +
    geom_tile(
    data = (
      WeekLongDF %>%
        left_join(SiteID_2_SiteIDNew, by = "SiteID") %>%
        filter(is.na(NbDetec) | NbDetec == 0) %>%
        mutate(NbDetec0NA = ifelse(is.na(NbDetec), NA, "No detection"))
    ),
    aes(
      x =reorder(as.factor(format(as_date(WeekSessionBegin), "%Y week %U")),
                 as_date(WeekSessionBegin)),
      y = SiteIDNew,
      fill = NbDetec0NA
    )
  ) +
  scale_fill_manual(
    values = c("No detection" = "#ba3c3c"),
    na.value = "transparent", name = ""
  ) +
  new_scale_fill() +
  geom_tile(data = (WeekLongDF %>% left_join(SiteID_2_SiteIDNew, by = "SiteID")),
            aes(
              x = reorder(as.factor(format(as_date(WeekSessionBegin), "%Y week %U")),
                          as_date(WeekSessionBegin)),
              y = SiteIDNew,
              fill = ifelse(NbDetec == 0, NA, NbDetec)
            )) +
  scale_fill_gradient(
    low = "#93c47d",
    high = "#274e13",
    name = "Detections",
    breaks = round(seq(
      from = 1,
      to = MaxNbDetec_AllDiscretisations,
      length.out = 4
    )),
    limits = c(1, MaxNbDetec_AllDiscretisations),
    na.value = "transparent"
  ) +
  scale_x_discrete(breaks = WeekSessionsLabels, labels = WeekSessionsLabelsTrimester) +
  labs(title = "(b) Weekly discretisation") +
  theme(    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 1
    ),
    axis.title = element_blank(),
    plot.title = element_text(size = 12, hjust = .5),
    panel.grid.major.x = element_blank(),
    legend.position = "bottom"
  )
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})

# Day ----
## tryCatch({
DaySessionsLabelsTrimester = format(DaySessions, "%b %y")
DaySessionsLabelsTrimester[duplicated(DaySessionsLabelsTrimester)] = ""
DaySessionsLabelsTrimester[
  !DaySessionsLabelsTrimester %in% MonthSessionsLabelsTrimester
  ] = ""

gg_Day = ggplot() +
    geom_tile(
    data = (
      DayLongDF %>%
        left_join(SiteID_2_SiteIDNew, by = "SiteID") %>%
        filter(is.na(NbDetec) | NbDetec == 0) %>%
        mutate(NbDetec0NA = ifelse(is.na(NbDetec), NA, "No detection"))
    ),
    aes(
      x =reorder(as.factor(format(as_date(DaySessionBegin), "%d %B %Y")),
                 as_date(DaySessionBegin)),
      y = SiteIDNew,
      fill = NbDetec0NA
    )
  ) +
  scale_fill_manual(
    values = c("No detection" = "#ba3c3c"),
    na.value = "transparent", name = ""
  ) +
  new_scale_fill() +
  geom_tile(data = (DayLongDF %>% left_join(SiteID_2_SiteIDNew, by = "SiteID")),
            aes(
              x = reorder(as.factor(format(as_date(DaySessionBegin), "%d %B %Y")),
                          as_date(DaySessionBegin)),
              y = SiteIDNew,
              fill = ifelse(NbDetec == 0, NA, NbDetec)
            )) +
  scale_fill_gradient(
    low = "#93c47d",
    high = "#274e13",
    name = "Detections",
    breaks = round(seq(
      from = 1,
      to = MaxNbDetec_AllDiscretisations,
      length.out = 4
    )),
    limits = c(1, MaxNbDetec_AllDiscretisations),
    na.value = "transparent"
  ) +
  scale_x_discrete(breaks = DaySessionsLabels, labels = DaySessionsLabelsTrimester) +
  labs(title = "(c) Daily discretisation")+
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 1
    ),
    axis.title = element_blank(),
    plot.title = element_text(size = 12, hjust = .5),
    panel.grid.major.x = element_blank(),
    legend.position = "bottom"
  )
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})

# Discrete ----
## tryCatch({
gg_Discrete = ggarrange(
  gg_Month + theme(axis.text.x = element_blank()),
  gg_Week + theme(axis.text.x = element_blank()),
  gg_Day + theme(axis.text.x = element_text(angle = 0, hjust = .5)),
  nrow = 3,
  heights = c(.8, .8, 1),
  common.legend = T,
  legend = "bottom"
)
print(gg_Discrete)


# Export
ggsave(
  "./output/lynx_detection_history_discrete.pdf",
  plot = gg_Discrete,
  width = 25,
  height = 18,
  unit = "cm"
)
jpeg(
  filename = "./output/lynx_detection_history_discrete.jpeg",
  width = 25,
  height = 18,
  unit = "cm",
  res = 100
)
print(gg_Discrete)
dev.off()
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


# print(MonthLongDF, n=40, width=Inf)
# print(MonthCountMatrix)
# quit()


## -----------------------------------------------------------------------------
ModelComparisonDF <- data.frame()


## ----class.source = 'fold-show'-----------------------------------------------
o = 10
o1 = 5
print(MonthCountMatrix)
umf <- unmarkedFrameOccu(y = (as.matrix(MonthCountMatrix) > 1) * 1)
summary(umf)
print(umf)


## ----class.source = 'fold-show'-----------------------------------------------
(psi_init <- mean(rowSums(getY(umf), na.rm = TRUE) > 0))


## ----class.source = 'fold-show'-----------------------------------------------
(p_init <- mean(
  getY(umf)[rowSums(getY(umf), na.rm = TRUE) > 0,] > 0, 
  na.rm = T
))


## ----class.source = 'fold-show'-----------------------------------------------
## tryCatch({
beforetime = Sys.time()
MonthOccuMod <- occu(formula =  ~ 1 ~ 1,
     data = umf,
     method = "Nelder-Mead",
     starts = c(qlogis(psi_init), qlogis(p_init))
)
aftertime = Sys.time()


## ----class.source = 'fold-show'-----------------------------------------------
# Estimates
backTransform(MonthOccuMod, type = "state")
backTransform(MonthOccuMod, type = "det")

# 95% Confidence intervals
plogis(confint(MonthOccuMod, type = 'state', method = 'normal'))
plogis(confint(MonthOccuMod, type = 'det', method = 'normal'))

# 50% Confidence intervals
plogis(confint(MonthOccuMod, type = 'state', method = 'normal', level = 0.50))
plogis(confint(MonthOccuMod, type = 'det', method = 'normal', level = 0.50))


## -----------------------------------------------------------------------------
ModelComparisonDF <- bind_rows(ModelComparisonDF, data.frame(
  "Model" = "BP",
  "Discretisation" = "Month",
  # psi
  "psi_TransformedPointEstimate" = unname(coef(MonthOccuMod)["psi(Int)"]),
  "psi_TransformedSE" = unname(SE(MonthOccuMod)["psi(Int)"]),
  "psi_PointEstimate" = backTransform(MonthOccuMod, type = "state")@estimate,
  "psi_CI95lower" = plogis(confint(MonthOccuMod, type = 'state', method = 'normal'))[1],
  "psi_CI95upper" = plogis(confint(MonthOccuMod, type = 'state', method = 'normal'))[2],
  "psi_CI50lower" = plogis(confint(MonthOccuMod, type = 'state', method = 'normal', level = 0.50))[1],
  "psi_CI50upper" = plogis(confint(MonthOccuMod, type = 'state', method = 'normal', level = 0.50))[2],
  # p
  "p_TransformedPointEstimate" = unname(coef(MonthOccuMod)["p(Int)"]),
  "p_TransformedSE" = unname(SE(MonthOccuMod)["p(Int)"]),
  "p_PointEstimate" = backTransform(MonthOccuMod, type = "det")@estimate,
  "p_CI95lower" = plogis(confint(MonthOccuMod, type = 'det', method = 'normal'))[1],
  "p_CI95upper" = plogis(confint(MonthOccuMod, type = 'det', method = 'normal'))[2],
  "p_CI50lower" = plogis(confint(MonthOccuMod, type = 'det', method = 'normal', level = 0.50))[1],
  "p_CI50upper" = plogis(confint(MonthOccuMod, type = 'det', method = 'normal', level = 0.50))[2]
))
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})

## ----class.source = 'fold-show'-----------------------------------------------
## tryCatch({

## ----class.source = 'fold-show'-----------------------------------------------
o = 10
o1 = 5
print(MonthCountMatrix)
y <- (as.matrix(MonthCountMatrix) > 1) * 1
umf <- unmarkedFrameOccuFP(y=y, type=c(0,dim(y)[2],0))
summary(umf)
print(umf)


## ----class.source = 'fold-show'-----------------------------------------------
(psi_init <- mean(rowSums(getY(umf), na.rm = TRUE) > 0))


## ----class.source = 'fold-show'-----------------------------------------------
(p_init <- mean(
  getY(umf)[rowSums(getY(umf), na.rm = TRUE) > 0,] > 0, 
  na.rm = T
))

beforetime = Sys.time()
MonthOccuModFP <- occuFP(detformula =  ~ 1, stateformula=~1, FPformula=~1,
     data = umf,
     method = "Nelder-Mead",
     starts = c(qlogis(psi_init), qlogis(p_init), 0.5)
)
aftertime = Sys.time()


## ----class.source = 'fold-show'-----------------------------------------------
# Estimates
backTransform(MonthOccuModFP, type = "state")
backTransform(MonthOccuModFP, type = "det")

# 95% Confidence intervals
plogis(confint(MonthOccuModFP, type = 'state', method = 'normal'))
plogis(confint(MonthOccuModFP, type = 'det', method = 'normal'))

# 50% Confidence intervals
plogis(confint(MonthOccuModFP, type = 'state', method = 'normal', level = 0.50))
plogis(confint(MonthOccuModFP, type = 'det', method = 'normal', level = 0.50))


## -----------------------------------------------------------------------------
ModelComparisonDF <- bind_rows(ModelComparisonDF, data.frame(
  "Model" = "BP_FP",
  "Discretisation" = "Month",
  # psi
  "psi_TransformedPointEstimate" = unname(coef(MonthOccuModFP)["psi(Int)"]),
  "psi_TransformedSE" = unname(SE(MonthOccuModFP)["psi(Int)"]),
  "psi_PointEstimate" = backTransform(MonthOccuModFP, type = "state")@estimate,
  "psi_CI95lower" = plogis(confint(MonthOccuModFP, type = 'state', method = 'normal'))[1],
  "psi_CI95upper" = plogis(confint(MonthOccuModFP, type = 'state', method = 'normal'))[2],
  "psi_CI50lower" = plogis(confint(MonthOccuModFP, type = 'state', method = 'normal', level = 0.50))[1],
  "psi_CI50upper" = plogis(confint(MonthOccuModFP, type = 'state', method = 'normal', level = 0.50))[2],
  # p
  "p_TransformedPointEstimate" = unname(coef(MonthOccuModFP)["p(Int)"]),
  "p_TransformedSE" = unname(SE(MonthOccuModFP)["p(Int)"]),
  "p_PointEstimate" = backTransform(MonthOccuModFP, type = "det")@estimate,
  "p_CI95lower" = plogis(confint(MonthOccuModFP, type = 'det', method = 'normal'))[1],
  "p_CI95upper" = plogis(confint(MonthOccuModFP, type = 'det', method = 'normal'))[2],
  "p_CI50lower" = plogis(confint(MonthOccuModFP, type = 'det', method = 'normal', level = 0.50))[1],
  "p_CI50upper" = plogis(confint(MonthOccuModFP, type = 'det', method = 'normal', level = 0.50))[2]
))
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


## ----class.source = 'fold-show'-----------------------------------------------
umf <- unmarkedFrameOccu(y = (as.matrix(WeekCountMatrix) > 1) * 1)
summary(umf)


## ----class.source = 'fold-show'-----------------------------------------------
(psi_init <- mean(rowSums(getY(umf), na.rm = TRUE) > 0))


## ----class.source = 'fold-show'-----------------------------------------------
(p_init <- mean(
  getY(umf)[rowSums(getY(umf), na.rm = TRUE) > 0,] > 0, 
  na.rm = T
))


## ----class.source = 'fold-show'-----------------------------------------------
beforetime = Sys.time()
## tryCatch({
WeekOccuMod <- occu(formula =  ~ 1 ~ 1,
     data = umf,
     method = "Nelder-Mead",
     starts = c(qlogis(psi_init), qlogis(p_init))
)
aftertime = Sys.time()
print(WeekOccuMod)


## ----class.source = 'fold-show'-----------------------------------------------
# Estimates
backTransform(WeekOccuMod, type = "state")
backTransform(WeekOccuMod, type = "det")

# 95% Confidence intervals
plogis(confint(WeekOccuMod, type = 'state', method = 'normal'))
plogis(confint(WeekOccuMod, type = 'det', method = 'normal'))

# 50% Confidence intervals
plogis(confint(WeekOccuMod, type = 'state', method = 'normal', level = 0.50))
plogis(confint(WeekOccuMod, type = 'det', method = 'normal', level = 0.50))


## -----------------------------------------------------------------------------
ModelComparisonDF <- bind_rows(ModelComparisonDF, data.frame(
  "Model" = "BP",
  "Discretisation" = "Week",
  # psi
  "psi_TransformedPointEstimate" = unname(coef(WeekOccuMod)["psi(Int)"]),
  "psi_TransformedSE" = unname(SE(WeekOccuMod)["psi(Int)"]),
  "psi_PointEstimate" = backTransform(WeekOccuMod, type = "state")@estimate,
  "psi_CI95lower" = plogis(confint(WeekOccuMod, type = 'state', method = 'normal'))[1],
  "psi_CI95upper" = plogis(confint(WeekOccuMod, type = 'state', method = 'normal'))[2],
  "psi_CI50lower" = plogis(confint(WeekOccuMod, type = 'state', method = 'normal', level = 0.50))[1],
  "psi_CI50upper" = plogis(confint(WeekOccuMod, type = 'state', method = 'normal', level = 0.50))[2],
  # p
  "p_TransformedPointEstimate" = unname(coef(WeekOccuMod)["p(Int)"]),
  "p_TransformedSE" = unname(SE(WeekOccuMod)["p(Int)"]),
  "p_PointEstimate" = backTransform(WeekOccuMod, type = "det")@estimate,
  "p_CI95lower" = plogis(confint(WeekOccuMod, type = 'det', method = 'normal'))[1],
  "p_CI95upper" = plogis(confint(WeekOccuMod, type = 'det', method = 'normal'))[2],
  "p_CI50lower" = plogis(confint(WeekOccuMod, type = 'det', method = 'normal', level = 0.50))[1],
  "p_CI50upper" = plogis(confint(WeekOccuMod, type = 'det', method = 'normal', level = 0.50))[2]
))
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


## ----class.source = 'fold-show'-----------------------------------------------
## tryCatch({

## ----class.source = 'fold-show'-----------------------------------------------
y <- (as.matrix(WeekCountMatrix) > 1) * 1
umf <- unmarkedFrameOccuFP(y=y, type=c(0,dim(y)[2],0))
summary(umf)


## ----class.source = 'fold-show'-----------------------------------------------
(psi_init <- mean(rowSums(getY(umf), na.rm = TRUE) > 0))


## ----class.source = 'fold-show'-----------------------------------------------
(p_init <- mean(
  getY(umf)[rowSums(getY(umf), na.rm = TRUE) > 0,] > 0, 
  na.rm = T
))

beforetime = Sys.time()
WeekOccuModFP <- occuFP(detformula =  ~ 1, stateformula=~1, FPformula=~1,
     data = umf,
     method = "Nelder-Mead",
     starts = c(qlogis(psi_init), qlogis(p_init), 0.5)
)
aftertime = Sys.time()
print(WeekOccuModFP)


## ----class.source = 'fold-show'-----------------------------------------------
# Estimates
backTransform(WeekOccuModFP, type = "state")
backTransform(WeekOccuModFP, type = "det")

# 95% Confidence intervals
plogis(confint(WeekOccuModFP, type = 'state', method = 'normal'))
plogis(confint(WeekOccuModFP, type = 'det', method = 'normal'))

# 50% Confidence intervals
plogis(confint(WeekOccuModFP, type = 'state', method = 'normal', level = 0.50))
plogis(confint(WeekOccuModFP, type = 'det', method = 'normal', level = 0.50))


## -----------------------------------------------------------------------------
ModelComparisonDF <- bind_rows(ModelComparisonDF, data.frame(
  "Model" = "BP_FP",
  "Discretisation" = "Week",
  # psi
  "psi_TransformedPointEstimate" = unname(coef(WeekOccuModFP)["psi(Int)"]),
  "psi_TransformedSE" = unname(SE(WeekOccuModFP)["psi(Int)"]),
  "psi_PointEstimate" = backTransform(WeekOccuModFP, type = "state")@estimate,
  "psi_CI95lower" = plogis(confint(WeekOccuModFP, type = 'state', method = 'normal'))[1],
  "psi_CI95upper" = plogis(confint(WeekOccuModFP, type = 'state', method = 'normal'))[2],
  "psi_CI50lower" = plogis(confint(WeekOccuModFP, type = 'state', method = 'normal', level = 0.50))[1],
  "psi_CI50upper" = plogis(confint(WeekOccuModFP, type = 'state', method = 'normal', level = 0.50))[2],
  # p
  "p_TransformedPointEstimate" = unname(coef(WeekOccuModFP)["p(Int)"]),
  "p_TransformedSE" = unname(SE(WeekOccuModFP)["p(Int)"]),
  "p_PointEstimate" = backTransform(WeekOccuModFP, type = "det")@estimate,
  "p_CI95lower" = plogis(confint(WeekOccuModFP, type = 'det', method = 'normal'))[1],
  "p_CI95upper" = plogis(confint(WeekOccuModFP, type = 'det', method = 'normal'))[2],
  "p_CI50lower" = plogis(confint(WeekOccuModFP, type = 'det', method = 'normal', level = 0.50))[1],
  "p_CI50upper" = plogis(confint(WeekOccuModFP, type = 'det', method = 'normal', level = 0.50))[2]
))
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


## ----class.source = 'fold-show'-----------------------------------------------
umf <- unmarkedFrameOccu(y = (as.matrix(DayCountMatrix) > 1) * 1)
summary(umf)


## ----class.source = 'fold-show'-----------------------------------------------
(psi_init <- mean(rowSums(getY(umf), na.rm = TRUE) > 0))


## ----class.source = 'fold-show'-----------------------------------------------
(p_init <- mean(
  getY(umf)[rowSums(getY(umf), na.rm = TRUE) > 0,] > 0, 
  na.rm = T
))


## ----class.source = 'fold-show'-----------------------------------------------
## tryCatch({
beforetime = Sys.time()
DayOccuMod <- occu(formula =  ~ 1 ~ 1,
     data = umf,
     method = "Nelder-Mead",
     starts = c(qlogis(psi_init), qlogis(p_init))
)
aftertime = Sys.time()
print(DayOccuMod)


## ----class.source = 'fold-show'-----------------------------------------------
# Estimates
backTransform(DayOccuMod, type = "state")
backTransform(DayOccuMod, type = "det")

# 95% Confidence intervals
plogis(confint(DayOccuMod, type = 'state', method = 'normal'))
plogis(confint(DayOccuMod, type = 'det', method = 'normal'))

# 50% Confidence intervals
plogis(confint(DayOccuMod, type = 'state', method = 'normal', level = 0.50))
plogis(confint(DayOccuMod, type = 'det', method = 'normal', level = 0.50))


## -----------------------------------------------------------------------------
ModelComparisonDF <- bind_rows(ModelComparisonDF, data.frame(
  "Model" = "BP",
  "Discretisation" = "Day",
  # psi
  "psi_TransformedPointEstimate" = unname(coef(DayOccuMod)["psi(Int)"]),
  "psi_TransformedSE" = unname(SE(DayOccuMod)["psi(Int)"]),
  "psi_PointEstimate" = backTransform(DayOccuMod, type = "state")@estimate,
  "psi_CI95lower" = plogis(confint(DayOccuMod, type = 'state', method = 'normal'))[1],
  "psi_CI95upper" = plogis(confint(DayOccuMod, type = 'state', method = 'normal'))[2],
  "psi_CI50lower" = plogis(confint(DayOccuMod, type = 'state', method = 'normal', level = 0.50))[1],
  "psi_CI50upper" = plogis(confint(DayOccuMod, type = 'state', method = 'normal', level = 0.50))[2],
  # p
  "p_TransformedPointEstimate" = unname(coef(DayOccuMod)["p(Int)"]),
  "p_TransformedSE" = unname(SE(DayOccuMod)["p(Int)"]),
  "p_PointEstimate" = backTransform(DayOccuMod, type = "det")@estimate,
  "p_CI95lower" = plogis(confint(DayOccuMod, type = 'det', method = 'normal'))[1],
  "p_CI95upper" = plogis(confint(DayOccuMod, type = 'det', method = 'normal'))[2],
  "p_CI50lower" = plogis(confint(DayOccuMod, type = 'det', method = 'normal', level = 0.50))[1],
  "p_CI50upper" = plogis(confint(DayOccuMod, type = 'det', method = 'normal', level = 0.50))[2]
))
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


## ----class.source = 'fold-show'-----------------------------------------------
y <- (as.matrix(DayCountMatrix) > 1) * 1
umf <- unmarkedFrameOccuFP(y=y, type=c(0,dim(y)[2],0))
summary(umf)


## ----class.source = 'fold-show'-----------------------------------------------
(psi_init <- mean(rowSums(getY(umf), na.rm = TRUE) > 0))


## ----class.source = 'fold-show'-----------------------------------------------
(p_init <- mean(
  getY(umf)[rowSums(getY(umf), na.rm = TRUE) > 0,] > 0, 
  na.rm = T
))

beforetime = Sys.time()
## tryCatch({
DayOccuModFP <- occuFP(detformula =  ~ 1, stateformula=~1, FPformula=~1,
     data = umf,
     method = "Nelder-Mead",
     starts = c(qlogis(psi_init), qlogis(p_init), 0.5)
)
aftertime = Sys.time()
print(DayOccuModFP)


## ----class.source = 'fold-show'-----------------------------------------------
# Estimates
backTransform(DayOccuModFP, type = "state")
backTransform(DayOccuModFP, type = "det")

# 95% Confidence intervals
plogis(confint(DayOccuModFP, type = 'state', method = 'normal'))
plogis(confint(DayOccuModFP, type = 'det', method = 'normal'))

# 50% Confidence intervals
plogis(confint(DayOccuModFP, type = 'state', method = 'normal', level = 0.50))
plogis(confint(DayOccuModFP, type = 'det', method = 'normal', level = 0.50))


## -----------------------------------------------------------------------------
ModelComparisonDF <- bind_rows(ModelComparisonDF, data.frame(
  "Model" = "BP_FP",
  "Discretisation" = "Day",
  # psi
  "psi_TransformedPointEstimate" = unname(coef(DayOccuModFP)["psi(Int)"]),
  "psi_TransformedSE" = unname(SE(DayOccuModFP)["psi(Int)"]),
  "psi_PointEstimate" = backTransform(DayOccuModFP, type = "state")@estimate,
  "psi_CI95lower" = plogis(confint(DayOccuModFP, type = 'state', method = 'normal'))[1],
  "psi_CI95upper" = plogis(confint(DayOccuModFP, type = 'state', method = 'normal'))[2],
  "psi_CI50lower" = plogis(confint(DayOccuModFP, type = 'state', method = 'normal', level = 0.50))[1],
  "psi_CI50upper" = plogis(confint(DayOccuModFP, type = 'state', method = 'normal', level = 0.50))[2],
  # p
  "p_TransformedPointEstimate" = unname(coef(DayOccuModFP)["p(Int)"]),
  "p_TransformedSE" = unname(SE(DayOccuModFP)["p(Int)"]),
  "p_PointEstimate" = backTransform(DayOccuModFP, type = "det")@estimate,
  "p_CI95lower" = plogis(confint(DayOccuModFP, type = 'det', method = 'normal'))[1],
  "p_CI95upper" = plogis(confint(DayOccuModFP, type = 'det', method = 'normal'))[2],
  "p_CI50lower" = plogis(confint(DayOccuModFP, type = 'det', method = 'normal', level = 0.50))[1],
  "p_CI50upper" = plogis(confint(DayOccuModFP, type = 'det', method = 'normal', level = 0.50))[2]
))
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


## ----class.source = 'fold-show'-----------------------------------------------
umf = unmarkedFrameOccuCOP(
  y = as.matrix(MonthCountMatrix),
  L = matrix(
    data = rep(days_in_month(MonthSessions[-length(MonthSessions)]), each =
                 nrow(MonthCountMatrix)),
    nrow = nrow(MonthCountMatrix),
    ncol = ncol(MonthCountMatrix),
    dimnames = dimnames(MonthCountMatrix)
  )
)
print(umf)
quit()
summary(umf)


## ----class.source = 'fold-show'-----------------------------------------------
(psi_init <- mean(rowSums(getY(umf), na.rm = TRUE) > 0))


## ----class.source = 'fold-show'-----------------------------------------------
(lambda_init <- mean((getY(umf) / getL(umf))[rowSums(getY(umf), na.rm = TRUE) > 0, ],
                     na.rm = T))


print(psi_init)
print(lambda_init)

## ----class.source = 'fold-show'-----------------------------------------------
## tryCatch({
beforetime = Sys.time()
MonthOccuCOPMod <- occuCOP(
  data = umf,
  psiformula =  ~ 1,
  lambdaformula =  ~ 1,
  method = "Nelder-Mead",
  psistarts = qlogis(psi_init),
  lambdastarts = log(lambda_init)
)
aftertime = Sys.time()
print(MonthOccuCOPMod)


## ----class.source = 'fold-show'-----------------------------------------------
# Estimates
backTransform(MonthOccuCOPMod, type = "psi")
backTransform(MonthOccuCOPMod, type = "lambda")

# 95% Confidence intervals
plogis(confint(MonthOccuCOPMod, type = 'psi', method = 'normal'))
plogis(confint(MonthOccuCOPMod, type = 'lambda', method = 'normal'))

# 50% Confidence intervals
plogis(confint(MonthOccuCOPMod, type = 'psi', method = 'normal', level = 0.50))
plogis(confint(MonthOccuCOPMod, type = 'lambda', method = 'normal', level = 0.50))


## -----------------------------------------------------------------------------
ModelComparisonDF <- bind_rows(ModelComparisonDF, data.frame(
  "Model" = "COP",
  "Discretisation" = "Month",
  # psi
  "psi_TransformedPointEstimate" = unname(coef(MonthOccuCOPMod)["psi(Int)"]),
  "psi_TransformedSE" = unname(SE(MonthOccuCOPMod)["psi(Int)"]),
  "psi_PointEstimate" = backTransform(MonthOccuCOPMod, type = "psi")@estimate,
  "psi_CI95lower" = plogis(confint(MonthOccuCOPMod, type = 'psi', method = 'normal'))[1],
  "psi_CI95upper" = plogis(confint(MonthOccuCOPMod, type = 'psi', method = 'normal'))[2],
  "psi_CI50lower" = plogis(confint(MonthOccuCOPMod, type = 'psi', method = 'normal', level = 0.50))[1],
  "psi_CI50upper" = plogis(confint(MonthOccuCOPMod, type = 'psi', method = 'normal', level = 0.50))[2],
  # lambda
  "lambda_TransformedPointEstimate" = unname(coef(MonthOccuCOPMod)["lambda(Int)"]),
  "lambda_TransformedSE" = unname(SE(MonthOccuCOPMod)["lambda(Int)"]),
  "lambda_PointEstimate" = backTransform(MonthOccuCOPMod, type = "lambda")@estimate,
  "lambda_CI95lower" = exp(confint(MonthOccuCOPMod, type = 'lambda', method = 'normal'))[1],
  "lambda_CI95upper" = exp(confint(MonthOccuCOPMod, type = 'lambda', method = 'normal'))[2],
  "lambda_CI50lower" = plogis(confint(MonthOccuCOPMod, type = 'lambda', method = 'normal', level = 0.50))[1],
  "lambda_CI50upper" = plogis(confint(MonthOccuCOPMod, type = 'lambda', method = 'normal', level = 0.50))[2]
))

## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


## ----class.source = 'fold-show'-----------------------------------------------
umf = unmarkedFrameOccuCOP(
  y = as.matrix(WeekCountMatrix),
  L = matrix(
    7,
    nrow = nrow(WeekCountMatrix),
    ncol = ncol(WeekCountMatrix),
    dimnames = dimnames(WeekCountMatrix)
  )
)
summary(umf)


## ----class.source = 'fold-show'-----------------------------------------------
(psi_init <- mean(rowSums(getY(umf), na.rm = TRUE) > 0))


## ----class.source = 'fold-show'-----------------------------------------------
(lambda_init <- mean((getY(umf) / getL(umf))[rowSums(getY(umf), na.rm = TRUE) > 0, ],
                     na.rm = T))


## ----class.source = 'fold-show'-----------------------------------------------
## tryCatch({
beforetime = Sys.time()
WeekOccuCOPMod <- occuCOP(
  data = umf,
  psiformula =  ~ 1,
  lambdaformula =  ~ 1,
  method = "Nelder-Mead",
  psistarts = qlogis(psi_init),
  lambdastarts = log(lambda_init)
)
aftertime = Sys.time()
print(WeekOccuCOPMod)


## ----class.source = 'fold-show'-----------------------------------------------
# Estimates
backTransform(WeekOccuCOPMod, type = "psi")
backTransform(WeekOccuCOPMod, type = "lambda")

# 95% Confidence intervals
plogis(confint(WeekOccuCOPMod, type = 'psi', method = 'normal'))
plogis(confint(WeekOccuCOPMod, type = 'lambda', method = 'normal'))

# 50% Confidence intervals
plogis(confint(WeekOccuCOPMod, type = 'psi', method = 'normal', level = 0.50))
plogis(confint(WeekOccuCOPMod, type = 'lambda', method = 'normal', level = 0.50))


## -----------------------------------------------------------------------------
ModelComparisonDF <- bind_rows(ModelComparisonDF, data.frame(
  "Model" = "COP",
  "Discretisation" = "Week",
  # psi
  "psi_TransformedPointEstimate" = unname(coef(WeekOccuCOPMod)["psi(Int)"]),
  "psi_TransformedSE" = unname(SE(WeekOccuCOPMod)["psi(Int)"]),
  "psi_PointEstimate" = backTransform(WeekOccuCOPMod, type = "psi")@estimate,
  "psi_CI95lower" = plogis(confint(WeekOccuCOPMod, type = 'psi', method = 'normal'))[1],
  "psi_CI95upper" = plogis(confint(WeekOccuCOPMod, type = 'psi', method = 'normal'))[2],
  "psi_CI50lower" = plogis(confint(WeekOccuCOPMod, type = 'psi', method = 'normal', level = 0.50))[1],
  "psi_CI50upper" = plogis(confint(WeekOccuCOPMod, type = 'psi', method = 'normal', level = 0.50))[2],
  # lambda
  "lambda_TransformedPointEstimate" = unname(coef(WeekOccuCOPMod)["lambda(Int)"]),
  "lambda_TransformedSE" = unname(SE(WeekOccuCOPMod)["lambda(Int)"]),
  "lambda_PointEstimate" = backTransform(WeekOccuCOPMod, type = "lambda")@estimate,
  "lambda_CI95lower" = exp(confint(WeekOccuCOPMod, type = 'lambda', method = 'normal'))[1],
  "lambda_CI95upper" = exp(confint(WeekOccuCOPMod, type = 'lambda', method = 'normal'))[2],
  "lambda_CI50lower" = plogis(confint(WeekOccuCOPMod, type = 'lambda', method = 'normal', level = 0.50))[1],
  "lambda_CI50upper" = plogis(confint(WeekOccuCOPMod, type = 'lambda', method = 'normal', level = 0.50))[2]
))
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


## ----class.source = 'fold-show'-----------------------------------------------
## tryCatch({
umf = unmarkedFrameOccuCOP(
  y = as.matrix(DayCountMatrix),
  L = matrix(
    1,
    nrow = nrow(DayCountMatrix),
    ncol = ncol(DayCountMatrix),
    dimnames = dimnames(DayCountMatrix)
  )
)
summary(umf)


## ----class.source = 'fold-show'-----------------------------------------------
(psi_init <- mean(rowSums(getY(umf), na.rm = TRUE) > 0))


## ----class.source = 'fold-show'-----------------------------------------------
(lambda_init <- mean((getY(umf) / getL(umf))[rowSums(getY(umf), na.rm = TRUE) > 0, ],
                     na.rm = T))


## ----class.source = 'fold-show'-----------------------------------------------
beforetime = Sys.time()
DayOccuCOPMod <- occuCOP(
  data = umf,
  psiformula =  ~ 1,
  lambdaformula =  ~ 1,
  method = "Nelder-Mead",
  psistarts = qlogis(psi_init),
  lambdastarts = log(lambda_init)
)
aftertime = Sys.time()
print(DayOccuCOPMod)


## ----class.source = 'fold-show'-----------------------------------------------
# Estimates
backTransform(DayOccuCOPMod, type = "psi")
backTransform(DayOccuCOPMod, type = "lambda")

# 95% Confidence intervals
plogis(confint(DayOccuCOPMod, type = 'psi', method = 'normal'))
plogis(confint(DayOccuCOPMod, type = 'lambda', method = 'normal'))

# 50% Confidence intervals
plogis(confint(DayOccuCOPMod, type = 'psi', method = 'normal', level = 0.50))
plogis(confint(DayOccuCOPMod, type = 'lambda', method = 'normal', level = 0.50))


## -----------------------------------------------------------------------------
ModelComparisonDF <- bind_rows(ModelComparisonDF, data.frame(
  "Model" = "COP",
  "Discretisation" = "Day",
  # psi
  "psi_TransformedPointEstimate" = unname(coef(DayOccuCOPMod)["psi(Int)"]),
  "psi_TransformedSE" = unname(SE(DayOccuCOPMod)["psi(Int)"]),
  "psi_PointEstimate" = backTransform(DayOccuCOPMod, type = "psi")@estimate,
  "psi_CI95lower" = plogis(confint(DayOccuCOPMod, type = 'psi', method = 'normal'))[1],
  "psi_CI95upper" = plogis(confint(DayOccuCOPMod, type = 'psi', method = 'normal'))[2],
  "psi_CI50lower" = plogis(confint(DayOccuCOPMod, type = 'psi', method = 'normal', level = 0.50))[1],
  "psi_CI50upper" = plogis(confint(DayOccuCOPMod, type = 'psi', method = 'normal', level = 0.50))[2],
  # lambda
  "lambda_TransformedPointEstimate" = unname(coef(DayOccuCOPMod)["lambda(Int)"]),
  "lambda_TransformedSE" = unname(SE(DayOccuCOPMod)["lambda(Int)"]),
  "lambda_PointEstimate" = backTransform(DayOccuCOPMod, type = "lambda")@estimate,
  "lambda_CI95lower" = exp(confint(DayOccuCOPMod, type = 'lambda', method = 'normal'))[1],
  "lambda_CI95upper" = exp(confint(DayOccuCOPMod, type = 'lambda', method = 'normal'))[2],
  "lambda_CI50lower" = plogis(confint(DayOccuCOPMod, type = 'lambda', method = 'normal', level = 0.50))[1],
  "lambda_CI50upper" = plogis(confint(DayOccuCOPMod, type = 'lambda', method = 'normal', level = 0.50))[2]
))
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


## ----class.source = 'fold-show'-----------------------------------------------
# Number of detections per site
NbDetecPerSite <- sapply(detection_times, function(x){length(x[[1]])})

# Duration of the monitoring period per site
MonitoringDurationPerSite <- unlist(list_T_ij)


## ----class.source = 'fold-show'-----------------------------------------------
(psi_init <- mean(NbDetecPerSite > 0))


## ----class.source = 'fold-show'-----------------------------------------------
(lambda_init <- mean((NbDetecPerSite / MonitoringDurationPerSite)[NbDetecPerSite > 0]))


## ----fit_PP_all, class.source = 'fold-show'-----------------------------------
## tryCatch({
beforetime = Sys.time()
fitted_PP <- optim(
  # Initial parameters
  par = c(
    'psi' = logit(psi_init),
    'lambda' = log(lambda_init)
  ),
  # Function to optimize
  fn = get_PP_neg_loglikelihood,
  # Optim parameters
  method = "Nelder-Mead",
  # Other parameters of get_likelihood
  detection_times = detection_times,
  NbSites = NbSites,
  list_T_ij = list_T_ij,
  list_R_i = list_R_i,
  hessian = T
)
aftertime = Sys.time()
print(fitted_PP)


## -----------------------------------------------------------------------------
fisher_info <- solve(fitted_PP$hessian)
if (any(diag(fisher_info) < 0)) {
  se <- sqrt(diag(fisher_info) + 0i)
} else{
  se <- sqrt(diag(fisher_info))
}

# 95% CI transformed (logit for psi, log for lambda)
upper95CI_fitted_PP <- fitted_PP$par + qnorm(.975) * se
lower95CI_fitted_PP <- fitted_PP$par - qnorm(.975) * se

# 50% CI transformed (logit for psi, log for lambda)
upper50CI_fitted_PP <- fitted_PP$par + qnorm(.75) * se
lower50CI_fitted_PP <- fitted_PP$par - qnorm(.75) * se


## -----------------------------------------------------------------------------
resDF <- data.frame(
  "Model" = "PP",
  "Discretisation" = "No discretisation",
  # psi
  "psi_TransformedPointEstimate" = unname(fitted_PP$par['psi']),
  "psi_TransformedSE" = unname(se["psi"]),
  "psi_PointEstimate" = plogis(unname(fitted_PP$par['psi'])),
  "psi_CI95lower" = plogis(unname(lower95CI_fitted_PP['psi'])),
  "psi_CI95upper" = plogis(unname(upper95CI_fitted_PP['psi'])),
  "psi_CI50lower" = plogis(unname(lower50CI_fitted_PP['psi'])),
  "psi_CI50upper" = plogis(unname(upper50CI_fitted_PP['psi'])),
  # lambda
  "lambda_TransformedPointEstimate" = unname(fitted_PP$par['lambda']),
  "lambda_TransformedSE" = unname(se["lambda"]),
  "lambda_PointEstimate" = exp(unname(fitted_PP$par['lambda'])),
  "lambda_CI95lower" = exp(unname(lower95CI_fitted_PP['lambda'])),
  "lambda_CI95upper" = exp(unname(upper95CI_fitted_PP['lambda'])),
  "lambda_CI50lower" = plogis(unname(lower50CI_fitted_PP['lambda'])),
  "lambda_CI50upper" = plogis(unname(upper50CI_fitted_PP['lambda']))
)
cat(paste(colnames(resDF), resDF[1, ], sep = ": ", collapse = "\n"))
ModelComparisonDF <- bind_rows(
  ModelComparisonDF,
  resDF
)
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


## ----class.source = 'fold-show'-----------------------------------------------
# Number of detections per site
NbDetecPerSite <- sapply(detection_times, function(x){length(x[[1]])})

# Duration of the monitoring period per site
MonitoringDurationPerSite <- unlist(list_T_ij)


## ----class.source = 'fold-show'-----------------------------------------------
(psi_init <- mean(NbDetecPerSite > 0))


## ----class.source = 'fold-show'-----------------------------------------------
mu_12_init = 1
mu_21_init = 1


## ----class.source = 'fold-show'-----------------------------------------------
(lambda2_init <- mean((NbDetecPerSite / MonitoringDurationPerSite)[NbDetecPerSite > 0])/2)


## ----fit_IPP_all--------------------------------------------------------------
## tryCatch({
beforetime = Sys.time()
fitted_IPP <- optim(
  # Initial parameters
  par = c(
    'psi' = logit(psi_init),
    'lambda_2' = log(lambda2_init),
    'mu_12' = log(mu_12_init),
    'mu_21' = log(mu_21_init)
  ),
  # Function to optimize
  fn = get_IPP_neg_loglikelihood,
  # Optim parameters
  method = "Nelder-Mead",
  # Other parameters of get_likelihood
  detection_times = detection_times,
  NbSites = NbSites,
  list_T_ij = list_T_ij,
  list_R_i = list_R_i,
  hessian = T
)
aftertime = Sys.time()
print(fitted_IPP)


## -----------------------------------------------------------------------------
fisher_info <- solve(fitted_IPP$hessian)
if (any(diag(fisher_info) < 0)) {
  se <- sqrt(diag(fisher_info) + 0i)
} else{
  se <- sqrt(diag(fisher_info))
}

# 95% CI transformed (logit for psi, log for lambda)
upper95CI_fitted_IPP <- fitted_IPP$par + qnorm(.975) * se
lower95CI_fitted_IPP <- fitted_IPP$par - qnorm(.975) * se

# 50% CI transformed (logit for psi, log for lambda)
upper50CI_fitted_IPP <- fitted_IPP$par + qnorm(.75) * se
lower50CI_fitted_IPP <- fitted_IPP$par - qnorm(.75) * se


## -----------------------------------------------------------------------------
resDF <- data.frame(
  "Model" = "IPP",
  "Discretisation" = "No discretisation",
  # psi
  "psi_TransformedPointEstimate" = unname(fitted_IPP$par['psi']),
  "psi_TransformedSE" = unname(se["psi"]),
  "psi_PointEstimate" = plogis(unname(fitted_IPP$par['psi'])),
  "psi_CI95lower" = plogis(unname(lower95CI_fitted_IPP['psi'])),
  "psi_CI95upper" = plogis(unname(upper95CI_fitted_IPP['psi'])),
  "psi_CI50lower" = plogis(unname(lower50CI_fitted_IPP['psi'])),
  "psi_CI50upper" = plogis(unname(upper50CI_fitted_IPP['psi'])),
  # lambda2
  "lambda2_TransformedPointEstimate" = unname(fitted_IPP$par['lambda_2']),
  "lambda2_TransformedSE" = unname(se["lambda_2"]),
  "lambda2_PointEstimate" = exp(unname(fitted_IPP$par['lambda_2'])),
  "lambda2_CI95lower" = exp(unname(lower95CI_fitted_IPP['lambda_2'])),
  "lambda2_CI95upper" = exp(unname(upper95CI_fitted_IPP['lambda_2'])),
  "lambda2_CI50lower" = exp(unname(lower50CI_fitted_IPP['lambda_2'])),
  "lambda2_CI50upper" = exp(unname(upper50CI_fitted_IPP['lambda_2'])),
  # mu12
  "mu12_TransformedPointEstimate" = unname(fitted_IPP$par['mu_12']),
  "mu12_TransformedSE" = unname(se["mu_12"]),
  "mu12_PointEstimate" = exp(unname(fitted_IPP$par['mu_12'])),
  "mu12_CI95lower" = exp(unname(lower95CI_fitted_IPP['mu_12'])),
  "mu12_CI95upper" = exp(unname(upper95CI_fitted_IPP['mu_12'])),
  "mu12_CI50lower" = exp(unname(lower50CI_fitted_IPP['mu_12'])),
  "mu12_CI50upper" = exp(unname(upper50CI_fitted_IPP['mu_12'])),
  # mu21
  "mu21_TransformedPointEstimate" = unname(fitted_IPP$par['mu_21']),
  "mu21_TransformedSE" = unname(se["mu_21"]),
  "mu21_PointEstimate" = exp(unname(fitted_IPP$par['mu_21'])),
  "mu21_CI95lower" = exp(unname(lower95CI_fitted_IPP['mu_21'])),
  "mu21_CI95upper" = exp(unname(upper95CI_fitted_IPP['mu_21'])),
  "mu21_CI50lower" = exp(unname(lower50CI_fitted_IPP['mu_21'])),
  "mu21_CI50upper" = exp(unname(upper50CI_fitted_IPP['mu_21']))
)
cat(paste(colnames(resDF), resDF[1, ], sep = ": ", collapse = "\n"))
ModelComparisonDF <- bind_rows(
  ModelComparisonDF,
  resDF
)
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


## -----------------------------------------------------------------------------
lambda1_init = 0.01


## ----fit_2MMPP_all------------------------------------------------------------
## tryCatch({
beforetime = Sys.time()
fitted_2MMPP <- optim(
  # Initial parameters
  par = c(
    'psi' = logit(psi_init),
    'lambda_1' = log(lambda1_init),
    'lambda_2' = log(lambda2_init),
    'mu_12' = log(mu_12_init),
    'mu_21' = log(mu_21_init)
  ),
  # Function to optimize
  fn = get_2MMPP_neg_loglikelihood,
  # Optim parameters
  method = "Nelder-Mead",
  # Other parameters of get_likelihood
  detection_times = detection_times,
  NbSites = NbSites,
  list_T_ij = list_T_ij,
  list_R_i = list_R_i,
  hessian = T
)
aftertime = Sys.time()
print(fitted_2MMPP)


## -----------------------------------------------------------------------------
fisher_info <- solve(fitted_2MMPP$hessian)
if (any(diag(fisher_info) < 0)) {
  se <- sqrt(diag(fisher_info) + 0i)
} else{
  se <- sqrt(diag(fisher_info))
}

# 95% CI transformed (logit for psi, log for lambda)
upper95CI_fitted_2MMPP <- fitted_2MMPP$par + qnorm(.975) * se
lower95CI_fitted_2MMPP <- fitted_2MMPP$par - qnorm(.975) * se

# 50% CI transformed (logit for psi, log for lambda)
upper50CI_fitted_2MMPP <- fitted_2MMPP$par + qnorm(.75) * se
lower50CI_fitted_2MMPP <- fitted_2MMPP$par - qnorm(.75) * se


## -----------------------------------------------------------------------------
resDF <- data.frame(
  "Model" = "2-MMPP",
  "Discretisation" = "No discretisation",
  # psi
  "psi_TransformedPointEstimate" = unname(fitted_2MMPP$par['psi']),
  "psi_TransformedSE" = unname(se["psi"]),
  "psi_PointEstimate" = plogis(unname(fitted_2MMPP$par['psi'])),
  "psi_CI95lower" = plogis(unname(lower95CI_fitted_2MMPP['psi'])),
  "psi_CI95upper" = plogis(unname(upper95CI_fitted_2MMPP['psi'])),
  "psi_CI50lower" = plogis(unname(lower50CI_fitted_2MMPP['psi'])),
  "psi_CI50upper" = plogis(unname(upper50CI_fitted_2MMPP['psi'])),
  # lambda1
  "lambda1_TransformedPointEstimate" = unname(fitted_2MMPP$par['lambda_1']),
  "lambda1_TransformedSE" = unname(se["lambda_1"]),
  "lambda1_PointEstimate" = exp(unname(fitted_2MMPP$par['lambda_1'])),
  "lambda1_CI95lower" = exp(unname(lower95CI_fitted_2MMPP['lambda_1'])),
  "lambda1_CI95upper" = exp(unname(upper95CI_fitted_2MMPP['lambda_1'])),
  "lambda1_CI50lower" = exp(unname(lower50CI_fitted_2MMPP['lambda_1'])),
  "lambda1_CI50upper" = exp(unname(upper50CI_fitted_2MMPP['lambda_1'])),
  # lambda2
  "lambda2_TransformedPointEstimate" = unname(fitted_2MMPP$par['lambda_2']),
  "lambda2_TransformedSE" = unname(se["lambda_2"]),
  "lambda2_PointEstimate" = exp(unname(fitted_2MMPP$par['lambda_2'])),
  "lambda2_CI95lower" = exp(unname(lower95CI_fitted_2MMPP['lambda_2'])),
  "lambda2_CI95upper" = exp(unname(upper95CI_fitted_2MMPP['lambda_2'])),
  "lambda2_CI50lower" = exp(unname(lower50CI_fitted_2MMPP['lambda_2'])),
  "lambda2_CI50upper" = exp(unname(upper50CI_fitted_2MMPP['lambda_2'])),
  # mu12
  "mu12_TransformedPointEstimate" = unname(fitted_2MMPP$par['mu_12']),
  "mu12_TransformedSE" = unname(se["mu_12"]),
  "mu12_PointEstimate" = exp(unname(fitted_2MMPP$par['mu_12'])),
  "mu12_CI95lower" = exp(unname(lower95CI_fitted_2MMPP['mu_12'])),
  "mu12_CI95upper" = exp(unname(upper95CI_fitted_2MMPP['mu_12'])),
  "mu12_CI50lower" = exp(unname(lower50CI_fitted_2MMPP['mu_12'])),
  "mu12_CI50upper" = exp(unname(upper50CI_fitted_2MMPP['mu_12'])),
  # mu21
  "mu21_TransformedPointEstimate" = unname(fitted_2MMPP$par['mu_21']),
  "mu21_TransformedSE" = unname(se["mu_21"]),
  "mu21_PointEstimate" = exp(unname(fitted_2MMPP$par['mu_21'])),
  "mu21_CI95lower" = exp(unname(lower95CI_fitted_2MMPP['mu_21'])),
  "mu21_CI95upper" = exp(unname(upper95CI_fitted_2MMPP['mu_21'])),
  "mu21_CI50lower" = exp(unname(lower50CI_fitted_2MMPP['mu_21'])),
  "mu21_CI50upper" = exp(unname(upper50CI_fitted_2MMPP['mu_21']))
)
cat(paste(colnames(resDF), resDF[1, ], sep = ": ", collapse = "\n"))
ModelComparisonDF <- bind_rows(
  ModelComparisonDF,
  resDF
)
## }, error=function(e){cat("ERROR :",conditionMessage(e), "\n")})


# # TODO: remove?
# quit()

## ----ComparisonResultDF-------------------------------------------------------
PublicationDF <- ModelComparisonDF %>% 
  pivot_longer(cols = starts_with(c("psi", "p", "lambda", "mu")), 
               names_to = c("Parameter", "Interval"), 
               names_sep = "_")%>% 
  pivot_wider(names_from = "Interval",values_from = "value") %>% 
  filter(rowSums(is.na(.)) < 2)


PublicationDF %>%
  mutate(TransformedPointEstimate = round(TransformedPointEstimate, 3),
         TransformedSE = round(TransformedSE, 3),
         PointEstimate = round(PointEstimate, 3),
         CI95lower = round(CI95lower, 3),
         CI95upper = round(CI95upper, 3),
         CI50lower = round(CI50lower, 3),
         CI50upper = round(CI50upper, 3),
         Model = factor(Model, levels = c("BP", "COP", "PP", "IPP", "2-MMPP")),
         Discretisation = factor(Discretisation, levels = c("No discretisation","Day","Week","Month")),
         Parameter = as.factor(Parameter)) %>%
  myDT(caption = "Occupancy models comparison on the lynx dataset")


## ----ComparisonResultPsiPlot--------------------------------------------------
plotPsiDiscrete = ModelComparisonDF %>%
  filter(Model %in% c("BP", "COP")) %>%
  mutate(
    Model = factor(
      as.character(Model),
      levels = c("BP", "COP"),
      labels = c("BP", "COP"),
      ordered = T
    ),
    Discretisation = factor(as.character(Discretisation), levels = c("Month", "Week", "Day"), ordered = T)
  ) %>%
  ggplot(aes(group = interaction(Model, Discretisation))) +
  geom_segment(
    aes(
      y = psi_CI95lower,
      yend = psi_CI95upper,
      x = Discretisation,
      xend = Discretisation,
      colour = "95% confidence interval"
    ),
    linewidth = 2
  ) +
  geom_segment(
    aes(
      y = psi_CI50lower,
      yend = psi_CI50upper,
      x = Discretisation,
      xend = Discretisation,
      colour = "50% confidence interval"
    ),
    linewidth = 2
  ) +
  geom_point(
    aes(x = Discretisation, y = psi_PointEstimate, fill = "Point estimate"),
    size = 3,
    shape = 23,
    colour = "transparent"
  ) +
  facet_grid(. ~ Model,
             switch = "y") +
  scale_fill_manual(values = c("Point estimate" = "grey5"),
                    name = "") +
  scale_colour_manual(
    values = c(
      "95% confidence interval" = "grey80",
      "50% confidence interval" = "grey65"
    ),
    name = ""
  ) +
  ylim(0, 1) +
  theme_minimal(base_size = 15) +
  labs(y = "Estimated occupancy probability") +
  scale_x_discrete(position = "top") +
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
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
    panel.border = element_rect(colour = "gray80", fill = "transparent"),
    panel.grid.major.x = element_blank()
  )

plotPsiContinuous = ModelComparisonDF %>%
  filter(Model %in% c("PP", "IPP", "2-MMPP")) %>%
  mutate(Model = factor(
    as.character(Model),
    levels = c("PP", "IPP", "2-MMPP"),
    labels = c("PP", "IPP", "2-MMPP"),
    ordered = T
  )) %>%
  ggplot(aes(group = interaction(Model, Discretisation))) +
  geom_segment(
    aes(
      y = psi_CI95lower,
      yend = psi_CI95upper,
      x = Discretisation,
      xend = Discretisation,
      colour = "95% confidence interval"
    ),
    linewidth = 2
  ) +
  geom_segment(
    aes(
      y = psi_CI50lower,
      yend = psi_CI50upper,
      x = Discretisation,
      xend = Discretisation,
      colour = "50% confidence interval"
    ),
    linewidth = 2
  ) +
  geom_point(
    aes(x = Discretisation, y = psi_PointEstimate, fill = "Point estimate"),
    size = 3,
    shape = 23,
    colour = "transparent"
  ) +
  facet_grid(. ~ Model,
             switch = "y") +
  scale_fill_manual(values = c("Point estimate" = "grey5"),
                    name = "") +
  scale_colour_manual(
    values = c(
      "95% confidence interval" = "grey80",
      "50% confidence interval" = "grey65"
    ),
    name = ""
  ) +
  ylim(0, 1) +
  theme_minimal(base_size = 15) +
  labs(y = "Estimated occupancy probability") +
  scale_x_discrete(position = "top") +
  theme(
    strip.background.y = element_rect(fill = "gray93", colour = "white"),
    strip.background.x = element_rect(fill = "gray80", colour = "white"),
    strip.text.x = element_text(colour = "black", face = "bold"),
    axis.title.x = element_blank(),
    axis.text.x = element_text(hjust=.5),
    legend.position = "bottom",
    legend.title = element_text(hjust = 0.5),
    strip.placement = "outside",
    plot.title = element_blank(),
    plot.margin = unit(c(0, 0, 0, 0), "cm"),
    panel.border = element_rect(colour = "gray80", fill = "transparent"),
    panel.grid.major.x = element_blank(),
    # Remove y axis informations
    strip.text.y = element_blank(),
    axis.text.y = element_blank(),
    axis.title.y = element_blank()
  )

# below doesn't work because `plotPsiContinuous` is empty, so replace with just `plotPsiDiscrete`
# plotPsi = ggpubr::ggarrange(
#   plotPsiDiscrete,
#   plotPsiContinuous,
#   common.legend = TRUE,
#   legend = "bottom",
#   ncol = 2,
#   widths = c(1, 0.7)
# ) +
#   theme(plot.background = element_rect(fill = "white"))
plotPsi = plotPsiDiscrete

print(plotPsi)

# Export
ggsave(
  "./output/lynx_psi.pdf",
  plot = plotPsi,
  width = 25,
  height = 10,
  unit = "cm"
)
jpeg(
  filename = "./output/lynx_psi.jpeg",
  width = 25,
  height = 10,
  unit = "cm",
  res = 100
)
print(plotPsi)
dev.off()


## -----------------------------------------------------------------------------
ModelComparisonDFlong = ModelComparisonDF %>%
  pivot_longer(cols = starts_with(c("psi", "p", "lambda", "mu")), 
               names_to = c("Parameter", "Interval"), 
               names_sep = "_") %>% 
  pivot_wider(names_from = "Interval",values_from = "value")


detecParams = unique(ModelComparisonDFlong$Parameter)
detecParams = detecParams[detecParams != "psi"]
for (param in detecParams) {
  ggDetec = ModelComparisonDFlong %>%
    filter(Parameter == param) %>%
    filter(!is.na(PointEstimate)) %>%
    ggplot(aes(group = interaction(Model, Discretisation))) +
    geom_segment(
      aes(
        y = CI95lower,
        yend = CI95upper,
        x = Discretisation,
        xend = Discretisation,
        colour = "95% confidence interval"
      ),
      linewidth = 2
    ) +
    geom_segment(
      aes(
        y = CI50lower,
        yend = CI50upper,
        x = Discretisation,
        xend = Discretisation,
        colour = "50% confidence interval"
      ),
      linewidth = 2
    ) +
    geom_point(
      aes(x = Discretisation, y = PointEstimate, fill = "Point estimate"),
      size = 3,
      shape = 23,
      colour = "transparent"
    ) +
    facet_grid(. ~ Model,
               switch = "y",
               scales = "free") +
    scale_fill_manual(values = c("Point estimate" = "grey5"),
                      name = "") +
    scale_colour_manual(
      values = c(
        "95% confidence interval" = "grey80",
        "50% confidence interval" = "grey65"
      ),
      name = ""
    ) +
    theme_minimal(base_size = 15) +
    labs(y = paste("Estimated", param),
         title =  paste("Detection parameter:", param)) +
    scale_x_discrete(position = "top") +
    theme(
      strip.background.y = element_rect(fill = "gray93", colour = "white"),
      strip.text.y = element_text(colour = "black", face = "bold"),
      strip.background.x = element_rect(fill = "gray80", colour = "white"),
      strip.text.x = element_text(colour = "black", face = "bold"),
      axis.title.x = element_blank(),
      axis.text.x = element_text(hjust = .5),
      axis.text.y = element_text(hjust = .5),
      legend.position = "bottom",
      legend.title = element_text(hjust = 0.5),
      strip.placement = "outside",
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.margin = unit(c(0, 0, 0, 0), "cm"),
      panel.border = element_rect(colour = "gray80", fill = "transparent"),
      panel.grid.major.x = element_blank()
    )
  print(ggDetec)
}

