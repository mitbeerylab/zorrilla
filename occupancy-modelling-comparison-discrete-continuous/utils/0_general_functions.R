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
# GENERALITIES                                                              ----
# ──────────────────────────────────────────────────────────────────────────────

newpb = function(total, txt = '', clear = FALSE, show_after = 0, ...) {
  "
  newpb
  This function produces a progress bar using package progress.
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  total
    The total number of iterations
  
  txt
    (facultative)
    A string that will be printed in the progress bar
    e.g. with txt='Simulation' ; it will print 'Simulation 1/100'
  
  clear
    See ?progress_bar
  
  show_after
    See ?progress_bar
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  A progress bar
  
  USE ──────────────────────────────────────────────────────────────────────────
  To create a new progress bar:
  pb = newpb(total)
  
  To increment with one:
  pb$tick()
  "
  require(stringr)
  require(progress)
  txt = ifelse(txt != '', 
               ifelse(stringr::str_sub(txt, -1, -1) != ' ',
                      paste0(txt, ' '),
                      txt),
               txt)
  pb = progress::progress_bar$new(
    format = paste0("[:bar] :percent | ", txt, ":current/:total | spent: :elapsedfull | left::eta | :tick_rate/sec"),
    total = total, clear = clear, width = 100, show_after = show_after, ...)
  return(pb)
}



solvable = function(m) {
  "
  solvable
  Check if a matrix is 'solvable', ie if this matrix has an inverse.
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  m
    A matrix
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  TRUE if this matrix is solvable, FALSE otherwise
  "
  any(class(try(solve(m), silent = T)
  )  ==  "matrix")
}



Flattener <- function(indf, vec2col = FALSE) {
  "
  Flattener
  Function to flatten dataframes from list, e.g. from json
  Source: https://stackoverflow.com/questions/35444968/read-json-file-into-a-data-frame-without-nested-lists

  INPUTS ───────────────────────────────────────────────────────────────────────
  indf
    A nested list
  
  vec2col
    Boolean
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  A dataframe
  "
  col_fixer <- function(x, vec2col = FALSE) {
    
    if (!all(sapply(x, is.list))) {
      x <- lapply(x, function(y) {
        ifelse(length(y) == 0, NA, y)
      })
    }
    
    if (!is.list(x[[1]])) {
      if (isTRUE(vec2col)) {
        as.data.table(data.table::transpose(x))
      } else {
        vapply(x, toString, character(1L))
      }
    } else {
      
      temp <- rbindlist(x2, use.names = TRUE, fill = TRUE, idcol = TRUE)
      temp[, .time := sequence(.N), by = .id]
      value_vars <- setdiff(names(temp), c(".id", ".time"))
      dcast(temp, .id ~ .time, value.var = value_vars)[, .id := NULL]
    }
  }
  require(data.table)
  require(jsonlite)
  indf <- flatten(indf)
  listcolumns <- sapply(indf, is.list)
  newcols <- do.call(cbind, lapply(indf[listcolumns], col_fixer, vec2col))
  indf[listcolumns] <- list(NULL)
  cbind(indf, newcols)
}


calcul_rmse = function(pred, true) {
  "
  calcul_rmse
  Calculate the RMSE
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  pred
    Prediction: numeric or vector of numeric
  
  true
    True value for comparison: numeric
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  RMSE
  
  USE ──────────────────────────────────────────────────────────────────────────
  calcul_rmse(pred = c(0.3, 0.4, 0.5), true = 0.35)
  "
  return(sqrt(mean((pred - true) ^ 2)))
}



SessionLength_to_text = function(param.SessionLength) {
  "
  SessionLength_to_text
  Transforms the session length (in days) to a text
  
  INPUTS ───────────────────────────────────────────────────────────────────────
  param.SessionLength
    A numeric
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  A string
  
  USE ──────────────────────────────────────────────────────────────────────────
  SessionLength_to_text(100)
  SessionLength_to_text(30)
  SessionLength_to_text(1)
  SessionLength_to_text(1/25)
  "
  
  ifelse(
    param.SessionLength == 30,
    "Month",
    ifelse(
      param.SessionLength == 7,
      "Week",
      ifelse(
        param.SessionLength == 1,
        "Day",
        ifelse(
          round(param.SessionLength, 4) == round(1 / 24, 4),
          "Hour",
          paste(param.SessionLength, "days")
        )
      )
    )
  )
}




stars.pval <- function(p.value) {
  "
  stars.pval
  Transform a p-value to a significance level
  Adapted from package gtools: https://rdrr.io/cran/gtools/src/R/stars.pval.R

  INPUTS ───────────────────────────────────────────────────────────────────────
  p.value
    A numeric
  
  OUTPUT ───────────────────────────────────────────────────────────────────────
  A factor
  
  USE ──────────────────────────────────────────────────────────────────────────
  stars.pval(1e-45)
  stars.pval(0.04)
  stars.pval(0.6)
  "
  
  factor(unclass(
    symnum(
      p.value,
      corr = FALSE,
      na = FALSE,
      cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
      symbols = c("***", "**", "*", ".", " ")
    )
  ),
  levels = c("***", "**", "*", ".", " "),
  ordered = T)
}

