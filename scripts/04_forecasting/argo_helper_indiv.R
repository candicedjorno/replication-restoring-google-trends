################################################################################
# Title: Helper functions for ARGO-based predictions
################################################################################
source("scripts/04_forecasting/argo_functions.R")
# library(argo) # using modified argo functions, not the package
library(forecast)
library(zoo)
library(parallel)
library(xts)

# read each location's data
directory <- "data/01_raw/individual_merged_trends"

# list files, sorted alphabetically
files <- list.files(directory, pattern = "\\.csv$", full.names = FALSE)
files <- sort(files)
cat(length(files), "csv files\n")

max_date <- "2024-05-04"

# retrieving hospitalizations and Google Trends for each state
data_prep_indiv <- function(state) {
  print(state)
  # matching location to hospitalization data
  state_code <- sub(".*-", "", state)
  if (state == "US-DC") {
    state_name <- "District of Columbia"
  } else if (state == "PR") {
    state_name <- "Puerto Rico"
  } else {
    # mapping the state name to is abbreviation
    # state.name/state.abb are functions related to the 50 states
    state_name <- state.name[match(state_code, state.abb)]
  }
  print(state_name)
  hosp_col <- paste0("hosp_", state_name)
  hosp_state <- hosp_ts[, hosp_col] # hosp time series for one state
  print(hosp_col)
  
  filename <- files[grep(paste0("^", state, "_"), files)][1]
  print(filename)
  gt_state <- read.csv(file.path(directory, filename),
                   as.is = TRUE, check.names = FALSE)
  
  # realigning the data
  gt_state_shifted <- gt_state
  cols_to_shift <- setdiff(names(gt_state), "date") # adjust "date" if needed
  gt_state_shifted[cols_to_shift] <- lapply(gt_state[cols_to_shift], function(x) c(NA, head(x, -1)))
  gt_state <- gt_state_shifted
  
  # remove columns with all zeros
  all_zero_columns <- names(gt_state)[colSums(gt_state == 0, na.rm = TRUE) == nrow(gt_state)]
  if (length(all_zero_columns) > 0) {
    gt_state <- gt_state[ , !(names(gt_state) %in% all_zero_columns), drop = FALSE]
  }
  
  gt_state <- gt_state[gt_state$date >= min(main$date), ] # aligning minimum date
  gt_state <- gt_state[gt_state$date <= max_date, ] # aligning maximum date
  gt_state <- xts::xts(gt_state[, -1], order.by = as.Date(gt_state$date))
  
  return(list(gt_state = gt_state, hosp_state = hosp_state, state_name = state_name))
}

data_prep_topics <- function(state) {
  print(state)
  # matching location to hospitalization data
  state_code <- sub(".*-", "", state)
  if (state == "US-DC") {
    state_name <- "District of Columbia"
  } else if (state == "PR") {
    state_name <- "Puerto Rico"
  } else {
    # mapping the state name to is abbreviation
    # state.name/state.abb are functions related to the 50 states
    state_name <- state.name[match(state_code, state.abb)]
  }
  print(state_name)
  hosp_col <- paste0("hosp_", state_name)
  hosp_state <- hosp_ts[, hosp_col] # hosp time series for one state
  print(hosp_col)
  
  filename <- files[grep(paste0("^", state, "_"), files)][1]
  print(filename)
  gt_state <- read.csv(file.path(directory, filename),
                       as.is = TRUE, check.names = FALSE)
  
  # realigning the data
  gt_state_shifted <- gt_state
  cols_to_shift <- setdiff(names(gt_state), "date") # adjust "date" if needed
  gt_state_shifted[cols_to_shift] <- lapply(gt_state[cols_to_shift], function(x) c(NA, head(x, -1)))
  gt_state <- gt_state_shifted
  
  # remove columns with all zeros
  all_zero_columns <- names(gt_state)[colSums(gt_state == 0, na.rm = TRUE) == nrow(gt_state)]
  if (length(all_zero_columns) > 0) {
    gt_state <- gt_state[ , !(names(gt_state) %in% all_zero_columns), drop = FALSE]
  }
  
  gt_state <- gt_state[gt_state$date >= min(main$date), ] # aligning minimum date
  gt_state <- gt_state[gt_state$date <= max_date, ] # aligning maximum date
  gt_state <- xts::xts(gt_state[, -1], order.by = as.Date(gt_state$date))
  
  topic_cols <- grep("/m/|\\(TOPIC\\)", names(gt_state), value = TRUE)
  gt_state <- gt_state[ , topic_cols, drop = FALSE]
  
  return(list(gt_state = gt_state, hosp_state = hosp_state, state_name = state_name))
}

# ARGO
argo_forecast <- function(gt_state, hosp_state, y_gap, forecast, percentile_value, top_n) {
  exogen <- gt_state
  if (!is.null(exogen)) {
    yx_merged <- merge(hosp_state, exogen, join = "right")
  } else {
    yx_merged <- hosp_state
  }
  y <- yx_merged[, 1]
  our_argo <-
    argo(y, exogen = exogen, alpha = 0, corr_cutoff = 1,
         percentile_value = percentile_value, top_n = top_n,
         N_lag=1:52, use_all_previous = FALSE, N_training = 104,
         schedule = list(y_gap = y_gap, forecast = forecast))
  
  # merging results
  pred_xts_blend <- merge(our_argo$pred, all=FALSE)
  names(pred_xts_blend) <- c("ARGO")
  
  # keeping only the data where there are no NA's
  first_non_na_index <- which(!is.na(pred_xts_blend$ARGO))[1]
  start_date <- index(pred_xts_blend[first_non_na_index])
  first_index <- which(time(pred_xts_blend) == start_date)
  end_date <- index(pred_xts_blend[nrow(pred_xts_blend)])
  pred_xts_blend_sub <- pred_xts_blend[first_index:nrow(pred_xts_blend),]
  
  true_hosp <- window(hosp_state, start = start_date, end = end_date)
  
  # Computing RMSE on last flu season: 2023-10-21 to 2024-04-13
  cutoff_start <- which(time(pred_xts_blend) == '2023-10-21')[1]
  cutoff_end <- which(time(pred_xts_blend) == '2024-04-13')[1]
  preds_sub <- pred_xts_blend[cutoff_start:cutoff_end,]
  cutoff_start <- which(time(true_hosp) == '2023-10-21')[1]
  cutoff_end <- which(time(true_hosp) == '2024-04-13')[1]
  true_sub <- true_hosp[cutoff_start:cutoff_end,]
  
  rmse_results <- numeric(ncol(pred_xts_blend))
  for(i in 1:ncol(pred_xts_blend)) {
    squared_diff <- (preds_sub[,i] - true_sub)^2
    rmse <- sqrt(mean(squared_diff, na.rm = TRUE))
    rmse_results[i] <- rmse
  }
  # names(rmse_results) <- c("ARGO", "AR3", "AR1", "AR52")
  names(rmse_results) <- c("ARGO")
  mae_results <- numeric(ncol(pred_xts_blend))
  for(i in 1:ncol(pred_xts_blend)) {
    squared_diff <- abs(preds_sub[,i] - true_sub)
    mae <- mean(squared_diff, na.rm = TRUE)
    mae_results[i] <- mae
  }
  # names(mae_results) <- c("ARGO", "AR3", "AR1", "AR52")
  names(rmse_results) <- c("ARGO")
  return(list(rmse_results = rmse_results, mae_results = mae_results, 
              argo1_preds = pred_xts_blend_sub$ARGO,
              truth = true_hosp))
}

# ARGO1 for different forecast horizons (0 or 1) and different Google Trends data
run_argo1 <- function(horizon, data_type) {
  
  # national level
  state <- "US"
  print(state)
  hosp_col <- "hosp_US"
  hosp_state <- hosp_ts[, hosp_col] # hosp time series for one state
  print(hosp_col)
  filename <- files[grep(paste0("^", state, "_"), files)][1]
  print(filename)
  gt_state <- read.csv(file.path(directory, filename),
                       as.is = TRUE, check.names = FALSE)
  
  # realigning the data
  gt_state_shifted <- gt_state
  cols_to_shift <- setdiff(names(gt_state), "date") # adjust "date" if needed
  gt_state_shifted[cols_to_shift] <- lapply(gt_state[cols_to_shift], function(x) c(NA, head(x, -1)))
  gt_state <- gt_state_shifted
  
  # remove columns with all zeros
  all_zero_columns <- names(gt_state)[colSums(gt_state == 0, na.rm = TRUE) == nrow(gt_state)]
  if (length(all_zero_columns) > 0) {
    gt_state <- gt_state[ , !(names(gt_state) %in% all_zero_columns), drop = FALSE]
  }
  gt_state <- gt_state[gt_state$date >= min(main$date), ] # aligning minimum date
  gt_state <- gt_state[gt_state$date <= max_date, ] # aligning maximum date
  gt_state <- xts::xts(gt_state[, -1], order.by = as.Date(gt_state$date))
  percentile_value = 0
  top_n = 100
  
  if (data_type == "topics") {
    topic_cols <- grep("/m/|\\(TOPIC\\)", names(gt_state), value = TRUE)
    gt_state <- gt_state[ , topic_cols, drop = FALSE]
    percentile_value = 0
    top_n = 100
  }
  
  if (horizon == 1) {
    # gt_state_h1 <- gt_state
    time(gt_state) <- as.Date(time(gt_state)) + 7 
  } else if (horizon == 2) {
    # gt_state_h1 <- gt_state
    time(gt_state) <- as.Date(time(gt_state)) + 14
  } else if (horizon == 3) {
    # gt_state_h1 <- gt_state
    time(gt_state) <- as.Date(time(gt_state)) + 21
  }
  
  # results
  results_us <- argo_forecast(gt_state, hosp_state, 
                              percentile_value = percentile_value, top_n = top_n,
                              y_gap = 1, forecast = horizon)
  rmse_us <- results_us$rmse_results
  mae_us <- results_us$mae_results
  argo_nat <- results_us$argo1_preds
  truth_nat <- results_us$truth
  # renaming 
  colnames(argo_nat) <- "US"
  colnames(truth_nat) <- "US"
  cat("RMSE US", rmse_us, "\n")
  cat("MAE US", mae_us, "\n")
  
  # state level
  argo_state <- data.frame("date" = time(argo_nat),
                           matrix(NA, nrow = nrow(argo_nat), ncol = length(geo)))
  argo_state <- xts::xts(argo_state[,-1], order.by = as.Date(argo_state$date))
  
  truth_state <- data.frame("date" = time(truth_nat), 
                            matrix(NA, nrow = nrow(truth_nat), ncol = length(geo)))
  truth_state <- xts::xts(truth_state[,-1], order.by = as.Date(truth_state$date))
  rmses_state <- list()
  maes_state <- list()
  
  state_names <- c()
  for (i in 1:length(geo)) {
    state <- geo[[i]]
    # preparing data
    if (data_type == "topics") {
      dat <- data_prep_topics(state)
    } else {
      dat <- data_prep_indiv(state)
    }
    gt_state <- dat$gt_state
    hosp_state <- dat$hosp_state
    # retrieving state names
    state_names <- c(state_names, dat$state_name)
    
    if (horizon == 1) {
      # gt_state_h1 <- gt_state
      time(gt_state) <- as.Date(time(gt_state)) + 7 
    } else if (horizon == 2) {
      # gt_state_h1 <- gt_state
      time(gt_state) <- as.Date(time(gt_state)) + 14
    } else if (horizon == 3) {
      # gt_state_h1 <- gt_state
      time(gt_state) <- as.Date(time(gt_state)) + 21
    }
    
    # ARGO1 results
    results_state <- argo_forecast(gt_state, hosp_state, 
                                   percentile_value = percentile_value, top_n = top_n, 
                                   y_gap = 1, forecast = horizon)
    argo1 <- results_state$argo1_preds
    truth1 <- results_state$truth
    
    # storing errors
    rmse_state <- results_state$rmse_results
    mae_state <- results_state$mae_results
    
    argo_state[,i] <- argo1
    truth_state[,i] <- truth1
    rmses_state[[state]] <- rmse_state
    maes_state[[state]] <- mae_state
  }
  state_names <- unique(state_names)
  # renaming column names to be states
  colnames(argo_state) <- state_names
  colnames(truth_state) <- state_names
  
  # Combine RMSE results for all states into a single data frame
  all_rmse_results <- do.call(rbind, rmses_state)
  all_mae_results <- do.call(rbind, maes_state)

  return(list(argo_nat = argo_nat,
              truth_nat = truth_nat,
              argo_state = argo_state, 
              truth_state = truth_state, 
              rmse_us = rmse_us,
              mae_us = mae_us,
              all_rmse_results = all_rmse_results,
              all_mae_results = all_mae_results))
}

# ARGO2 at state level
run_argo2 <- function(truth_state, argo_state, argo_nat, horizon) {
  state_names <- c()
  for (i in 1:length(geo)) {
    state <- geo[[i]]
    # preparing data
    state_code <- sub(".*-", "", state)
    if (state == "US-DC") {
      state_name <- "District of Columbia"
    } else if (state == "PR") {
      state_name <- "Puerto Rico"
    } else {
      # mapping the state name to is abbreviation
      # state.name/state.abb are functions related to the 50 states
      state_name <- state.name[match(state_code, state.abb)]
    }
    # retrieving state names
    state_names <- c(state_names, state_name)
  }
  state_names <- unique(state_names)
  
  # preparing data
  truth <- truth_state
  argo1.p <- argo_state
  argo.nat.p <- argo_nat
  
  # ARGO2 results
  argo2_state <- argo2(truth, argo1.p, argo.nat.p, horizon = horizon)
  argo1_res <- argo2_state$onestep # ARGO1
  colnames(argo1_res) <- state_names 
  argo2_res <- argo2_state$twostep # ARGO2
  argo2_res <- argo2_res[complete.cases(argo2_res), ] # removing NAs
  colnames(argo2_res) <- state_names
  argo_truths <- argo2_state$truth # truth
  colnames(argo_truths) <- state_names
  naive_preds <- argo2_state$naive # locf
  colnames(naive_preds) <- state_names
  
  return(list(argo2_res = argo2_res,
              naive_preds = naive_preds))
}

# format for submissions
format_output <- function(preds_wide, model_name, horizon = 0) {
  preds_long <- reshape(
    preds_wide,
    varying = list(names(preds_wide)[-1]),    # Columns to stack (all except the 'date' column)
    v.names = "value",                # Name of the new column for the values
    timevar = "location",             # Name of the new column for the state names
    times = names(preds_wide)[-1],            # The names of the states (columns to be stacked)
    idvar = "date",                   # Column to keep fixed ('date')
    direction = "long"                # We are converting from wide to long format
  )
  rownames(preds_long) <- NULL
  preds_long$location <- gsub("\\.", " ", preds_long$location)
  
  # format of submissions
  names(preds_long)[names(preds_long) == "date"] <- "reference_date"
  preds_long$horizon <- horizon
  preds_long$target <- 'wk inc flu hosp'
  preds_long$target_end_date <- preds_long$reference_date
  
  if (horizon == 0) {
    preds_long$reference_date <- preds_long$target_end_date
  } else if (horizon == 1){
    preds_long$reference_date <- preds_long$target_end_date - 7
  } else if (horizon == 2){
    preds_long$reference_date <- preds_long$target_end_date - 14
  } else if (horizon == 3){
    preds_long$reference_date <- preds_long$target_end_date - 21
  }
  
  preds_long$output_type <- "quantile"
  preds_long$output_type_id <- 0.5
  preds_long$model <- model_name
  preds_long <- preds_long[, c("reference_date", "location", 
                               "horizon", "target", "target_end_date", 
                               "output_type", "output_type_id", 
                               "value", "model")]
  preds_long <- preds_long[order(preds_long$reference_date, 
                                 preds_long$location), ]
  preds_long <- preds_long[preds_long$reference_date >= as.Date('2022-10-01'), ]
  rownames(preds_long) <- NULL
  return(preds_long)
  
}
