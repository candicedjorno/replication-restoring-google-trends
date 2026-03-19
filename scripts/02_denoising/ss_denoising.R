################################################################################
#' ss_denoising.R                                                   
#' Title: Denoising Google Trends data using Smoothing Splines 
#' Variables are denoised based on RMSE, where RMSE is computed as the error 
#' between the raw time series and its smoothed predictions
#' If RMSE >= 0.05 for a variable, perform denoising
#' Using parallel computing
################################################################################

library(xts)
library(parallel)

# Finding best smoothing parameter for each keyword based on data from train set
# `original` is the initial data to denoise
# `window_size` is the number of weeks in the rolling window 
# `spars` is the vector of smoothing parameters to try for each keyword
# output = set of optimal parameters for each keyword and corresponding minimum errors
ss_optimal_denoising_rmse <- function(original, window_size, spars) {
  # initializing matrix to store smoothed trends
  pred_df <- data.frame("date" = time(original), 
                        matrix(NA, nrow = nrow(original), ncol = ncol(original)))
  # transforming into a time series
  pred_df <- xts::xts(pred_df[,-1], order.by = as.Date(pred_df$date))
  names(pred_df) <- colnames(original)
  # initializing matrix to store errors between original and smoothed data
  err_df <- data.frame("date" = time(original), 
                       matrix(NA, nrow = nrow(original), ncol = ncol(original)))
  # transforming into a time series
  err_df <- xts::xts(err_df[,-1], order.by = as.Date(err_df$date))
  names(err_df) <- colnames(original)
  # initializing vector of smoothing parameters, one for each keyword
  best_spars <- rep(NA, ncol(original))
  min_errors <- rep(NA, ncol(original))
  
  dates <- time(original)
  ts <- 1:length(dates)
  
  # Get the number of available cores
  n_cores <- detectCores() - 1  # Leave one core free for other tasks
# n_cores <- 1
  
  # Use mclapply for parallel processing
  results <- mclapply(1:ncol(original), function(i) {
    keyword <- original[, i]
    cat(names(keyword), "\n")
    
    # initializing matrix to store errors for different smoothing parameters
    errors <- matrix(NA, nrow = length(keyword), ncol = length(spars))
    
    # Check if the keyword is all zeros
    if (all(keyword == 0)) {
      best_spar <- 0.1
      min_error <- 0.01
    } else {
      for (s in seq_along(spars)) {
        spar <- spars[s]
        
        # fitting smoothing splines on data in rolling window
        preds <- rep(NA, length(keyword))
        
        for (n in window_size:(length(keyword) - 1)) {
          # training set = points in the window up to current point
          train <- keyword[(n - window_size + 1):n]
          train_dates <- time(train)
          train_ts <- 1:length(train_dates)
          
          # fitting smoothing spline with current smoothing parameter
          ss <- smooth.spline(x = train_ts, y = train, spar = spar)
          # predicting the next point
          next_point <- predict(ss, x = (length(train_ts) + 1))$y
          preds[n + 1] <- next_point
        }
        
        # calculating the MSE for the current smoothing parameter
        errors[, s] <- (preds - keyword)^2
      }
      
      # choosing the smoothing parameter with the minimum RMSE
      rmses <- sqrt(colMeans(errors, na.rm = TRUE)) / max(keyword)
      best_spar <- spars[which.min(rmses)]  # RMSE
      min_error <- rmses[which.min(rmses)]
    }
    
    # return(best_spar, best_error)
    list(best_spar = best_spar, min_error = min_error)
  }, mc.cores = n_cores)  # Set the number of cores for parallel processing
  
  # Combine results back into data frames
  for (i in 1:ncol(original)) {
    best_spars[i] <- results[[i]]$best_spar
    min_errors[i] <- results[[i]]$min_error
  }
  
  return(list(parameters = best_spars, errors = min_errors)) 
}

# Using the best smoothing parameters to denoise the data from test set
# `original` is the initial data to denoise
# `window_size` is the number of weeks in the rolling window 
# `best_spars` is the vector of optimal smoothing parameters for each keyword
# `min_errors` is the vector of minimal errors
# output = smoothed data and RMSE
ss_denoising_rmse <- function(original, window_size, best_spars, min_errors) {
  # initializing matrix to store smoothed trends
  pred_df <- data.frame("date" = time(original), 
                        matrix(NA, nrow = nrow(original), ncol = ncol(original)))
  # transform into a time series
  pred_df <- xts::xts(pred_df[,-1], order.by = as.Date(pred_df$date))
  names(pred_df) <- colnames(original)
  
  # Get the number of available cores
  n_cores <- detectCores() - 1  # Leave one core free for other tasks
  
  # Use mclapply for parallel processing
  results <- mclapply(1:ncol(original), function(i) {
    keyword <- original[, i]
    
    # fitting smoothing splines on data in rolling window
    preds <- rep(NA, length(keyword))
    
    if (min_errors[i] >= 0.05) { # proceed to denoise
      
      for (n in window_size:(length(keyword) - 1)) {
        train <- keyword[(n - window_size + 1):n]
        train_dates <- time(train)
        train_ts <- 1:length(train_dates)
        
        # fitting smoothing spline with specified smoothing parameter
        ss <- smooth.spline(x = train_ts, y = train, spar = best_spars[i])
        # predicting the next point
        next_point <- predict(ss, x = (length(train_ts)))$y
        preds[n + 1] <- next_point
      }
    } else { # no denoising
      for (n in 1:length(keyword)) {
        preds[n] <- unname(coredata(keyword[n]))
      }
    }
    
    # Return predictions and errors
    list(preds = preds)
  }, mc.cores = n_cores)  # Set the number of cores for parallel processing
  
  # Combine results back into data frames
  for (i in 1:ncol(original)) {
    pred_df[, i] <- results[[i]]$preds
  }
  
  smoothed <- data.frame(date = index(pred_df), pred_df)  
  rownames(smoothed) <- NULL
  names(smoothed) <- c("date", colnames(original))
  
  return(smoothed = smoothed)
}
