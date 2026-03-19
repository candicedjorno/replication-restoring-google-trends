################################################################################
#' ssa_denoising.R                                                   
#' Title: Denoising Google Trends data using Singular Spectrum Analysis  
#' Using fixed parameters for window length L and number of components k
#' Variables are denoised based on RMSE, where RMSE is computed as the error 
#' between the raw time series and its smoothed predictions
#' If RMSE >= 0.05 for a variable, perform denoising
#' Using parallel computing
################################################################################

library(Rssa)
library(xts)
library(parallel)

# Finding the RMSE for each keyword based on data from train set
# `original` is the initial data to denoise
# `window_size` is the number of weeks in the rolling window 
# output = set of errors
ssa_optimal_denoising_rmse <- function(original, window_size, L, k) {
  rmses <- rep(NA, ncol(original))
  # Get the number of available cores
  n_cores <- detectCores() - 1  # Leave one core free for other tasks
  # n_cores <- 1
  
  # Use mclapply for parallel processing
  results <- mclapply(1:ncol(original), function(i) {
    keyword <- original[, i]
    cat(names(keyword), "\n")
    
    # Check if the keyword is all zeros
    if (all(keyword == 0)) {
      rmse <- 0.01
    } else {
      # applying moving averages on data in rolling window
      preds <- rep(NA, length(keyword))
      
      for (n in window_size:(length(keyword) - 1)) {
        # training set = points in the window up to current point
        train <- keyword[(n - window_size + 1):n]
        train <- as.numeric(train)
      #   if (all(train == 0)) { # SSA can’t decompose/forecast because no signal
      #     next_point <- 0
      #   } else {
      #     # compute SSA instead of smoothing spline
      #     ssa_obj <- ssa(train, L = L)
      #     next_point <- rforecast(ssa_obj, groups = list(1:k), len = 1)
      #   }
      #   preds[n + 1] <- next_point
        
        # compute SSA instead of smoothing spline
        # Try SSA and fallback if it errors
        next_point <- tryCatch({
          ssa_obj <- ssa(train, L = L)
          rforecast(ssa_obj, groups = list(1:k), len = 1)
        }, error = function(e) {
          # Fallback: use last observation
          train[length(train)]
        }, warning = function(w) {
          # Some warnings are recoverable, but if needed:
          train[length(train)]
        })
        preds[n + 1] <- next_point
      }
    
    # choosing the smoothing parameter with the minimum RMSE
    rmse <- sqrt(mean((preds - keyword)^2, na.rm = TRUE)) / max(keyword)
    }
    
    list(rmse = rmse)
  }, mc.cores = n_cores)  # Set the number of cores for parallel processing
  
  # Combine results back into data frames
  for (i in 1:ncol(original)) {
    rmses[i] <- results[[i]]$rmse
  }
  
  return(list(rmses = rmses)) 
}

# Using the RMSEs to denoise the data from test set
# `original` is the initial data to denoise
# `window_size` is the number of weeks in the rolling window 
# `rmses` is the vector of errors
# output = smoothed data
ssa_denoising_rmse <- function(original, window_size, rmses, L, k) {
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
    
    if (rmses[i] >= 0.05) { # proceed to denoise
      
      for (n in window_size:(length(keyword) - 1)) {
        train <- keyword[(n - window_size + 1):n]
        train <- as.numeric(train)
        # compute SSA instead of smoothing spline
        # ssa_obj <- ssa(train, L = L)
        # next_point <- reconstruct(ssa_obj, groups = list(1:k))$F1
        # Try SSA and fallback if it errors
        next_point <- tryCatch({
          ssa_obj <- ssa(train, L = L)
          rforecast(ssa_obj, groups = list(1:k), len = 1)
        }, error = function(e) {
          # Fallback: use last observation
          train[length(train)]
        }, warning = function(w) {
          # Some warnings are recoverable, but if needed:
          train[length(train)]
        })
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
