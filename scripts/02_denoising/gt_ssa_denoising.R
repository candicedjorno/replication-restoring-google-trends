################################################################################
#' gt_ssa_denoising.R                                                   
#' Title: Denoising Google Trends data using Singular Spectrum Analysis 
#' Using fixed parameters for window length L and number of components k
#' Finding RMSE on the train set
#' Denoising data from the train and test sets based on error
################################################################################

library(xts)
# accessing the smoothing functions
source("scripts/02_denoising/ssa_denoising.R")

# denoising Google trends raw data
file_name <- paste0("data/03_preprocessed/cluster_gt.csv")
cat(file_name, "\n")
gt_all <- read.csv(file_name, as.is = TRUE, check.names = FALSE)
cat("dim of gt_all:", dim(gt_all), "\n")
gt_all$date <-  as.Date(gt_all$date)

# setting the cutoff for the train and test sets
window_size <- 104 # number of weeks in the rolling window 
# parameters for SSA
L <- 52 # window length
k <- 3 # number of components
cutoff_train <- as.Date("2022-10-01")
cutoff_test <- cutoff_train - (window_size * 7) # to account for weeks of rolling window
cat("cutoff_train:")
print(cutoff_train)
cat("cutoff_test:")
print(cutoff_test)

gt_train <- gt_all[gt_all$date < cutoff_train, ]
cat("dim of train:", dim(gt_train), "\n")
gt_test <- gt_all[gt_all$date >= cutoff_test, ]
cat("dim of test:", dim(gt_test), "\n")

# transforming into time series
gt_train <- xts::xts(gt_train[, -1], order.by = as.Date(gt_train$date))
gt_test <- xts::xts(gt_test[, -1], order.by = as.Date(gt_test$date))

# finding rmse for keywords to decide whether to denoise
cat("START \n")
original <- gt_train # data from train set
result <- ssa_optimal_denoising_rmse(original, window_size, L, k)
rmses <- result$rmses
cat("RMSEs based on train set: \n")
print(summary(rmses))

# denoising the data from train set based on best smoothing parameters and error
# if error >= 0.05, keyword is denoised
cat("Denoising train set \n")
original <- gt_train # data from test set
smoothing1 <- ssa_denoising_rmse(original, window_size, rmses, L, k)
smoothed_train <- smoothing1

# denoising the data from test set based on best smoothing parameters and error
# if error >= 0.05, keyword is denoised
cat("Denoising test set \n")
original <- gt_test # data from test set
smoothing2 <- ssa_denoising_rmse(original, window_size, rmses, L, k)
smoothed_test <- smoothing2

cat("END \n")
cat("Smoothing finished. \n")

# doncatenating smoothed_train and smoothed_test
smoothed_combined <- rbind(smoothed_train, smoothed_test)

# renaming column names with "raw_" or "smooth_"
smoothed_cols <- colnames(smoothed_combined)[colSums(is.na(smoothed_combined)) > 0]
colnames(smoothed_combined) <- sapply(colnames(smoothed_combined), function(col) {
  if (col == "date") {
    col  # Keep "date" column unchanged
  } else if (col %in% smoothed_cols) {
    paste0("smooth_", col)  # Add "smooth_" for columns with NAs
  } else {
    paste0("raw_", col)  # Add "raw_" for columns without NAs
  }
})

smoothed_combined <- na.omit(smoothed_combined)

# creating csv file of denoised data
# output file contains both raw and denoised data, depending on the keyword
# "_raw" or "_smooth" indicates whether the keyword has been denoised or not
combined_out <- paste0("results/denoising_results/smooth_ssa_gt.csv")
write.csv(smoothed_combined, file = combined_out, row.names = FALSE)





