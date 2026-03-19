################################################################################
# Author: Candice Djorno                                                        
# Title: ARGO and ARGO2 predictions for horizons 0-3
################################################################################

# Using ARGO and ARGO2 to predict hospitalizations
# library(argo) # using modified argo functions, not the package
library(forecast)
library(zoo)
library(parallel)
library(xts)
source("scripts/04_forecasting/argo_functions.R")
source("scripts/04_forecasting/argo_helper.R")

# Test set = 2022-10-01 to present
# Train set for ARGO = 2017-2019 (3 years)
# Train set for ARGO2 = 2020-2021 (2 years)

# reading data
main <- read.csv("data/01_raw/hospitalizations.csv", as.is = TRUE, check.names = FALSE)
main <- main[main$date >= "2017-09-01",] # to have ARGO2 test set start on week 2022-10-01
main <- main[main$date <= "2024-04-27",]
# keeping only the hospitalization rates
ind <- grep("^hosp_", names(main))
hosp <- main[, ind]
hosp$date <- main$date
# reordering columns to have date first
hosp <- hosp[, c("date", setdiff(names(hosp), "date"))] 
cutoff <- '2022-10-01'
hosp_train <- hosp[hosp$date <= cutoff,]
# transforming data into time series
hosp_ts <- xts::xts(hosp[, -1], order.by = as.Date(hosp$date))

# Google Trends data: raw, smooth, or detrend
max_date <- "2024-05-04"
raw_gt <- read.csv("data/03_preprocessed/cluster_gt.csv", as.is = TRUE, check.names = FALSE)
raw_gt <- raw_gt[raw_gt$date <= max_date, ]
smooth_gt <- read.csv("data/03_preprocessed/smooth_gt.csv", as.is = TRUE, check.names = FALSE)
smooth_gt <- smooth_gt[smooth_gt$date <= max_date, ]
detrend_gt <- read.csv("data/03_preprocessed/detrend_gt.csv", as.is = TRUE, check.names = FALSE)
detrend_gt <- detrend_gt[detrend_gt$date <= max_date, ]

# locations: all states + PR + DC
geo <- read.csv("data/01_raw/geo_argo.txt", as.is = TRUE, check.names = FALSE, header = FALSE)
geo <- geo$V1

# num_cores <- 1

## HORIZON 0 ##

# ARGO1 for horizon 0 for different Google Trends datasets
# ARGO without exogenous variables is AR(52)
argo_raw_h0 <- run_argo1(horizon = 0, gt_all = raw_gt)

argo_smooth_h0 <- run_argo1(horizon = 0, gt_all = smooth_gt)

argo_detrend_h0 <- run_argo1(horizon = 0, gt_all = detrend_gt)

# ARGO2 for horizon 0 for different Google Trends datasets
argo2_noexog_h0 <- run_argo2(truth_state = argo_raw_h0$truth_state, 
                          argo_state = argo_raw_h0$ar52_state, 
                          argo_nat = argo_raw_h0$ar52_nat,
                          horizon = 0)

argo2_raw_h0 <- run_argo2(truth_state = argo_raw_h0$truth_state, 
                          argo_state = argo_raw_h0$argo_state, 
                          argo_nat = argo_raw_h0$argo_nat,
                          horizon = 0)

argo2_smooth_h0 <- run_argo2(truth_state = argo_smooth_h0$truth_state, 
                             argo_state = argo_smooth_h0$argo_state, 
                             argo_nat = argo_smooth_h0$argo_nat,
                             horizon = 0)

argo2_detrend_h0 <- run_argo2(truth_state = argo_detrend_h0$truth_state, 
                             argo_state = argo_detrend_h0$argo_state, 
                             argo_nat = argo_detrend_h0$argo_nat,
                             horizon = 0)

## HORIZON 1 ##

# shifting data forward by one week to use for 2-week ahead prediction
raw_gt_h1 <- raw_gt
raw_gt_h1$date <- as.Date(raw_gt$date) + 7
smooth_gt_h1 <- smooth_gt
smooth_gt_h1$date <- as.Date(smooth_gt$date) + 7
detrend_gt_h1 <- detrend_gt
detrend_gt_h1$date <- as.Date(detrend_gt$date) + 7

# ARGO1 for horizon 1 for different Google Trends datasets
argo_raw_h1 <- run_argo1(horizon = 1, gt_all = raw_gt_h1)
argo_smooth_h1 <- run_argo1(horizon = 1, gt_all = smooth_gt_h1)
argo_detrend_h1 <- run_argo1(horizon = 1, gt_all = detrend_gt_h1)

# retrieving last week of ARGO2 predictions from h0 as fitted value for last week of truth for h1
last_week_noexog <- argo2_noexog_h0$argo2_res[nrow(argo2_noexog_h0$argo2_res),] # h0 preds
truth_noexog_h1 <- c(argo_raw_h0$truth_state, last_week_noexog)

last_week_raw <- argo2_raw_h0$argo2_res[nrow(argo2_raw_h0$argo2_res),] # h0 preds
truth_raw_h1 <- c(argo_raw_h0$truth_state, last_week_raw)

last_week_smooth <- argo2_smooth_h0$argo2_res[nrow(argo2_smooth_h0$argo2_res),] # h0 preds
truth_smooth_h1 <- c(argo_smooth_h0$truth_state, last_week_smooth)

last_week_detrend <- argo2_detrend_h0$argo2_res[nrow(argo2_detrend_h0$argo2_res),] # h0 preds
truth_detrend_h1 <- c(argo_detrend_h0$truth_state, last_week_detrend)

# ARGO2 for horizon 1 for different Google Trends datasets
argo2_noexog_h1 <- run_argo2(truth_state = truth_noexog_h1, 
                             argo_state = argo_raw_h1$ar52_state, 
                             argo_nat = argo_raw_h1$ar52_nat,
                             horizon = 1)

argo2_raw_h1 <- run_argo2(truth_state = truth_raw_h1, 
                          argo_state = argo_raw_h1$argo_state, 
                          argo_nat = argo_raw_h1$argo_nat,
                          horizon = 1)

argo2_smooth_h1 <- run_argo2(truth_state = truth_smooth_h1, 
                             argo_state = argo_smooth_h1$argo_state, 
                             argo_nat = argo_smooth_h1$argo_nat,
                             horizon = 1)

argo2_detrend_h1 <- run_argo2(truth_state = truth_detrend_h1, 
                             argo_state = argo_detrend_h1$argo_state, 
                             argo_nat = argo_detrend_h1$argo_nat,
                             horizon = 1)

## HORIZON 2 ##

# shifting data forward by one week to use for 2-week ahead prediction
raw_gt_h2 <- raw_gt
raw_gt_h2$date <- as.Date(raw_gt$date) + 14
smooth_gt_h2 <- smooth_gt
smooth_gt_h2$date <- as.Date(smooth_gt$date) + 14
detrend_gt_h2 <- detrend_gt
detrend_gt_h2$date <- as.Date(detrend_gt$date) + 14

# ARGO1 for horizon 2 for different Google Trends datasets
argo_raw_h2 <- run_argo1(horizon = 2, gt_all = raw_gt_h2)
argo_smooth_h2 <- run_argo1(horizon = 2, gt_all = smooth_gt_h2)
argo_detrend_h2 <- run_argo1(horizon = 2, gt_all = detrend_gt_h2)

# retrieving last week of ARGO2 predictions from h0 as fitted value for last week of truth for h2
last_week_raw <- argo2_raw_h0$argo2_res[nrow(argo2_raw_h0$argo2_res),] # h0 preds
last_week_raw2 <- argo2_raw_h1$argo2_res[nrow(argo2_raw_h1$argo2_res),] # h1 preds
truth_raw_h2 <- c(argo_raw_h0$truth_state, last_week_raw, last_week_raw2)

last_week_smooth <- argo2_smooth_h0$argo2_res[nrow(argo2_smooth_h0$argo2_res),] # h0 preds
last_week_smooth2 <- argo2_smooth_h1$argo2_res[nrow(argo2_smooth_h1$argo2_res),] # h1 preds
truth_smooth_h2 <- c(argo_smooth_h0$truth_state, last_week_smooth, last_week_smooth2)

last_week_detrend <- argo2_detrend_h0$argo2_res[nrow(argo2_detrend_h0$argo2_res),] # h0 preds
last_week_detrend2 <- argo2_detrend_h1$argo2_res[nrow(argo2_detrend_h1$argo2_res),] # h1 preds
truth_detrend_h2 <- c(argo_detrend_h0$truth_state, last_week_detrend, last_week_detrend2)

# ARGO2 for horizon 2 for different Google Trends datasets
argo2_raw_h2 <- run_argo2(truth_state = truth_raw_h2, 
                          argo_state = argo_raw_h2$argo_state, 
                          argo_nat = argo_raw_h2$argo_nat,
                          horizon = 2)

argo2_smooth_h2 <- run_argo2(truth_state = truth_smooth_h2, 
                             argo_state = argo_smooth_h2$argo_state, 
                             argo_nat = argo_smooth_h2$argo_nat,
                             horizon = 2)

argo2_detrend_h2 <- run_argo2(truth_state = truth_detrend_h2, 
                             argo_state = argo_detrend_h2$argo_state, 
                             argo_nat = argo_detrend_h2$argo_nat,
                             horizon = 2)

## HORIZON 3 ##

# shifting data forward by one week to use for 2-week ahead prediction
raw_gt_h3 <- raw_gt
raw_gt_h3$date <- as.Date(raw_gt$date) + 21
smooth_gt_h3 <- smooth_gt
smooth_gt_h3$date <- as.Date(smooth_gt$date) + 21
detrend_gt_h3 <- detrend_gt
detrend_gt_h3$date <- as.Date(detrend_gt$date) + 21

# ARGO1 for horizon 2 for different Google Trends datasets
argo_raw_h3 <- run_argo1(horizon = 3, gt_all = raw_gt_h3)
argo_smooth_h3 <- run_argo1(horizon = 3, gt_all = smooth_gt_h3)
argo_detrend_h3 <- run_argo1(horizon = 3, gt_all = detrend_gt_h3)

# retrieving last week of ARGO2 predictions from h0 as fitted value for last week of truth for h3
last_week_raw <- argo2_raw_h0$argo2_res[nrow(argo2_raw_h0$argo2_res),] # h0 preds
last_week_raw2 <- argo2_raw_h1$argo2_res[nrow(argo2_raw_h1$argo2_res),] # h1 preds
last_week_raw3 <- argo2_raw_h2$argo2_res[nrow(argo2_raw_h2$argo2_res),] # h2 preds
truth_raw_h3 <- c(argo_raw_h0$truth_state, last_week_raw, last_week_raw2, last_week_raw3)

last_week_smooth <- argo2_smooth_h0$argo2_res[nrow(argo2_smooth_h0$argo2_res),] # h0 preds
last_week_smooth2 <- argo2_smooth_h1$argo2_res[nrow(argo2_smooth_h1$argo2_res),] # h1 preds
last_week_smooth3 <- argo2_smooth_h2$argo2_res[nrow(argo2_smooth_h2$argo2_res),] # h2 preds
truth_smooth_h3 <- c(argo_smooth_h0$truth_state, last_week_smooth, last_week_smooth2, last_week_smooth3)

last_week_detrend <- argo2_detrend_h0$argo2_res[nrow(argo2_detrend_h0$argo2_res),] # h0 preds
last_week_detrend2 <- argo2_detrend_h1$argo2_res[nrow(argo2_detrend_h1$argo2_res),] # h1 preds
last_week_detrend3 <- argo2_detrend_h2$argo2_res[nrow(argo2_detrend_h2$argo2_res),] # h2 preds
truth_detrend_h3 <- c(argo_detrend_h0$truth_state, last_week_detrend, last_week_detrend2, last_week_detrend3)

# ARGO2 for horizon 2 for different Google Trends datasets
argo2_raw_h3 <- run_argo2(truth_state = truth_raw_h3, 
                          argo_state = argo_raw_h3$argo_state, 
                          argo_nat = argo_raw_h3$argo_nat,
                          horizon = 3)

argo2_smooth_h3 <- run_argo2(truth_state = truth_smooth_h3, 
                             argo_state = argo_smooth_h3$argo_state, 
                             argo_nat = argo_smooth_h3$argo_nat,
                             horizon = 3)

argo2_detrend_h3 <- run_argo2(truth_state = truth_detrend_h3, 
                             argo_state = argo_detrend_h3$argo_state, 
                             argo_nat = argo_detrend_h3$argo_nat,
                             horizon = 3)


#### OUTPUTS ####

# TRUTH
truths <- data.frame(date = index(argo_raw_h0$truth_nat), 
                     cbind(na.omit(coredata(argo_raw_h0$truth_state)), argo_raw_h0$truth_nat), 
                     check.names = FALSE)
rownames(truths) <- NULL
truths_long <- reshape(
  truths, 
  varying = list(names(truths)[-1]),
  v.names = "value",
  timevar = "location",
  times = names(truths)[-1],
  idvar = "date",
  direction = "long"
)
rownames(truths_long) <- NULL
truths_long$location <- gsub("\\.", " ", truths_long$location)
names(truths_long)[names(truths_long) == "date"] <- "reference_date"
truths_long <- truths_long[order(truths_long$reference_date, 
                                 truths_long$location), ]
truths_long <- truths_long[truths_long$reference_date >= as.Date('2022-10-01'), ]
rownames(truths_long) <- NULL

## HORIZON 0

# ARGO1
argo_raw_h0_preds <- data.frame(date = index(argo_raw_h0$argo_state), 
                                cbind(coredata(argo_raw_h0$argo_state), argo_raw_h0$argo_nat), 
                                check.names = FALSE)
argo_raw_h0_preds_long <- format_output(argo_raw_h0_preds, "argo_raw", horizon = 0)

argo_smooth_h0_preds <- data.frame(date = index(argo_smooth_h0$argo_state), 
                                cbind(coredata(argo_smooth_h0$argo_state), argo_smooth_h0$argo_nat), 
                                check.names = FALSE)
argo_smooth_h0_preds_long <- format_output(argo_smooth_h0_preds, "argo_smooth", horizon = 0)

argo_detrend_h0_preds <- data.frame(date = index(argo_detrend_h0$argo_state), 
                                   cbind(coredata(argo_detrend_h0$argo_state), argo_detrend_h0$argo_nat), 
                                   check.names = FALSE)
argo_detrend_h0_preds_long <- format_output(argo_detrend_h0_preds, "argo_detrend", horizon = 0)

# AR52 = ARGO without exogenous variables
ar52_h0_preds <- data.frame(date = index(argo_raw_h0$ar52_state), 
                           cbind(coredata(argo_raw_h0$ar52_state), argo_raw_h0$ar52_nat), 
                           check.names = FALSE)
ar52_h0_preds_long <- format_output(ar52_h0_preds, "argo_noexog", horizon = 0)
argo_noexog_h0_preds <- ar52_h0_preds

# ARGO2
# without exogenous variables
argo2_noexog_h0_preds <- data.frame(date = index(argo2_noexog_h0$argo2_res), 
                                 coredata(argo2_noexog_h0$argo2_res), 
                                 check.names = FALSE)
argo2_noexog_h0_preds_long <- format_output(argo2_noexog_h0_preds, "argo2_noexog", horizon = 0)

argo2_raw_h0_preds <- data.frame(date = index(argo2_raw_h0$argo2_res), 
                                 coredata(argo2_raw_h0$argo2_res), 
                                 check.names = FALSE)
argo2_raw_h0_preds_long <- format_output(argo2_raw_h0_preds, "argo2_raw", horizon = 0)

argo2_smooth_h0_preds <- data.frame(date = index(argo2_smooth_h0$argo2_res), 
                                 coredata(argo2_smooth_h0$argo2_res), 
                                 check.names = FALSE)
argo2_smooth_h0_preds_long <- format_output(argo2_smooth_h0_preds, "argo2_smooth", horizon = 0)

argo2_detrend_h0_preds <- data.frame(date = index(argo2_detrend_h0$argo2_res), 
                                    coredata(argo2_detrend_h0$argo2_res), 
                                    check.names = FALSE)
argo2_detrend_h0_preds_long <- format_output(argo2_detrend_h0_preds, "argo2_detrend", horizon = 0)


## HORIZON 1

# ARGO1
argo_raw_h1_preds <- data.frame(date = index(argo_raw_h1$argo_state), 
                                cbind(coredata(argo_raw_h1$argo_state), argo_raw_h1$argo_nat), 
                                check.names = FALSE)
argo_raw_h1_preds_long <- format_output(argo_raw_h1_preds, "argo_raw", horizon = 1)

argo_smooth_h1_preds <- data.frame(date = index(argo_smooth_h1$argo_state), 
                                cbind(coredata(argo_smooth_h1$argo_state), argo_smooth_h1$argo_nat), 
                                check.names = FALSE)
argo_smooth_h1_preds_long <- format_output(argo_smooth_h1_preds, "argo_smooth", horizon = 1)

argo_detrend_h1_preds <- data.frame(date = index(argo_detrend_h1$argo_state), 
                                   cbind(coredata(argo_detrend_h1$argo_state), argo_detrend_h1$argo_nat), 
                                   check.names = FALSE)
argo_detrend_h1_preds_long <- format_output(argo_detrend_h1_preds, "argo_detrend", horizon = 1)

# AR52
ar52_h1_preds <- data.frame(date = index(argo_raw_h1$ar52_state), 
                           cbind(coredata(argo_raw_h1$ar52_state), argo_raw_h1$ar52_nat), 
                           check.names = FALSE)
ar52_h1_preds_long <- format_output(ar52_h1_preds, "argo_noexog", horizon = 1)
argo_noexog_h1_preds <- ar52_h1_preds

# ARGO2
argo2_noexog_h1_preds <- data.frame(date = index(argo2_noexog_h1$argo2_res), 
                                 coredata(argo2_noexog_h1$argo2_res), 
                                 check.names = FALSE)
argo2_noexog_h1_preds_long <- format_output(argo2_noexog_h1_preds, "argo2_noexog", horizon = 1)

argo2_raw_h1_preds <- data.frame(date = index(argo2_raw_h1$argo2_res), 
                                 coredata(argo2_raw_h1$argo2_res), 
                                 check.names = FALSE)
argo2_raw_h1_preds_long <- format_output(argo2_raw_h1_preds, "argo2_raw", horizon = 1)

argo2_smooth_h1_preds <- data.frame(date = index(argo2_smooth_h1$argo2_res), 
                                 coredata(argo2_smooth_h1$argo2_res), 
                                 check.names = FALSE)
argo2_smooth_h1_preds_long <- format_output(argo2_smooth_h1_preds, "argo2_smooth", horizon = 1)

argo2_detrend_h1_preds <- data.frame(date = index(argo2_detrend_h1$argo2_res), 
                                    coredata(argo2_detrend_h1$argo2_res), 
                                    check.names = FALSE)
argo2_detrend_h1_preds_long <- format_output(argo2_detrend_h1_preds, "argo2_detrend", horizon = 1)

## HORIZON 2

# ARGO1
argo_raw_h2_preds <- data.frame(date = index(argo_raw_h2$argo_state), 
                                cbind(coredata(argo_raw_h2$argo_state), argo_raw_h2$argo_nat), 
                                check.names = FALSE)
argo_raw_h2_preds_long <- format_output(argo_raw_h2_preds, "argo_raw", horizon = 2)

argo_smooth_h2_preds <- data.frame(date = index(argo_smooth_h2$argo_state), 
                                   cbind(coredata(argo_smooth_h2$argo_state), argo_smooth_h2$argo_nat), 
                                   check.names = FALSE)
argo_smooth_h2_preds_long <- format_output(argo_smooth_h2_preds, "argo_smooth", horizon = 2)

argo_detrend_h2_preds <- data.frame(date = index(argo_detrend_h2$argo_state), 
                                   cbind(coredata(argo_detrend_h2$argo_state), argo_detrend_h2$argo_nat), 
                                   check.names = FALSE)
argo_detrend_h2_preds_long <- format_output(argo_detrend_h2_preds, "argo_detrend", horizon = 2)

# AR52
ar52_h2_preds <- data.frame(date = index(argo_raw_h2$ar52_state), 
                           cbind(coredata(argo_raw_h2$ar52_state), argo_raw_h2$ar52_nat), 
                           check.names = FALSE)
ar52_h2_preds_long <- format_output(ar52_h2_preds, "argo_noexog", horizon = 2)
argo_noexog_h2_preds <- ar52_h2_preds

# ARGO2
argo2_noexog_h2_preds <- data.frame(date = index(argo2_noexog_h2$argo2_res), 
                                 coredata(argo2_noexog_h2$argo2_res), 
                                 check.names = FALSE)
argo2_noexog_h2_preds_long <- format_output(argo2_noexog_h2_preds, "argo2_noexog", horizon = 2)

argo2_raw_h2_preds <- data.frame(date = index(argo2_raw_h2$argo2_res), 
                                 coredata(argo2_raw_h2$argo2_res), 
                                 check.names = FALSE)
argo2_raw_h2_preds_long <- format_output(argo2_raw_h2_preds, "argo2_raw", horizon = 2)

argo2_smooth_h2_preds <- data.frame(date = index(argo2_smooth_h2$argo2_res), 
                                    coredata(argo2_smooth_h2$argo2_res), 
                                    check.names = FALSE)
argo2_smooth_h2_preds_long <- format_output(argo2_smooth_h2_preds, "argo2_smooth", horizon = 2)

argo2_detrend_h2_preds <- data.frame(date = index(argo2_detrend_h2$argo2_res), 
                                    coredata(argo2_detrend_h2$argo2_res), 
                                    check.names = FALSE)
argo2_detrend_h2_preds_long <- format_output(argo2_detrend_h2_preds, "argo2_detrend", horizon = 2)

## HORIZON 3

# ARGO1
argo_raw_h3_preds <- data.frame(date = index(argo_raw_h3$argo_state), 
                                cbind(coredata(argo_raw_h3$argo_state), argo_raw_h3$argo_nat), 
                                check.names = FALSE)
argo_raw_h3_preds_long <- format_output(argo_raw_h3_preds, "argo_raw", horizon = 3)

argo_smooth_h3_preds <- data.frame(date = index(argo_smooth_h3$argo_state), 
                                   cbind(coredata(argo_smooth_h3$argo_state), argo_smooth_h3$argo_nat), 
                                   check.names = FALSE)
argo_smooth_h3_preds_long <- format_output(argo_smooth_h3_preds, "argo_smooth", horizon = 3)

argo_detrend_h3_preds <- data.frame(date = index(argo_detrend_h3$argo_state), 
                                   cbind(coredata(argo_detrend_h3$argo_state), argo_detrend_h3$argo_nat), 
                                   check.names = FALSE)
argo_detrend_h3_preds_long <- format_output(argo_detrend_h3_preds, "argo_detrend", horizon = 3)

# AR52
ar52_h3_preds <- data.frame(date = index(argo_raw_h3$ar52_state), 
                           cbind(coredata(argo_raw_h3$ar52_state), argo_raw_h3$ar52_nat), 
                           check.names = FALSE)
ar52_h3_preds_long <- format_output(ar52_h3_preds, "argo_noexog", horizon = 3)
argo_noexog_h3_preds <- ar52_h3_preds

# ARGO2
argo2_noexog_h3_preds <- data.frame(date = index(argo2_noexog_h3$argo2_res), 
                                 coredata(argo2_noexog_h3$argo2_res), 
                                 check.names = FALSE)
argo2_noexog_h3_preds_long <- format_output(argo2_noexog_h3_preds, "argo2_noexog", horizon = 3)

argo2_raw_h3_preds <- data.frame(date = index(argo2_raw_h3$argo2_res), 
                                 coredata(argo2_raw_h3$argo2_res), 
                                 check.names = FALSE)
argo2_raw_h3_preds_long <- format_output(argo2_raw_h3_preds, "argo2_raw", horizon = 3)

argo2_smooth_h3_preds <- data.frame(date = index(argo2_smooth_h3$argo2_res), 
                                    coredata(argo2_smooth_h3$argo2_res), 
                                    check.names = FALSE)
argo2_smooth_h3_preds_long <- format_output(argo2_smooth_h3_preds, "argo2_smooth", horizon = 3)

argo2_detrend_h3_preds <- data.frame(date = index(argo2_detrend_h3$argo2_res), 
                                    coredata(argo2_detrend_h3$argo2_res), 
                                    check.names = FALSE)
argo2_detrend_h3_preds_long <- format_output(argo2_detrend_h3_preds, "argo2_detrend", horizon = 3)

# saving files to csv
save_dir <- "results/argo_results"

## Combining datasets for horizons 0-3 into a single dataset
argo_raw <- rbind(argo_raw_h0_preds_long, argo_raw_h1_preds_long, 
                  argo_raw_h2_preds_long, argo_raw_h3_preds_long)
argo_raw_final <- argo_raw[order(argo_raw$reference_date, argo_raw$location, argo_raw$horizon), ]
write.csv(argo_raw_final, file.path(save_dir, "argo_clusters.csv"), row.names = FALSE)

argo_smooth <- rbind(argo_smooth_h0_preds_long, argo_smooth_h1_preds_long,
                     argo_smooth_h2_preds_long, argo_smooth_h3_preds_long)
argo_smooth_final <- argo_smooth[order(argo_smooth$reference_date, argo_smooth$location, argo_smooth$horizon), ]
write.csv(argo_smooth_final, file.path(save_dir, "argo_smooth.csv"), row.names = FALSE)

argo_detrend <- rbind(argo_detrend_h0_preds_long, argo_detrend_h1_preds_long,
                     argo_detrend_h2_preds_long, argo_detrend_h3_preds_long)
argo_detrend_final <- argo_detrend[order(argo_detrend$reference_date, argo_detrend$location, argo_detrend$horizon), ]
write.csv(argo_detrend_final, file.path(save_dir, "argo_detrend.csv"), row.names = FALSE)

ar52 <- rbind(ar52_h0_preds_long, ar52_h1_preds_long,
             ar52_h2_preds_long, ar52_h3_preds_long)
ar52_final <- ar52[order(ar52$reference_date, ar52$location, ar52$horizon), ]
write.csv(ar52_final, file.path(save_dir, "argo_noexog.csv"), row.names = FALSE)

# computing RMSEs
save_dir_rmse <- "results/forecast_rmses"

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

rmse <- function(truth, predicted) {
  truth <- as.data.frame(truth, check.names = FALSE)
  min_date <- "2022-10-17"
  max_date <- "2024-04-27"
  truth <- truth[truth$date >= min_date & truth$date <= max_date, ]
  predicted <- predicted[predicted$date >= min_date & predicted$date <= max_date, ]
  
  rmses <- data.frame(geo = state_names,
                      rmse = numeric(length(state_names)))
  for (i in 1:length(state_names)) {
    state <- state_names[i]
    rmses$rmse[i] <- sqrt(mean((truth[,gsub(" ", ".", state)] - predicted[,state])^2))
  }
  return(rmses)
}

horizons <- paste0("h", 0:3)
data_names <- c("raw", "smooth", "detrend", "noexog")
method <- "argo" 
print(method)
if(method == "argo") {
  state_names <- c(state_names, "US")
  state_names <- gsub(" ", ".", state_names)
}
for (data_name in data_names) {
  print(data_name)
  all_rmses <- data.frame(geo = state_names,
                          rmse_argo_h0 = numeric(length(state_names)),
                          rmse_argo_h1 = numeric(length(state_names)),
                          rmse_argo_h2 = numeric(length(state_names)),
                          rmse_argo_h3 = numeric(length(state_names)))
  for (h in horizons) {
    dataset_name <- paste0(method, "_", data_name, "_", h, "_preds")
    print(dataset_name)
    dataset <- get(dataset_name)
    result <- rmse(truths, dataset)
    colname <- paste0("rmse_argo_", h)
    all_rmses[[colname]] <- result$rmse
  }
  write.csv(
    all_rmses,
    file.path(save_dir_rmse, paste0(method, "_", data_name, "_rmses.csv")),
    row.names = FALSE
  )
}

