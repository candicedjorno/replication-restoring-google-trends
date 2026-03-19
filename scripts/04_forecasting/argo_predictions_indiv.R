################################################################################
# Title: ARGO and ARGO2 predictions for horizons 0-3
################################################################################

# Using ARGO and ARGO2 to predict hospitalizations
# library(argo) # using modified argo functions, not the package
library(forecast)
library(zoo)
library(parallel)
library(xts)
source("scripts/04_forecasting/argo_functions.R")
source("scripts/04_forecasting/argo_helper_indiv.R")

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


# locations: all states + PR + DC
geo <- read.csv("data/01_raw/geo_argo.txt", as.is = TRUE, check.names = FALSE, header = FALSE)
geo <- geo$V1

## HORIZON 0 ##

# ARGO1 for horizon 0 for different Google Trends datasets
argo_indiv_h0 <- run_argo1(horizon = 0, "indiv")
argo_topics_h0 <- run_argo1(horizon = 0, "topics")

# ARGO2 for horizon 0 for different Google Trends datasets

argo2_indiv_h0 <- run_argo2(truth_state = argo_indiv_h0$truth_state, 
                          argo_state = argo_indiv_h0$argo_state, 
                          argo_nat = argo_indiv_h0$argo_nat,
                          horizon = 0)
argo2_topics_h0 <- run_argo2(truth_state = argo_topics_h0$truth_state, 
                            argo_state = argo_topics_h0$argo_state, 
                            argo_nat = argo_topics_h0$argo_nat,
                            horizon = 0)

## HORIZON 1 ##

# ARGO1 for horizon 1 for different Google Trends datasets
argo_indiv_h1 <- run_argo1(horizon = 1, "indiv")
argo_topics_h1 <- run_argo1(horizon = 1, "topics")

# retrieving last week of ARGO2 predictions from h0 as fitted value for last week of truth for h1
last_week_indiv <- argo2_indiv_h0$argo2_res[nrow(argo2_indiv_h0$argo2_res),] # h0 preds
truth_indiv_h1 <- c(argo_indiv_h0$truth_state, last_week_indiv)

last_week_topics <- argo2_topics_h0$argo2_res[nrow(argo2_topics_h0$argo2_res),] # h0 preds
truth_topics_h1 <- c(argo_topics_h0$truth_state, last_week_topics)

# ARGO2 for horizon 1 for different Google Trends datasets
argo2_indiv_h1 <- run_argo2(truth_state = truth_indiv_h1, 
                          argo_state = argo_indiv_h1$argo_state, 
                          argo_nat = argo_indiv_h1$argo_nat,
                          horizon = 1)

argo2_topics_h1 <- run_argo2(truth_state = truth_topics_h1, 
                            argo_state = argo_topics_h1$argo_state, 
                            argo_nat = argo_topics_h1$argo_nat,
                            horizon = 1)

## HORIZON 2 ##

# ARGO1 for horizon 2 for different Google Trends datasets
argo_indiv_h2 <- run_argo1(horizon = 2, "indiv")
argo_topics_h2 <- run_argo1(horizon = 2, "topics")

# retrieving last week of ARGO2 predictions from h0 as fitted value for last week of truth for h2
last_week_indiv <- argo2_indiv_h0$argo2_res[nrow(argo2_indiv_h0$argo2_res),] # h0 preds
last_week_indiv2 <- argo2_indiv_h1$argo2_res[nrow(argo2_indiv_h1$argo2_res),] # h1 preds
truth_indiv_h2 <- c(argo_indiv_h0$truth_state, last_week_indiv, last_week_indiv2)

last_week_topics <- argo2_topics_h0$argo2_res[nrow(argo2_topics_h0$argo2_res),] # h0 preds
last_week_topics2 <- argo2_topics_h1$argo2_res[nrow(argo2_topics_h1$argo2_res),] # h1 preds
truth_topics_h2 <- c(argo_topics_h0$truth_state, last_week_topics, last_week_topics2)

# ARGO2 for horizon 2 for different Google Trends datasets
argo2_indiv_h2 <- run_argo2(truth_state = truth_indiv_h2, 
                          argo_state = argo_indiv_h2$argo_state, 
                          argo_nat = argo_indiv_h2$argo_nat,
                          horizon = 2)

argo2_topics_h2 <- run_argo2(truth_state = truth_topics_h2, 
                            argo_state = argo_topics_h2$argo_state, 
                            argo_nat = argo_topics_h2$argo_nat,
                            horizon = 2)

## HORIZON 3 ##

# ARGO1 for horizon 2 for different Google Trends datasets
argo_indiv_h3 <- run_argo1(horizon = 3, "indiv")
argo_topics_h3 <- run_argo1(horizon = 3, "topics")

# retrieving last week of ARGO2 predictions from h0 as fitted value for last week of truth for h3
last_week_indiv <- argo2_indiv_h0$argo2_res[nrow(argo2_indiv_h0$argo2_res),] # h0 preds
last_week_indiv2 <- argo2_indiv_h1$argo2_res[nrow(argo2_indiv_h1$argo2_res),] # h1 preds
last_week_indiv3 <- argo2_indiv_h2$argo2_res[nrow(argo2_indiv_h2$argo2_res),] # h2 preds
truth_indiv_h3 <- c(argo_indiv_h0$truth_state, last_week_indiv, last_week_indiv2, last_week_indiv3)

last_week_topics <- argo2_topics_h0$argo2_res[nrow(argo2_topics_h0$argo2_res),] # h0 preds
last_week_topics2 <- argo2_topics_h1$argo2_res[nrow(argo2_topics_h1$argo2_res),] # h1 preds
last_week_topics3 <- argo2_topics_h2$argo2_res[nrow(argo2_topics_h2$argo2_res),] # h2 preds
truth_topics_h3 <- c(argo_topics_h0$truth_state, last_week_topics, last_week_topics2, last_week_topics3)

# ARGO2 for horizon 2 for different Google Trends datasets
argo2_indiv_h3 <- run_argo2(truth_state = truth_indiv_h3, 
                          argo_state = argo_indiv_h3$argo_state, 
                          argo_nat = argo_indiv_h3$argo_nat,
                          horizon = 3)

argo2_topics_h3 <- run_argo2(truth_state = truth_topics_h3, 
                            argo_state = argo_topics_h3$argo_state, 
                            argo_nat = argo_topics_h3$argo_nat,
                            horizon = 3)

#### OUTPUTS ####

# TRUTH
truths <- data.frame(date = index(argo_topics_h0$truth_nat), 
                     cbind(na.omit(coredata(argo_topics_h0$truth_state)), argo_topics_h0$truth_nat), 
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
argo_indiv_h0_preds <- data.frame(date = index(argo_indiv_h0$argo_state), 
                                cbind(coredata(argo_indiv_h0$argo_state), argo_indiv_h0$argo_nat), 
                                check.names = FALSE)
argo_indiv_h0_preds_long <- format_output(argo_indiv_h0_preds, "argo_indiv", horizon = 0)

argo_topics_h0_preds <- data.frame(date = index(argo_topics_h0$argo_state), 
                                  cbind(coredata(argo_topics_h0$argo_state), argo_topics_h0$argo_nat), 
                                  check.names = FALSE)
argo_topics_h0_preds_long <- format_output(argo_topics_h0_preds, "argo_topics", horizon = 0)

# ARGO2
argo2_indiv_h0_preds <- data.frame(date = index(argo2_indiv_h0$argo2_res), 
                                 coredata(argo2_indiv_h0$argo2_res), 
                                 check.names = FALSE)
argo2_indiv_h0_preds_long <- format_output(argo2_indiv_h0_preds, "argo2_indiv", horizon = 0)


argo2_topics_h0_preds <- data.frame(date = index(argo2_topics_h0$argo2_res), 
                                   coredata(argo2_topics_h0$argo2_res), 
                                   check.names = FALSE)
argo2_topics_h0_preds_long <- format_output(argo2_topics_h0_preds, "argo2_topics", horizon = 0)

## HORIZON 1

# ARGO1
argo_indiv_h1_preds <- data.frame(date = index(argo_indiv_h1$argo_state), 
                                cbind(coredata(argo_indiv_h1$argo_state), argo_indiv_h1$argo_nat), 
                                check.names = FALSE)
argo_indiv_h1_preds_long <- format_output(argo_indiv_h1_preds, "argo_indiv", horizon = 1)

argo_topics_h1_preds <- data.frame(date = index(argo_topics_h1$argo_state), 
                                  cbind(coredata(argo_topics_h1$argo_state), argo_topics_h1$argo_nat), 
                                  check.names = FALSE)
argo_topics_h1_preds_long <- format_output(argo_topics_h1_preds, "argo_topics", horizon = 1)

# ARGO2
argo2_indiv_h1_preds <- data.frame(date = index(argo2_indiv_h1$argo2_res), 
                                 coredata(argo2_indiv_h1$argo2_res), 
                                 check.names = FALSE)
argo2_indiv_h1_preds_long <- format_output(argo2_indiv_h1_preds, "argo2_indiv", horizon = 1)

argo2_topics_h1_preds <- data.frame(date = index(argo2_topics_h1$argo2_res), 
                                   coredata(argo2_topics_h1$argo2_res), 
                                   check.names = FALSE)
argo2_topics_h1_preds_long <- format_output(argo2_topics_h1_preds, "argo2_topics", horizon = 1)

## HORIZON 2

# ARGO1
argo_indiv_h2_preds <- data.frame(date = index(argo_indiv_h2$argo_state), 
                                cbind(coredata(argo_indiv_h2$argo_state), argo_indiv_h2$argo_nat), 
                                check.names = FALSE)
argo_indiv_h2_preds_long <- format_output(argo_indiv_h2_preds, "argo_indiv", horizon = 2)

argo_topics_h2_preds <- data.frame(date = index(argo_topics_h2$argo_state), 
                                  cbind(coredata(argo_topics_h2$argo_state), argo_topics_h2$argo_nat), 
                                  check.names = FALSE)
argo_topics_h2_preds_long <- format_output(argo_topics_h2_preds, "argo_topics", horizon = 2)

# ARGO2
argo2_indiv_h2_preds <- data.frame(date = index(argo2_indiv_h2$argo2_res), 
                                 coredata(argo2_indiv_h2$argo2_res), 
                                 check.names = FALSE)
argo2_indiv_h2_preds_long <- format_output(argo2_indiv_h2_preds, "argo2_indiv", horizon = 2)

argo2_topics_h2_preds <- data.frame(date = index(argo2_topics_h2$argo2_res), 
                                   coredata(argo2_topics_h2$argo2_res), 
                                   check.names = FALSE)
argo2_topics_h2_preds_long <- format_output(argo2_topics_h2_preds, "argo2_topics", horizon = 2)

## HORIZON 3

# ARGO1
argo_indiv_h3_preds <- data.frame(date = index(argo_indiv_h3$argo_state), 
                                cbind(coredata(argo_indiv_h3$argo_state), argo_indiv_h3$argo_nat), 
                                check.names = FALSE)
argo_indiv_h3_preds_long <- format_output(argo_indiv_h3_preds, "argo_indiv", horizon = 3)

argo_topics_h3_preds <- data.frame(date = index(argo_topics_h3$argo_state), 
                                  cbind(coredata(argo_topics_h3$argo_state), argo_topics_h3$argo_nat), 
                                  check.names = FALSE)
argo_topics_h3_preds_long <- format_output(argo_topics_h3_preds, "argo_topics", horizon = 3)

# ARGO2
argo2_indiv_h3_preds <- data.frame(date = index(argo2_indiv_h3$argo2_res), 
                                 coredata(argo2_indiv_h3$argo2_res), 
                                 check.names = FALSE)
argo2_indiv_h3_preds_long <- format_output(argo2_indiv_h3_preds, "argo2_indiv", horizon = 3)

argo2_topics_h3_preds <- data.frame(date = index(argo2_topics_h3$argo2_res), 
                                   coredata(argo2_topics_h3$argo2_res), 
                                   check.names = FALSE)
argo2_topics_h3_preds_long <- format_output(argo2_topics_h3_preds, "argo2_topics", horizon = 3)

## Combining datasets for horizons 0-3 into a single dataset
argo_indiv <- rbind(argo_indiv_h0_preds_long, argo_indiv_h1_preds_long, 
                  argo_indiv_h2_preds_long, argo_indiv_h3_preds_long)
argo_indiv_final <- argo_indiv[order(argo_indiv$reference_date, argo_indiv$location, argo_indiv$horizon), ]
write.csv(argo_indiv_final, "results/argo_results/argo_indiv.csv", row.names = FALSE)

argo_topics <- rbind(argo_topics_h0_preds_long, argo_topics_h1_preds_long, 
                    argo_topics_h2_preds_long, argo_topics_h3_preds_long)
argo_topics_final <- argo_topics[order(argo_topics$reference_date, argo_topics$location, argo_topics$horizon), ]
write.csv(argo_topics_final, "results/argo_results/argo_topics.csv", row.names = FALSE)

# computing RMSEs

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
    # print(state)
    rmses$rmse[i] <- sqrt(mean((truth[,gsub(" ", ".", state)] - predicted[,state])^2))
  }
  return(rmses)
}

horizons <- paste0("h", 0:3)
data_names <- c("indiv", "topics")
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
  write.csv(all_rmses, paste0("results/forecast_rmses/", method, "_", data_name, "_rmses.csv"), row.names = FALSE)
}

