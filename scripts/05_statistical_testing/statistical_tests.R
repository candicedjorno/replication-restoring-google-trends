# Title: Statistical Tests for Google Trends Manuscript

# =========================================================
# Setup
# =========================================================

knitr::opts_chunk$set(echo = FALSE)

# =========================================================
# Wilcoxon Signed-Rank Test
# =========================================================

# ---------------------------------------------------------
# Each dataset against "no exogenous"
# ---------------------------------------------------------

save_dir <- "results/forecast_rmses"
models <- c("arimax_111", "sarimax_010", "argo", "lgbm", "adaboost")
horizons  <- c('h0', 'h1', 'h2', 'h3')
method_names  <- c('noexog', 'detrend', 'smooth', 'clusters', 'topics', 'indiv')

# Helper function to read and flatten errors
state_mses <- function(model, method) {
  file <- file.path(save_dir, sprintf("%s_%s_rmses.csv", model, method))
  df <- read.csv(file, as.is = TRUE, check.names = FALSE)
  df <- df[complete.cases(df), ]
  # Remove columns if they exist
  rows_to_remove <- c("PR", "Puerto Rico", "Puerto.Rico", "US")
  df <- df[!df$geo %in% rows_to_remove,]
  mses <- df[,2:5] # RMSE
  return(mses)
}

# Store MSEs for all models, horizons, and methods
error_list <- list()

for (model in models) {
  for (method in method_names) {
    key <- paste(model, method, sep = "_")
    error_list[[key]] <- state_mses(model, method)
  }
}

# Run Wilcoxon signed rank tests for all method_names (except "noexog") against "noexog" for each model and horizon
wilcox_results <- list()
for (model in models) {
  for (method_x in method_names[method_names != "noexog"]) {
    key_x <- paste(model, method_x, sep = "_")
    key_noexog <- paste(model, "noexog", sep = "_")
    x <- error_list[[key_x]]
    noexog <- error_list[[key_noexog]]
    for (h in horizons) {
      # Only run test if both vectors exist and are of equal length
      x_h <- x[[paste("rmse", model, h, sep = "_")]]
      noexog_h <- noexog[[paste("rmse", model, h, sep = "_")]]
      if (!is.null(x_h) && !is.null(noexog_h) && length(x_h) == length(noexog_h)) {
        w_res <- wilcox.test(x_h, noexog_h, alternative = "less", paired = TRUE)
        pval <- w_res$p.value
        stars <- ifelse(pval < 0.05, "*", "")
        
        result_key <- paste(model, h, method_x, "vs_noexog", sep = "_")
        wilcox_results[[result_key]] <- list(
          model = model,
          horizon = h,
          name_x = method_x,
          name_y = "noexog",
          p.value = pval,
          w.value = w_res$statistic
        )
        # cat(sprintf("%s: %s vs %s (horizon %s): p-value = %.4g, W = %.4g %s\n",
        #             model, method_x, "noexog", h, round(pval, 3), w_res$statistic, stars))
      }
    }
  }
}

# Initialize an empty data frame to store the results in the desired long format
results_table <- data.frame(
  Method = character(),
  Horizon = character(),
  Model = character(),
  P_value = numeric(),
  Significant = character(),
  stringsAsFactors = FALSE
)

# Populate the table with results
for (key in names(wilcox_results)) {
  result <- wilcox_results[[key]]
  
  pval <- result$p.value
  stars <- ifelse(pval < 0.05, "*", "")
  # Add a row to the table for each combination
  results_table <- rbind(results_table, data.frame(
    Horizon = result$horizon,
    Model = result$model,
    Method = result$name_x,
    P_value = result$p.value,
    Significant = stars,
    stringsAsFactors = FALSE
  ))
}

# Convert the Model column to a factor with the specified levels
results_table$Model <- factor(results_table$Model, levels = models)
# Order the table by Model (factored) and then Horizon
results_table <- results_table[order(results_table$Model, results_table$Horizon), ]

cat("### Models with exogenous data against baselines ###\n")
# print(results_table)

wilcox_exog_vs_baseline <- wilcox_results

# ---------------------------------------------------------
# Each preprocessing step against non-preprocessed data
# ---------------------------------------------------------

# Run Wilcoxon signed rank tests for all method_names (except "indiv") against "indiv" for each model and horizon
wilcox_results <- list()
models <- c("arimax_111", "sarimax_010", "argo", "lgbm", "adaboost")
horizons  <- c('h0', 'h1', 'h2', 'h3')
method_names  <- c('detrend', 'smooth', 'clusters')
for (model in models) {
  for (method_x in method_names[method_names != "indiv"]) {
    key_x <- paste(model, method_x, sep = "_")
    key_noexog <- paste(model, "indiv", sep = "_")
    x <- error_list[[key_x]]
    indiv <- error_list[[key_noexog]]
    for (h in horizons) {
      # Only run test if both vectors exist and are of equal length
      x_h <- x[[paste("rmse", model, h, sep = "_")]]
      noexog_h <- indiv[[paste("rmse", model, h, sep = "_")]]
      if (!is.null(x_h) && !is.null(noexog_h) && length(x_h) == length(noexog_h)) {
        w_res <- wilcox.test(x_h, noexog_h, alternative = "less", paired = TRUE)
        pval <- w_res$p.value
        stars <- ifelse(pval < 0.05, "*", "")
        
        result_key <- paste(model, h, method_x, "vs_noexog", sep = "_")
        wilcox_results[[result_key]] <- list(
          model = model,
          horizon = h,
          name_x = method_x,
          name_y = "indiv",
          p.value = pval,
          w.value = w_res$statistic
        )
        # cat(sprintf("%s: %s vs %s (horizon %s): p-value = %.4g, W = %.4g %s\n",
        #             model, method_x, "indiv", h, round(pval, 3), w_res$statistic, stars))
      }
    }
  }
}

# Initialize an empty data frame to store the results in the desired long format
results_table <- data.frame(
  Method = character(),
  Horizon = character(),
  Model = character(),
  P_value = numeric(),
  Significant = character(),
  stringsAsFactors = FALSE
)

# Populate the table with results
for (key in names(wilcox_results)) {
  result <- wilcox_results[[key]]
  
  pval <- result$p.value
  stars <- ifelse(pval < 0.05, "*", "")
  # Add a row to the table for each combination
  results_table <- rbind(results_table, data.frame(
    Horizon = result$horizon,
    Model = result$model,
    Method = result$name_x,
    P_value = result$p.value,
    Significant = stars,
    stringsAsFactors = FALSE
  ))
}

# Convert the Model column to a factor with the specified levels
results_table$Model <- factor(results_table$Model, levels = models)
# Order the table by Model (factored) and then Horizon
results_table <- results_table[order(results_table$Model, results_table$Horizon), ]

cat("### Preprocessed versus non-preprocessed data ###\n")
# print(results_table)

wilcox_pre_vs_indiv <- wilcox_results

# ---------------------------------------------------------
# Each preprocessing step against topic-only
# ---------------------------------------------------------

# Run Wilcoxon signed rank tests for all method_names (except "topics") against "topics" for each model and horizon
wilcox_results <- list()
models <- c("arimax_111", "sarimax_010", "argo", "lgbm", "adaboost")
horizons  <- c('h0', 'h1', 'h2', 'h3')
method_names  <- c('detrend', 'smooth', 'clusters')
for (model in models) {
  for (method_x in method_names[method_names != "topics"]) {
    key_x <- paste(model, method_x, sep = "_")
    key_noexog <- paste(model, "topics", sep = "_")
    x <- error_list[[key_x]]
    topics <- error_list[[key_noexog]]
    for (h in horizons) {
      # Only run test if both vectors exist and are of equal length
      x_h <- x[[paste("rmse", model, h, sep = "_")]]
      noexog_h <- topics[[paste("rmse", model, h, sep = "_")]]
      if (!is.null(x_h) && !is.null(noexog_h) && length(x_h) == length(noexog_h)) {
        w_res <- wilcox.test(x_h, noexog_h, alternative = "less", paired = TRUE)
        pval <- w_res$p.value
        stars <- ifelse(pval < 0.05, "*", "")
        
        result_key <- paste(model, h, method_x, "vs_topics", sep = "_")
        wilcox_results[[result_key]] <- list(
          model = model,
          horizon = h,
          name_x = method_x,
          name_y = "topics",
          p.value = pval,
          w.value = w_res$statistic
        )
        # cat(sprintf("%s: %s vs %s (horizon %s): p-value = %.4g, W = %.4g %s\n",
        #             model, method_x, "topics", h, round(pval, 3), w_res$statistic, stars))
      }
    }
  }
}

# Initialize an empty data frame to store the results in the desired long format
results_table <- data.frame(
  Method = character(),
  Horizon = character(),
  Model = character(),
  P_value = numeric(),
  Significant = character(),
  stringsAsFactors = FALSE
)

# Populate the table with results
for (key in names(wilcox_results)) {
  result <- wilcox_results[[key]]
  
  pval <- result$p.value
  stars <- ifelse(pval < 0.05, "*", "")
  # Add a row to the table for each combination
  results_table <- rbind(results_table, data.frame(
    Horizon = result$horizon,
    Model = result$model,
    Method = result$name_x,
    P_value = result$p.value,
    Significant = stars,
    stringsAsFactors = FALSE
  ))
}

# Convert the Model column to a factor with the specified levels
results_table$Model <- factor(results_table$Model, levels = models)
# Order the table by Model (factored) and then Horizon
results_table <- results_table[order(results_table$Model, results_table$Horizon), ]

cat("### Preprocessed versus topic-only data ###\n")
# print(results_table)

wilcox_pre_vs_topics <- wilcox_results

# =========================================================
# Build significance-only LaTeX table from three Wilcoxon test blocks
# =========================================================

library(dplyr)
library(tidyr)
library(stringr)
library(purrr)

models <- c("arimax_111", "sarimax_010", "argo", "lgbm", "adaboost")
model_labels <- c(
  arimax_111 = "ARIMAX",
  sarimax_010 = "SARIMAX",
  argo       = "ARGO",
  lgbm       = "LightGBM",
  adaboost   = "AdaBoost"
)

horizons <- c("h0", "h1", "h2", "h3")
methods_order <- c("detrend", "smooth", "clusters", "topics", "indiv")
method_labels <- c(
  detrend = "Detrending",
  smooth  = "Denoising",
  clusters= "Clustering",
  topics  = "Topics",
  indiv   = "Non-preprocessed"
)

# helper to convert wilcox_results list -> tidy df
wilcox_list_to_df <- function(wilcox_results, symbol_name) {
  bind_rows(lapply(wilcox_results, function(x) {
    data.frame(
      Horizon = x$horizon,
      Model   = x$model,
      Method  = x$name_x,
      P_value = x$p.value,
      Symbol  = ifelse(x$p.value < 0.05, symbol_name, ""),
      stringsAsFactors = FALSE
    )
  }))
}

df_star    <- wilcox_list_to_df(wilcox_exog_vs_baseline, "*")
df_dagger  <- wilcox_list_to_df(wilcox_pre_vs_indiv, "\\dagger")
df_ddagger <- wilcox_list_to_df(wilcox_pre_vs_topics, "\\ddagger")

# union all tested cells from first family so Topics/indiv are included there,
# then left-join daggers/ddaggers where applicable (mostly detrend/smooth/clusters)
base_cells <- df_star %>%
  select(Horizon, Model, Method) %>%
  distinct()

all_marks <- base_cells %>%
  left_join(df_star    %>% select(Horizon, Model, Method, star = Symbol), by = c("Horizon","Model","Method")) %>%
  left_join(df_dagger  %>% select(Horizon, Model, Method, dag  = Symbol), by = c("Horizon","Model","Method")) %>%
  left_join(df_ddagger %>% select(Horizon, Model, Method, ddag = Symbol), by = c("Horizon","Model","Method")) %>%
  mutate(
    star = replace_na(star, ""),
    dag  = replace_na(dag, ""),
    ddag = replace_na(ddag, ""),
    sig  = ifelse(star == "" & dag == "" & ddag == "",
                  "",
                  paste0("$^{", star, dag, ddag, "}$"))
  ) %>%
  mutate(
    Horizon_num = as.integer(str_remove(Horizon, "^h")),
    Method = factor(Method, levels = methods_order),
    Model  = factor(Model, levels = models)
  ) %>%
  arrange(Horizon_num, Method, Model)

# Wide format: one row per horizon/method, one column per model
tbl <- all_marks %>%
  select(Horizon_num, Method, Model, sig) %>%
  tidyr::pivot_wider(names_from = Model, values_from = sig) %>%
  arrange(Horizon_num, Method)

# render latex
latex_file <- "tables/table1_wilcoxon_test.tex"
con <- file(latex_file, open = "wt")

writeLines("\\begin{table}[H]", con)
writeLines("\\centering", con)
writeLines("\\resizebox{\\textwidth}{!}{%", con)
writeLines("\\begin{tabular}{llccccc}", con)
writeLines("\\toprule", con)
writeLines("{Horizon} & {Method} & {ARIMAX} & {SARIMAX} & {ARGO} & {LightGBM} & {AdaBoost} \\\\", con)
writeLines("\\midrule", con)

for (h in 0:3) {
  sub <- tbl %>% filter(Horizon_num == h)
  n <- nrow(sub)
  for (i in seq_len(n)) {
    r <- sub[i, ]
    method_lab <- method_labels[[as.character(r$Method)]]
    hcell <- if (i == 1) sprintf("\\multirow{%d}{*}{%d}", n, h) else ""
    line <- sprintf(
      "%s & %s & %s & %s & %s & %s & %s \\\\",
      hcell,
      method_lab,
      ifelse(is.na(r$arimax_111), "", r$arimax_111),
      ifelse(is.na(r$sarimax_010), "", r$sarimax_010),
      ifelse(is.na(r$argo), "", r$argo),
      ifelse(is.na(r$lgbm), "", r$lgbm),
      ifelse(is.na(r$adaboost), "", r$adaboost)
    )
    writeLines(line, con)
  }
  if (h < 3) writeLines("\\midrule", con)
}

writeLines("\\bottomrule", con)
writeLines("\\end{tabular}", con)
writeLines("} % end resizebox", con)
writeLines("\\vspace{5pt}", con)
writeLines("\\caption{Wilcoxon Signed-Rank Test significance markers across horizons, methods, and models. Symbols indicate: $*: p < 0.05$ for models with exogenous data against baselines; $\\dagger: p < 0.05$ for preprocessed versus non-preprocessed data; and $\\ddagger: p < 0.05$ for preprocessed versus topic-only data.}", con)
writeLines("\\label{tab:wilcoxon_significance}", con)
writeLines("\\end{table}", con)

close(con)

cat("Wrote:", latex_file, "\n")

# =========================================================
# Text file
# =========================================================

library(dplyr)
library(tidyr)
library(stringr)

# --- Safety checks ---
if (!exists("tbl")) stop("Object `tbl` not found. Run the Wilcoxon table-building code first.")
if (!exists("method_labels")) stop("Object `method_labels` not found.")

# enforce desired method order
method_order <- c("detrend", "smooth", "clusters", "topics", "indiv")

# Helper to convert latex superscript markers to plain text
conv_sig <- function(x) {
  if (length(x) == 0 || is.null(x) || is.na(x) || x == "") return("")
  s <- as.character(x)
  s <- gsub("^\\$\\^\\{|\\}\\$$", "", s)   # remove $^{ ... }$
  s <- gsub("\\\\dagger", "†", s)
  s <- gsub("\\\\ddagger", "‡", s)
  s <- gsub("\\$", "", s)
  s
}

needed_cols <- c("arimax_111", "sarimax_010", "argo", "lgbm", "adaboost")
for (cc in needed_cols) if (!cc %in% names(tbl)) tbl[[cc]] <- ""

tbl_txt <- tbl %>%
  mutate(
    across(all_of(needed_cols), ~replace_na(as.character(.), "")),
    Horizon_num = as.integer(Horizon_num),
    Method = factor(as.character(Method), levels = method_order, ordered = TRUE)
  ) %>%
  arrange(Horizon_num, Method)

print_df <- tbl_txt %>%
  transmute(
    Horizon = as.character(Horizon_num),
    Method = unname(method_labels[as.character(Method)]),
    ARIMAX = sapply(arimax_111, conv_sig),
    SARIMAX = sapply(sarimax_010, conv_sig),
    ARGO = sapply(argo, conv_sig),
    LightGBM = sapply(lgbm, conv_sig),
    AdaBoost = sapply(adaboost, conv_sig)
  ) %>%
  group_by(Horizon) %>%
  mutate(Horizon = ifelse(row_number() == 1, Horizon, "")) %>%
  ungroup()

# fixed widths (wider method col, compact metric cols centered)
w_h <- max(7, nchar("Horizon"), max(nchar(print_df$Horizon)))
w_m <- max(18, nchar("Method"), max(nchar(print_df$Method)))
w_a <- max(7, nchar("ARIMAX"), max(nchar(print_df$ARIMAX)))
w_s <- max(7, nchar("SARIMAX"), max(nchar(print_df$SARIMAX)))
w_g <- max(6, nchar("ARGO"), max(nchar(print_df$ARGO)))
w_l <- max(8, nchar("LightGBM"), max(nchar(print_df$LightGBM)))
w_d <- max(8, nchar("AdaBoost"), max(nchar(print_df$AdaBoost)))

center <- function(x, width) {
  x <- ifelse(is.na(x), "", x)
  n <- nchar(x, type = "width")
  left <- pmax(0, floor((width - n) / 2))
  right <- pmax(0, width - n - left)
  paste0(strrep(" ", left), x, strrep(" ", right))
}
left <- function(x, width) sprintf(paste0("%-", width, "s"), x)

header <- paste(
  left("Horizon", w_h),
  left("Method", w_m),
  center("ARIMAX", w_a),
  center("SARIMAX", w_s),
  center("ARGO", w_g),
  center("LightGBM", w_l),
  center("AdaBoost", w_d),
  sep = "  "
)

sep <- strrep("-", nchar(header))

txt_file <- "tables/table1_wilcoxon_test.txt"
con <- file(txt_file, open = "wt")

writeLines(header, con)
writeLines(sep, con)

for (i in seq_len(nrow(print_df))) {
  row <- print_df[i, ]
  line <- paste(
    left(row$Horizon, w_h),
    left(row$Method, w_m),
    center(row$ARIMAX, w_a),
    center(row$SARIMAX, w_s),
    center(row$ARGO, w_g),
    center(row$LightGBM, w_l),
    center(row$AdaBoost, w_d),
    sep = "  "
  )
  writeLines(line, con)
  
  # separator after each horizon block
  if (i %% 5 == 0 && i < nrow(print_df)) writeLines(sep, con)
}

writeLines("", con)
writeLines("Notes:", con)
writeLines("* : p < 0.05 for models with exogenous data against baselines", con)
writeLines("† : p < 0.05 for preprocessed versus non-preprocessed data", con)
writeLines("‡ : p < 0.05 for preprocessed versus topic-only data", con)
writeLines("blank: not significant at 0.05", con)

close(con)

cat("TXT table written to:", txt_file, "\n")

# =========================================================
# Panel DM test from Pesaran et al. (2013)
# =========================================================

save_dir <- "results/forecast_errors/"
models <- c("arimax_111", "sarimax_010", "argo", "lgbm", "adaboost")
horizons  <- c('h0', 'h1', 'h2', 'h3')
method_names  <- c('noexog', 'detrend', 'smooth', 'clusters', 'topics', 'indiv')

# Helper function to read and flatten errors
state_errors <- function(model, method, horizon) {
  file <- file.path(save_dir, sprintf("%s_%s_%s_errors.csv", model, method, horizon))
  df <- read.csv(file, as.is = TRUE, check.names = FALSE)
  df <- df[complete.cases(df), ]
  cols_to_remove <- c("PR", "Puerto Rico", "Puerto.Rico", "US")
  df <- df[, !names(df) %in% cols_to_remove]
  return(df)
}

# Store errors for all models, horizons, and methods
error_list <- list()

for (model in models) {
  for (horizon in horizons) {
    for (method in method_names) {
      key <- paste(model, method, horizon, sep = "_")
      error_list[[key]] <- state_errors(model, method, horizon)
    }
  }
}

panel_dm_test_pesaran2013 <- function(errors_model1, errors_model2, h = 1,
                                      weights = NULL,
                                      use_harvey_correction = TRUE) {
  # Pesaran et al.  (2013) Panel DM Test with optional Harvey (1997) correction
  #
  # errors_model1: T x N matrix (time x states)
  # errors_model2: T x N matrix
  # h:  forecast horizon
  # weights: N x 1 vector of state weights (default: equal weights)
  # use_harvey_correction: logical, whether to apply Harvey (1997) finite-sample correction
  
  T <- nrow(errors_model1)  # time periods
  N <- ncol(errors_model1)  # states
  
  # Default: equal weights
  if (is.null(weights)) {
    weights <- rep(1/N, N)
  } else {
    weights <- weights / sum(weights)
  }
  
  # Loss differential z_it(h)
  z <- errors_model1^2 - errors_model2^2  # T x N matrix
  
  # Mean loss differential per state
  z_bar <- colMeans(z)  # N x 1 vector
  
  # Weighted mean
  z_bar_weighted <- sum(weights * z_bar)
  
  # Compute state-specific variances sigma_i^2(h)
  sigma_sq <- numeric(N)
  
  for (i in 1:N) {
    z_i <- z[, i]
    
    # # Assumes forecast errors are serially independent once forecast is made
    # sigma_sq[i] <- sum((z_i - z_bar[i])^2) / (T - 1)
    
    # Original conservative approach was:
    if (h == 1) {
      # For h=1: simple variance
      sigma_sq[i] <- sum((z_i - z_bar[i])^2) / (T - 1)
      
    } else {
      # For h>1: HAC variance with Bartlett window
      s <- h - 1
      # s <- 0
      var_term <- sum((z_i - z_bar[i])^2) / (T - 1)
      
      autocov_sum <- 0
      for (j in 1:s) {
        if (T > j) {
          bartlett_weight <- 1 - j / (s + 1)
          
          autocov_j <- 0
          for (t in (j+1):T) {
            autocov_j <- autocov_j + (z_i[t] - z_bar[i]) * (z_i[t-j] - z_bar[i])
          }
          
          autocov_sum <- autocov_sum + (2 / T) * bartlett_weight * autocov_j
        }
      }
      
      sigma_sq[i] <- var_term + autocov_sum
    }
  }
  
  # Variance of weighted mean
  V_z_bar <- (1/T) * sum(weights^2 * sigma_sq)
  
  # Standard Panel DM statistic
  PDM_standard <- z_bar_weighted / sqrt(V_z_bar)
  
  # ===== HARVEY (1997) MODIFICATION =====
  if (use_harvey_correction) {
    # Harvey et al. (1997) finite-sample correction factor
    harvey_factor <- sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    
    # Modified PDM statistic
    PDM <- PDM_standard * harvey_factor
    
    correction_applied <- TRUE
  } else {
    PDM <- PDM_standard
    harvey_factor <- 1.0
    correction_applied <- FALSE
  }
  
  # P-values (PDM ~ N(0,1) under null)
  # For testing if method1 < method2 (method1 is better):
  # Small p-value (p < 0.05) with negative PDM = method1 significantly better
  p_value_one_sided <- pnorm(PDM)
  p_value_two_sided <- 2 * pnorm(-abs(PDM))
  
  # Significance test
  # Method1 is significantly better if PDM < -1.64 (p < 0.05, one-sided)
  is_significant_better <- (PDM < -1.645)
  # Method1 is significantly worse if PDM > 1.64
  is_significant_worse <- (PDM > 1.645)
  
  return(list(
    PDM_statistic = PDM,
    PDM_standard = PDM_standard,
    harvey_correction_factor = harvey_factor,
    harvey_correction_applied = correction_applied,
    weighted_mean_diff = z_bar_weighted,
    variance = V_z_bar,
    p_value_one_sided = p_value_one_sided,
    p_value_two_sided = p_value_two_sided,
    method1_significantly_better = is_significant_better,
    method1_significantly_worse = is_significant_worse,
    individual_means = z_bar,
    individual_variances = sigma_sq,
    weights = weights,
    n_states = N,
    n_periods = T,
    horizon = h
  ))
}

# get population size for each location
locs <- read.csv("scripts/05_statistical_testing/locations_202324.csv", as.is = TRUE, check.names = FALSE)
locs <- locs[!(locs$abbreviation %in% c("US", "PR") |
                 locs$location_name == "Puerto Rico"), ]

# Function to create population weights from locs dataframe
create_population_weights <- function(locs, error_matrix) {
  # """
  # Create population weights matching the column order of error matrix
  #
  # locs: dataframe with location_name and population columns
  # error_matrix:  T x N matrix with state names as column names
  # """
  
  # Get state names from error matrix columns
  state_names <- colnames(error_matrix)
  
  # Match populations to state order in error matrix
  # This ensures weights align with columns
  state_pops <- sapply(state_names, function(state) {
    # Handle potential naming issues (e.g., spaces, periods)
    idx <- which(locs$location_name == state)
    if (length(idx) == 0) {
      # Try with spaces replaced by periods (common in R column names)
      state_clean <- gsub("\\.", " ", state)
      idx <- which(locs$location_name == state_clean)
    }
    if (length(idx) == 0) {
      warning(paste("State not found:", state))
      return(NA)
    }
    return(locs$population[idx])
  })
  
  # Check for any missing matches
  if (any(is.na(state_pops))) {
    stop("Some states could not be matched to population data")
  }
  
  # Normalize to sum to 1
  pop_weights <- state_pops / sum(state_pops)
  
  return(pop_weights)
}

# Create weights
key_x <- paste("arimax_111", "clusters", "h0", sep = "_")
errors_model1 <- error_list[[key_x]][,-1]
pop_weights <- create_population_weights(locs, errors_model1)

# Verify
cat("Number of weights:", length(pop_weights), "\n")
cat("Sum of weights:", sum(pop_weights), "\n")

dmtest_table <- data.frame(
  Model = character(),
  Horizon = character(),
  Method = character(),
  P_Value = numeric(),
  T_Value = numeric(),
  Significant = character(),
  stringsAsFactors = FALSE
)

for (model in models) {
  for (h in horizons) {
    for (method_x in method_names[method_names != "noexog"]) {
      key_x <- paste(model, method_x, h, sep = "_")
      key_noexog <- paste(model, "noexog", h, sep = "_")
      
      # Extract error matrices for the method and "noexog"
      x <- error_list[[key_x]][,-1]
      noexog <- error_list[[key_noexog]][,-1]
      
      # Run the Panel DM Test
      dm_test <- panel_dm_test_pesaran2013(x, noexog, weights = pop_weights,
                                           h = as.numeric(substring(h, 2)) + 1)
      
      # Calculate p-value and significance
      pval <- unname(dm_test$p_value_one_sided)
      stars <- ifelse(pval < 0.05, "*",
                      ifelse(pval > 0.95, "-", ""))
      
      # Add test result to the data frame
      dmtest_table <- rbind(dmtest_table, data.frame(
        Model = model,
        Horizon = h,
        Method = method_x,
        P_Value = pval,
        T_Value = dm_test$PDM_statistic,
        Significant = stars,
        stringsAsFactors = FALSE
      ))
    }
  }
}

cat("### Panel DM Test Results Table ###\n")
# print(dmtest_table)

# =========================================================
# Build Panel DM significance LaTeX table
# =========================================================

# Assumes dmtest_table already exists from your previous chunk

library(dplyr)
library(tidyr)
library(stringr)

# Keep consistent order/labels
models <- c("arimax_111", "sarimax_010", "argo", "lgbm", "adaboost")
model_labels <- c(
  arimax_111 = "ARIMAX",
  sarimax_010 = "SARIMAX",
  argo = "ARGO",
  lgbm = "LightGBM",
  adaboost = "AdaBoost"
)

method_order <- c("detrend", "smooth", "clusters", "topics", "indiv")
method_labels <- c(
  detrend = "Detrending",
  smooth = "Denoising",
  clusters = "Clustering",
  topics = "Topics",
  indiv = "Non-preprocessed"
)

horizon_order <- c("h0", "h1", "h2", "h3")

# Build a clean wide table with significance symbols only
dm_sig_wide <- dmtest_table %>%
  mutate(
    Horizon_num = as.integer(str_remove(Horizon, "^h")),
    Horizon = factor(Horizon, levels = horizon_order),
    Method = factor(Method, levels = method_order),
    Model = factor(Model, levels = models),
    Sig = case_when(
      P_Value < 0.05 ~ "$*$",
      P_Value > 0.95 ~ "$-$",
      TRUE ~ ""
    )
  ) %>%
  arrange(Horizon, Method, Model) %>%
  select(Horizon_num, Horizon, Method, Model, Sig) %>%
  pivot_wider(names_from = Model, values_from = Sig) %>%
  arrange(Horizon_num, Method)

# Write LaTeX
out_file <- "tables/table2_panel_dm_test.tex"
con <- file(out_file, open = "wt")

writeLines("\\begin{table}[H]", con)
writeLines("\\centering", con)
writeLines("\\begin{tabular}{clccccc}", con)
writeLines("\\toprule", con)
writeLines("Horizon & Method & ARIMAX & SARIMAX & ARGO & LightGBM & AdaBoost \\\\", con)
writeLines("\\midrule", con)

for (h in 0:3) {
  sub <- dm_sig_wide %>% filter(Horizon_num == h)
  n_rows <- nrow(sub)
  
  for (i in seq_len(n_rows)) {
    r <- sub[i, ]
    
    horizon_cell <- if (i == 1) sprintf("\\multirow{%d}{*}{%d}", n_rows, h) else ""
    method_cell <- method_labels[[as.character(r$Method)]]
    
    arimax_cell   <- ifelse(is.na(r$arimax_111), "", r$arimax_111)
    sarimax_cell  <- ifelse(is.na(r$sarimax_010), "", r$sarimax_010)
    argo_cell     <- ifelse(is.na(r$argo), "", r$argo)
    lgbm_cell     <- ifelse(is.na(r$lgbm), "", r$lgbm)
    adaboost_cell <- ifelse(is.na(r$adaboost), "", r$adaboost)
    
    line <- sprintf(
      "%s & %s & %s & %s & %s & %s & %s \\\\",
      horizon_cell, method_cell,
      arimax_cell, sarimax_cell, argo_cell, lgbm_cell, adaboost_cell
    )
    writeLines(line, con)
  }
  
  if (h < 3) writeLines("\\midrule", con)
}

writeLines("\\bottomrule", con)
writeLines("\\end{tabular}", con)
writeLines("\\vspace{5pt}", con)
writeLines("\\caption{Panel Modified Diebold-Mariano test for models with exogenous data against omitting exogenous variables. p-values are reported as $*:  p < 0.05$ for models better than the baseline; $-: p > 0.95$ for models worse than the baseline; and blank cells for non-significant p-values at 0.05.}", con)
writeLines("\\label{tab:panel_dm}", con)
writeLines("\\end{table}", con)

close(con)

cat("LaTeX table written to:", out_file, "\n")

# =========================================================
# Text file
# =========================================================

library(dplyr)

# Output txt file
out_file_txt <- "tables/table2_panel_dm_test.txt"
con_txt <- file(out_file_txt, open = "wt")

# Header
header <- sprintf(
  "%-7s | %-18s | %-7s | %-7s | %-8s | %-8s | %-8s",
  "Horizon", "Method", "ARIMAX", "SARIMAX", "ARGO", "LightGBM", "AdaBoost"
)
sep <- paste(rep("-", nchar(header)), collapse = "")

writeLines(header, con_txt)
writeLines(sep, con_txt)

# Body
for (h in 0:3) {
  sub <- dm_sig_wide %>% filter(Horizon_num == h)
  
  for (i in seq_len(nrow(sub))) {
    r <- sub[i, ]
    
    horizon_cell <- if (i == 1) as.character(h) else ""
    method_cell <- method_labels[[as.character(r$Method)]]
    
    # Convert LaTeX markers to plain text markers
    conv_sig <- function(x) {
      if (is.na(x) || x == "") return("")
      if (x == "$*$") return("*")
      if (x == "$-$") return("-")
      return(x)
    }
    
    arimax_cell   <- conv_sig(r$arimax_111)
    sarimax_cell  <- conv_sig(r$sarimax_010)
    argo_cell     <- conv_sig(r$argo)
    lgbm_cell     <- conv_sig(r$lgbm)
    adaboost_cell <- conv_sig(r$adaboost)
    
    line <- sprintf(
      "%-7s | %-18s | %-7s | %-7s | %-8s | %-8s | %-8s",
      horizon_cell, method_cell,
      arimax_cell, sarimax_cell, argo_cell, lgbm_cell, adaboost_cell
    )
    writeLines(line, con_txt)
  }
  
  if (h < 3) writeLines(sep, con_txt)
}

# Footer / notes
writeLines("", con_txt)
writeLines("Notes:", con_txt)
writeLines("*  : p < 0.05  (model better than baseline)", con_txt)
writeLines("-  : p > 0.95  (model worse than baseline)", con_txt)
writeLines("blank: not significant at 0.05", con_txt)

close(con_txt)

cat("TXT table written to:", out_file_txt, "\n")

# =========================================================
# Fluctuation test
# =========================================================

library(murphydiagram)

save_dir <- "results/forecast_errors/"
models <- c("arimax_111", "sarimax_010", "argo", "lgbm", "adaboost")
horizons  <- c('h0', 'h1', 'h2', 'h3')
method_names  <- c('noexog', 'clusters', 'smooth', 'detrend')

# Helper function to read and flatten errors
state_errors <- function(model, method, horizon) {
  file <- file.path(save_dir, sprintf("%s_%s_%s_errors.csv", model, method, horizon))
  df <- read.csv(file, as.is = TRUE, check.names = FALSE)
  df <- df[complete.cases(df), ]
  cols_to_remove <- c("PR", "Puerto Rico", "Puerto.Rico", "US")
  df <- df[, !names(df) %in% cols_to_remove]
  row_means <- unname(sqrt(rowMeans(df[, !(names(df) %in% "date")]^2))) # RMSE
  return(row_means)
}

# Store errors for all models, horizons, and methods
error_list <- list()

for (model in models) {
  for (horizon in horizons) {
    for (method in method_names) {
      key <- paste(model, method, horizon, sep = "_")
      error_list[[key]] <- state_errors(model, method, horizon)
    }
  }
}

# Map horizons to figure letters
fig_letters <- c(
  h0 = "a",
  h1 = "b",
  h2 = "c",
  h3 = "d"
)

for (h in horizons) {
  
  fig_number <- fig_letters[[h]]
  
  model_display_names <- c(
    arimax_111 = "ARIMAX",
    sarimax_010 = "SARIMAX",
    argo = "ARGO",
    lgbm = "LightGBM",
    adaboost = "AdaBoost"
  )
  
  comparisons <- list(
    list(a = "clusters", b = "smooth",  label = "Clustering vs Denoising"),
    list(a = "smooth",   b = "detrend", label = "Denoising vs Detrending"),
    list(a = "clusters", b = "detrend", label = "Clustering vs Detrending")
  )
  
  # ---- PDF device instead of PNG ----
  pdf(
    file = paste0("figures/figS4", fig_number, "_fluctuation_tests_", h, ".pdf"),
    width = 13.3,
    height = 16
  )
  
  # par(mfrow = c(5, 3), mar = c(4, 6, 2, 4))
  par(mfrow = c(5, 3), mar = c(5, 6, 2, 4))
  
  for (i in seq_along(models)) {
    model <- models[i]
    
    file <- file.path(save_dir, sprintf("%s_%s_%s_errors.csv",
                                        model, "noexog", h))
    df <- read.csv(file, as.is = TRUE, check.names = FALSE)
    
    cols_to_remove <- c("PR", "Puerto Rico", "Puerto.Rico", "US")
    df <- df[, !names(df) %in% cols_to_remove]
    df <- df[complete.cases(df), ]
    
    for (j in seq_along(comparisons)) {
      cmp <- comparisons[[j]]
      
      key_a <- paste(model, cmp$a, h, sep = "_")
      key_b <- paste(model, cmp$b, h, sep = "_")
      
      a_vals <- error_list[[key_a]]
      b_vals <- error_list[[key_b]]
      
      # Suppress warnings when calling fluctuation_test
      suppressWarnings({
        fluctuation_test(
          a_vals,
          b_vals,
          time_labels = df$date,
          conf_level = 0.05,
          mu = 0.3
        )
      })
      
      # Row label (model name)
      if (j == 1) {
        mtext(
          model_display_names[[model]],
          side = 2, line = 2.5, cex = 1.2, font = 2
        )
      }
      
      # Column label
      if (i == 1) {
        mtext(
          cmp$label,
          side = 3, line = -3, cex = 1.1, font = 2
        )
      }
    }
  }
  
  dev.off()
}