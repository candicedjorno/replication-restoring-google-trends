################################################################################
# Title: ARGO and ARGO2 functions modified from the `argo` package
################################################################################

# ARGO and ARGO2 functions
# Re-coded to modify hard-coded parts
# Modified both ARGO and ARGO2 functions for them to output predictions 
# for the week following the last week of hospitalizations
# Detect number of cores
# Using correlations to select variables

filter_corr <- function(data, y, percentile_value, top_n, corr_cutoff = 0.9) {
  # data: predictors (data.frame or matrix)
  # y: response (numeric vector, same length as nrow(data))
  # percentile_value: percentile for cutoff on correlation
  # top_n: max number of predictors to keep
  # corr_cutoff: threshold for removing highly correlated predictors
  
  # 1. Compute absolute correlations
  corrs <- apply(data, 2, function(x) cor(x, y, use = "complete.obs"))
  corrs_abs <- abs(corrs)
  
  # 2. Percentile-based threshold
  corr_threshold <- quantile(corrs_abs, probs = percentile_value/100, na.rm = TRUE)
  selected <- which(corrs_abs >= corr_threshold)
  selected_names <- names(corrs_abs)[selected]
  
  # 3. Remove highly correlated pairs
  if (length(selected_names) > 1) {
    sub_data <- data[, selected_names, drop = FALSE]
    sub_corrmat <- abs(cor(sub_data, use = "pairwise.complete.obs"))
    to_remove <- character(0)
    for (i in seq_along(selected_names)) {
      for (j in (i + 1):length(selected_names)) {
        if (j > length(selected_names)) next
        if (sub_corrmat[i, j] > corr_cutoff) {
          to_remove <- c(to_remove, selected_names[j])
        }
      }
    }
    final_names <- setdiff(selected_names, to_remove)
  } else {
    final_names <- selected_names
  }
  
  # 4. Take top_n by correlation
  if (length(final_names) > top_n) {
    top_names <- names(sort(corrs_abs[final_names], decreasing = TRUE))[1:top_n]
  } else {
    top_names <- final_names
  }
  
  return(list(
    selected_names = top_names,
    selected_corrs = corrs[top_names],
    corr_threshold = corr_threshold
  ))
}

num_cores <- detectCores() - 1  # Leave 1 core free for other tasks
# num_cores <- 1

argo <- function(data, exogen=xts::xts(NULL), N_lag=1:52, N_training=104,
                 percentile_value, top_n, corr_cutoff = 0.9,
                 alpha, use_all_previous=FALSE, mc.cores=num_cores, schedule = list()){

  data_save <- data
  if(is.null(schedule$y_gap)){
    schedule$y_gap <- 1 # default information gap is 1
  }
  if(is.null(schedule$forecast)){
    schedule$forecast <- 0 # default is now-cast
  }

  parm <- list(N_lag = N_lag, N_training = N_training,
               alpha = alpha, use_all_previous = use_all_previous,
               schedule = schedule)
  if(ncol(data)==1){
    data_mat <- matrix(rep(data, nrow(data)), nrow=nrow(data))
    colnames(data_mat) <- as.character(index(data))
    for(i in schedule$y_gap:min(ncol(data_mat), ncol(data_mat)-1+schedule$y_gap)){
      data_mat[(i-schedule$y_gap+1):nrow(data_mat),i] <- NA
    }
    data_mat <- xts::xts(data_mat, zoo::index(data))
    data <- data_mat
  }

  lasso.pred <- c()
  lasso.coef <- list()

  if(length(exogen)>0) # exogenous variables must have the same timestamp as y
    if(!all(zoo::index(data)==zoo::index(exogen)))
      stop("error in data and exogen: their time steps must match")

  starttime <- N_training+max(c(N_lag,0)) + 2*schedule$y_gap + schedule$forecast
  endtime <- nrow(data)
  
  y_train <- data_save[index(data_save) < as.Date("2022-10-01")]
  exogen_train <- exogen[index(exogen) < as.Date("2022-10-01")]
  
  if(length(exogen) > 0 && schedule$y_gap > 0){
    corr_results <- filter_corr(exogen_train, y_train,
                                percentile_value = percentile_value,
                                top_n = top_n, corr_cutoff = corr_cutoff)
    X_selected <- exogen[, corr_results$selected_names, drop = FALSE]
    exogen <- X_selected
  }else{
    exogen <- exogen
  }

  each_iteration <- function(i) {
    if(use_all_previous){
      training_idx <- (schedule$y_gap + schedule$forecast + max(c(N_lag, 0))):(i - schedule$y_gap)
    }else{
      training_idx <- (i - N_training + 1):i - schedule$y_gap
    }

    lagged_y <- sapply(N_lag, function(l)
      as.numeric(diag(data.matrix(data[training_idx - l + 1 - schedule$y_gap - schedule$forecast,training_idx- schedule$forecast]))))

    if(length(lagged_y) == 0){
      lagged_y <- NULL
    }else{
      colnames(lagged_y) <- paste0("lag_", N_lag+schedule$y_gap-1)
    }

    # design matrix for training phase
    if(length(exogen) > 0 && schedule$y_gap > 0){
      xmat <- lapply(1:schedule$y_gap, function(l)
        as.matrix(exogen[training_idx - schedule$forecast - l + 1, ]))
      xmat <- do.call(cbind, xmat)
      design_matrix <- cbind(lagged_y, xmat)
    }else{
      design_matrix <- cbind(lagged_y)
    }
    y.response <- data[training_idx - schedule$forecast, i]
    

    if(is.finite(alpha)){
      lasso.fit <-
        glmnet::cv.glmnet(x=design_matrix,y=y.response,nfolds=10,
                          grouped=FALSE,alpha=alpha)

      lam.s <- lasso.fit$lambda.1se

    }else{
      lasso.fit <- lm(y.response ~ ., data=data.frame(design_matrix))
    }

    if(is.finite(alpha)){
      lasso.coef[[i]] <- as.matrix(coef(lasso.fit, lambda = lam.s))
    }else{
      lasso.coef[[i]] <- as.matrix(coef(lasso.fit))
    }
    lagged_y_next <- matrix(sapply(N_lag, function(l)
      as.numeric(data[i - schedule$y_gap + 1 - schedule$forecast - l, i])), nrow=1)

    # design matrix for new predictors during forecast phase
    # on unseen datapoints
    if(length(lagged_y_next) == 0)
      lagged_y_next <- NULL
    if(length(exogen) > 0 && schedule$y_gap > 0){
      xmat.new <- lapply(1:(schedule$y_gap), function(l)
        data.matrix(exogen[i - l + 1, ]))
      xmat.new <- do.call(cbind, xmat.new)
      newx <- cbind(lagged_y_next, xmat.new)
    }else{
      newx <- lagged_y_next
    }
    if(is.finite(alpha)){
      # forecasting
      if(length(exogen) > 0 && schedule$y_gap > 0){
        colnames(newx) <- c(paste0("lag_", N_lag+schedule$y_gap-1), colnames(xmat.new))
        lasso.pred[i] <- predict(lasso.fit, newx = newx, s = lam.s)
      } else{
        lasso.pred[i] <- predict(lasso.fit, newx = newx, s = lam.s)
      }
    }else{
      colnames(newx) <- c(paste0("lag_", N_lag+schedule$y_gap-1), colnames(xmat.new))
      newx <- as.data.frame(newx)
      colnames(newx) <- make.names(colnames(newx))
      lasso.pred[i] <- predict(lasso.fit, newdata = as.data.frame(newx))
    }
    result_i <- list()
    result_i$pred <- lasso.pred[i]
    result_i$coef <- lasso.coef[[i]]
    result_i
  }

  result_all <- parallel::mclapply(starttime:endtime, each_iteration,
                                   mc.cores = mc.cores, mc.set.seed = FALSE)

  lasso.pred[starttime:endtime] <- sapply(result_all, function(x) x$pred)
  lasso.coef <- lapply(result_all, function(x) x$coef)

  data$predict <- lasso.pred
  argo <- list(pred = data$predict, parm = parm)
  class(argo) <- "argo"
  argo
}


argo2 <- function(truth, argo1.p, argo.nat.p, N_training=104, horizon){ 
  # adding the last value of truth as the prediction for the next time step
  # for which there is no true hospitalizations
  if (horizon == 0){
    naive_pred <- lag(truth, 1)
    naive.p <- c(naive_pred, truth[nrow(truth)])
    index(naive.p) <- index(argo1.p)
  } else if (horizon == 1) {
    naive_pred <- lag(truth[1:(nrow(truth)-1),], 2)
    last_rows <- truth[(nrow(truth)-2):(nrow(truth)-1),]
    index(last_rows) <- index(last_rows)+14 # shifting index of last 2 rows
    naive.p <- c(naive_pred, last_rows)
  } else if (horizon == 2) {
    naive_pred <- lag(truth[1:(nrow(truth)-2),], 3)
    last_rows <- truth[(nrow(truth)-4):(nrow(truth)-2),]
    index(last_rows) <- index(last_rows)+21 # shifting index of last 3 rows
    naive.p <- c(naive_pred, last_rows)
  } else if (horizon == 3) {
    naive_pred <- lag(truth[1:(nrow(truth)-3),], 4)
    last_rows <- truth[(nrow(truth)-6):(nrow(truth)-3),]
    index(last_rows) <- index(last_rows)+28 # shifting index of last 4 rows
    naive.p <- c(naive_pred, last_rows)
  }
  
  common_idx <- zoo::index(na.omit(merge(truth, naive.p, argo1.p, argo.nat.p)))
  if(ncol(argo.nat.p)==1){
    argo.nat.p <- do.call(cbind, lapply(1:ncol(truth), function(i) argo.nat.p))
  }
  colnames(argo.nat.p) <- colnames(truth)
  
  truth_save <- truth
  # adding a row of NAs to truth, which is the next time step to predict
  last_row <- xts(matrix(as.numeric(NA), nrow = 1, ncol = ncol(truth)), 
                  order.by = index(naive.p[nrow(naive.p)]))
  colnames(last_row) <- colnames(truth)
  truth <- rbind(truth_save, last_row)
  
  Z <- truth - naive.p
  lag_Z <- lag(Z, 1)
  W <- merge(lag_Z, argo1.p - naive.p, argo.nat.p - naive.p)
  
  arg2zw_result <- argo2zw(Z, W, N_training=104, truth) # function def modified
  Z.hat <- arg2zw_result$Z.hat
  Z.hat <- xts(Z.hat, as.Date(rownames(Z.hat)))
  
  argo2.p <- Z.hat + naive.p
  
  heat.vec <- na.omit(merge(Z, lag(Z), argo1.p - truth, argo.nat.p - truth))
  colnames(heat.vec) <- paste0(rep(c("detla.ili.", "err.argo.", "err.nat.", "lag.detla.ili."), each=ncol(truth)), rep(1:ncol(truth),3))
  result_wrapper <- list(onestep=argo1.p, twostep=argo2.p, naive=naive.p, truth=truth,
                         heat.vec=heat.vec)
  c(result_wrapper, arg2zw_result)
}

# function definition has been modified
argo2zw <- function(Z, W, N_training=104, truth){
  zw.mat <- as.matrix(merge(Z, W)) # not removing NAs to keep the last row to predict
  if(is.null(rownames(zw.mat)))
    stop("row name must exist as index")
  projection.mat <- list()
  mean.mat <- list()
  zw_used <- list()
  
  sigma_ww.structured <- sigma_ww.empirical <-
    sigma_zw.structured <- sigma_zw.empirical <-
    heat.vec.structured <-
    sigma_zwzw.structured <- sigma_zwzw.empirical <- list()
  
  epsilon <- list()
  Z <- zw.mat[,1:ncol(truth)]
  z_columns_indices <- 1:ncol(truth)
  W <- zw.mat[, -z_columns_indices]
  W1 <- W[,1:ncol(truth)]
  argo1_columns <- (ncol(truth)+1):(2*ncol(truth))
  W2 <- W[,argo1_columns]
  argo_nat_columns <- (2*ncol(truth)+1):(3*ncol(truth))
  W3 <- W[,argo_nat_columns]
  
  Z.hat <- Z
  Z.hat[] <- NA
  for(it in (N_training+1):nrow(zw.mat)){
    training_idx <- (it-N_training):(it-1)
    t.now <- rownames(zw.mat)[it]
    if(is.null(t.now))
      t.now <- it
    
    epsilon[[as.character(t.now)]] <- list()
    
    # omitting the first row of NAs
    sigma_zz <- var(na.omit(Z[training_idx,]))
    sigma_zz_chol <- chol(sigma_zz)
    epsilon[[as.character(t.now)]]$z <- solve(t(sigma_zz_chol), t(data.matrix(Z[training_idx,])))
    
    zw_used[[as.character(t.now)]] <- cbind(Z[training_idx,], W[training_idx,])
    
    m1 <- cor(Z[training_idx,], W[training_idx,z_columns_indices]) 
    m2 <- cor(Z[training_idx,])
    rho <- sum(m1*m2)/sum(m2^2)
    
    d.gt <- diag(diag(var(W2[training_idx,] - Z[training_idx,])))
    epsilon[[as.character(t.now)]]$argoreg <- t(data.matrix((W2 - Z)[training_idx,]))
    epsilon[[as.character(t.now)]]$argoreg <- epsilon[[as.character(t.now)]]$argoreg / sqrt(diag(d.gt))
    # omitting the first row of NAs
    sigma.nat <- var(na.omit((W3 - Z)[training_idx,]))
    epsilon[[as.character(t.now)]]$argonat <- t(data.matrix((W3 - Z)[training_idx,]))
    sigma.nat_chol <- chol(sigma.nat)
    epsilon[[as.character(t.now)]]$argonat <- solve(t(sigma.nat_chol), epsilon[[as.character(t.now)]]$argonat)
    
    sigma_ww <- rbind(
      cbind(sigma_zz, rho*sigma_zz, rho*sigma_zz),
      cbind(rho*sigma_zz, sigma_zz + d.gt, sigma_zz),
      cbind(rho*sigma_zz, sigma_zz, sigma_zz + sigma.nat)
    )
    
    sigma_zw <- cbind(rho*sigma_zz, sigma_zz, sigma_zz)
    
    mu_w <- colMeans(W[training_idx,])
    mu_z <- colMeans(Z[training_idx,])
    
    d_ww <- diag(diag(var(W[training_idx,])))
    
    pred.blp <- mu_z + sigma_zw %*% solve(sigma_ww, W[t.now,] - mu_w)
    
    Kzz <- solve((1-rho^2)*sigma_zz)
    Kgt <- diag(1/diag(var((W2 - Z)[training_idx,])))
    Knat <- solve(var((W3 - Z)[training_idx,]))
    
    
    pred.bayes <- mu_z +
      solve(Kzz+Knat+Kgt,
            Knat%*%(W[t.now,argo_nat_columns] - mu_w[argo_nat_columns]) + 
              Kgt%*%(W[t.now,argo1_columns] - mu_w[argo1_columns]) +
              Kzz%*%(rho*(W[t.now,z_columns_indices] - mu_w[z_columns_indices])))
    
    
    if(all(is.finite(pred.blp))){
      stopifnot(all(abs(pred.blp-pred.bayes) < 1e-5))
    }
    
    # shrinked
    z.hat <- mu_z + sigma_zw %*% solve(sigma_ww + d_ww, (W[t.now,] - mu_w))
    Z.hat[t.now, ] <- t(z.hat)
    
    projection.mat[[as.character(t.now)]] <- sigma_zw %*% solve(sigma_ww + d_ww)
    mean.mat[[as.character(t.now)]] <- c(mu_z, mu_w)
    sigma_ww.structured[[as.character(t.now)]] <- sigma_ww
    sigma_ww.empirical[[as.character(t.now)]] <- var(W[training_idx,])
    sigma_zw.structured[[as.character(t.now)]] <- sigma_zw
    sigma_zw.empirical[[as.character(t.now)]] <- cov(Z[training_idx,], W[training_idx,])
    
    sigma_zwzw.structured[[as.character(t.now)]] <- rbind(
      cbind(sigma_zz, sigma_zw),
      cbind(t(sigma_zw), sigma_ww)
    )
    sigma_zwzw.empirical[[as.character(t.now)]] <- var(cbind(Z, W)[training_idx,])
    
    heat.vec.struc <- rbind(
      cbind(sigma_zz, rho*sigma_zz),
      cbind(rho*sigma_zz, sigma_zz)
    )
    heat.vec.struc <- Matrix::bdiag(heat.vec.struc, d.gt, sigma.nat)
    heat.vec.structured[[as.character(t.now)]] <- as.matrix(heat.vec.struc)
  }
  projection.mat <- sapply(projection.mat, identity, simplify = "array")
  mean.mat <- sapply(mean.mat, identity, simplify = "array")
  
  sigma_ww.structured <- sapply(sigma_ww.structured, identity, simplify = "array")
  sigma_ww.empirical <- sapply(sigma_ww.empirical, identity, simplify = "array")
  sigma_zw.structured <- sapply(sigma_zw.structured, identity, simplify = "array")
  sigma_zw.empirical <- sapply(sigma_zw.empirical, identity, simplify = "array")
  sigma_zwzw.structured <- sapply(sigma_zwzw.structured, identity, simplify = "array")
  sigma_zwzw.empirical <- sapply(sigma_zwzw.empirical, identity, simplify = "array")
  zw_used <- sapply(zw_used, identity, simplify = "array")
  
  heat.vec.structured <- sapply(heat.vec.structured, identity, simplify = "array")
  
  return(list(
    Z.hat=Z.hat,
    heat.vec.structured=heat.vec.structured,
    projection.mat=projection.mat, mean.mat=mean.mat,
    sigma_ww.structured=sigma_ww.structured, sigma_ww.empirical=sigma_ww.empirical,
    sigma_zw.structured=sigma_zw.structured, sigma_zw.empirical=sigma_zw.empirical,
    sigma_zwzw.structured=sigma_zwzw.structured, sigma_zwzw.empirical=sigma_zwzw.empirical,
    zw_used=zw_used, epsilon=epsilon
  ))
}
