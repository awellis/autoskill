#' Prepare Stan data list from response data and model config
#' @param responses A `response_data` object.
#' @param config A `model_config` object.
#' @return A named list suitable for `cmdstanr::CmdStanModel$sample(data = ...)`.
#' @export
prepare_stan_data <- function(responses, config) {
  mask <- config$structure$lambda_mask
  Y <- responses$Y
  I <- ncol(Y)
  J <- nrow(Y)
  K <- ncol(mask)

  # Long-format observation triplets, dropping NAs
  obs <- expand.grid(j = seq_len(J), i = seq_len(I))
  obs$y <- as.integer(Y[cbind(obs$j, obs$i)])
  obs <- obs[!is.na(obs$y), ]

  # Loading index arrays (sparse lambda_mask)
  loading_pairs <- which(mask, arr.ind = TRUE)
  loading_item <- as.integer(loading_pairs[, 1])
  loading_skill <- as.integer(loading_pairs[, 2])

  stan_data <- list(
    N_obs = nrow(obs),
    I = I, J = J, K = K,
    N_loadings = length(loading_item),
    loading_item = loading_item,
    loading_skill = loading_skill,
    ii = as.integer(obs$i),
    jj = as.integer(obs$j),
    y = obs$y
  )

  # DAG-specific data
  if (config$spec$structural == "dag") {
    ep <- config$edge_prior
    skill_ids <- colnames(mask)
    skill_idx <- setNames(seq_along(skill_ids), skill_ids)
    topo <- topological_sort(ep$edges$from, ep$edges$to, skill_ids)
    stan_data$N_edges <- nrow(ep$edges)
    stan_data$edge_from <- as.integer(skill_idx[ep$edges$from])
    stan_data$edge_to <- as.integer(skill_idx[ep$edges$to])
    stan_data$topo_order <- as.integer(skill_idx[topo])
  }

  # Interaction-specific data
  if (config$spec$measurement == "interaction") {
    interact <- compute_interaction_indices(mask)
    stan_data$N_interactions <- nrow(interact)
    stan_data$interact_item <- as.integer(interact$item)
    stan_data$interact_skill1 <- as.integer(interact$skill1)
    stan_data$interact_skill2 <- as.integer(interact$skill2)
  }

  # Grouped population data
  if (config$spec$population == "grouped") {
    if (!is.null(responses$groups)) {
      stan_data$N_groups <- length(unique(responses$groups))
      stan_data$group <- as.integer(as.factor(responses$groups))
    } else {
      stan_data$N_groups <- 1L
      stan_data$group <- rep(1L, J)
    }
  }

  stan_data
}

#' @noRd
compute_interaction_indices <- function(mask) {
  result <- list(item = integer(0), skill1 = integer(0), skill2 = integer(0))
  for (i in seq_len(nrow(mask))) {
    skills <- which(mask[i, ])
    if (length(skills) >= 2) {
      pairs <- utils::combn(skills, 2)
      for (p in seq_len(ncol(pairs))) {
        result$item <- c(result$item, i)
        result$skill1 <- c(result$skill1, pairs[1, p])
        result$skill2 <- c(result$skill2, pairs[2, p])
      }
    }
  }
  tibble::tibble(item = result$item, skill1 = result$skill1, skill2 = result$skill2)
}
