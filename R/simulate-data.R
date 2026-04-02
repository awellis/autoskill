#' Simulate student response data from a model configuration
#' @param config A `model_config` object.
#' @param n_students Number of students to simulate.
#' @param seed Random seed for reproducibility.
#' @param params Optional named list of true parameter values.
#' @return A list with responses, params, config.
#' @export
simulate_responses <- function(config, n_students = 200, seed = NULL, params = NULL) {
  if (!is.null(seed)) set.seed(seed)

  mask <- config$structure$lambda_mask
  I <- nrow(mask)
  K <- ncol(mask)
  J <- n_students

  if (is.null(params)) params <- list()

  alpha <- params$alpha %||% stats::rnorm(I, 0, 1)
  n_load <- sum(mask)
  lambda_vals <- params$lambda_free %||% abs(stats::rnorm(n_load, 0.8, 0.3))
  Lambda <- matrix(0, I, K)
  Lambda[mask] <- lambda_vals

  theta <- params$theta %||% simulate_theta(config, J, K)

  eta <- matrix(NA_real_, J, I)
  for (j in seq_len(J)) {
    eta[j, ] <- alpha + as.numeric(Lambda %*% theta[j, ])
  }

  link_inv <- if (config$spec$link == "logit") stats::plogis else stats::pnorm
  prob <- link_inv(eta)

  if (config$spec$item == "slip_guess") {
    guess <- params$guess %||% stats::rbeta(I, 1, 9)
    slip <- params$slip %||% stats::rbeta(I, 1, 9)
    prob <- t(guess + (1 - guess - slip) * t(link_inv(eta)))
    params$guess <- guess
    params$slip <- slip
  }

  Y <- matrix(stats::rbinom(J * I, 1, as.numeric(prob)), nrow = J, ncol = I)
  rownames(Y) <- paste0("student_", seq_len(J))
  colnames(Y) <- rownames(mask)

  params$alpha <- alpha
  params$lambda_free <- lambda_vals
  params$Lambda <- Lambda
  params$theta <- theta

  list(responses = response_data(Y), params = params, config = config)
}

#' @noRd
simulate_theta <- function(config, J, K) {
  switch(config$spec$structural,
    independent = matrix(stats::rnorm(J * K), J, K),
    correlated = simulate_theta_correlated(J, K),
    dag = simulate_theta_dag(config, J, K),
    hierarchical = simulate_theta_hierarchical(J, K)
  )
}

#' @noRd
simulate_theta_correlated <- function(J, K, Sigma = NULL) {
  if (is.null(Sigma)) {
    L <- matrix(0, K, K)
    L[1, 1] <- 1
    for (i in 2:K) {
      for (j in 1:(i - 1)) {
        L[i, j] <- stats::rnorm(1, 0, 0.3)
      }
      L[i, i] <- sqrt(max(0.01, 1 - sum(L[i, 1:(i - 1)]^2)))
    }
    Sigma <- L %*% t(L)
  }
  # Minimal MVN sampler (no MASS dependency)
  L_chol <- chol(Sigma)
  Z <- matrix(stats::rnorm(J * K), J, K)
  sweep(Z %*% L_chol, 2, rep(0, K), "+")
}

#' @noRd
simulate_theta_dag <- function(config, J, K) {
  ep <- config$edge_prior
  skill_ids <- colnames(config$structure$lambda_mask)
  topo <- topological_sort(ep$edges$from, ep$edges$to, skill_ids)
  skill_idx <- setNames(seq_along(skill_ids), skill_ids)
  B <- matrix(0, K, K)
  for (i in seq_len(nrow(ep$edges))) {
    from_idx <- skill_idx[ep$edges$from[i]]
    to_idx <- skill_idx[ep$edges$to[i]]
    B[to_idx, from_idx] <- stats::rnorm(1, 0.5, 0.3)
  }
  theta <- matrix(0, J, K)
  for (node_name in topo) {
    k <- skill_idx[node_name]
    parent_contribution <- theta %*% B[k, ]
    theta[, k] <- as.numeric(parent_contribution) + stats::rnorm(J)
  }
  theta
}

#' @noRd
simulate_theta_hierarchical <- function(J, K) {
  phi <- stats::rnorm(J)
  beta_hier <- abs(stats::rnorm(K, 0.7, 0.2))
  outer(phi, beta_hier) + matrix(stats::rnorm(J * K), J, K)
}

#' Create an SBC generator function
#' @param config A `model_config` object.
#' @param n_students Number of students per simulation.
#' @return A function that returns a list with `variables` and `generated`.
#' @export
sbc_generator <- function(config, n_students = 200) {
  function() {
    sim <- simulate_responses(config, n_students = n_students)
    stan_data <- prepare_stan_data(sim$responses, config)
    named_list <- setNames(
      as.list(c(sim$params$alpha, sim$params$lambda_free)),
      c(
        paste0("alpha[", seq_along(sim$params$alpha), "]"),
        paste0("lambda_free[", seq_along(sim$params$lambda_free), "]")
      )
    )
    list(
      variables = do.call(posterior::draws_matrix, named_list),
      generated = stan_data
    )
  }
}
