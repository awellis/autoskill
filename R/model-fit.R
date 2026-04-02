#' Fit a model to student response data
#' @param responses A `response_data` object.
#' @param config A `model_config` object.
#' @param chains Number of MCMC chains.
#' @param iter_warmup Number of warmup iterations per chain.
#' @param iter_sampling Number of sampling iterations per chain.
#' @param adapt_delta Target acceptance rate.
#' @param max_treedepth Maximum tree depth for NUTS.
#' @param ... Additional arguments passed to `CmdStanModel$sample()`.
#' @return An S3 object of class `fit_result`.
#' @export
fit_model <- function(responses, config,
                      chains = 4, iter_warmup = 1000, iter_sampling = 1000,
                      adapt_delta = 0.95, max_treedepth = 12, ...) {
  model <- compile_model(config)
  stan_data <- prepare_stan_data(responses, config)

  fit <- model$sample(
    data = stan_data, chains = chains,
    iter_warmup = iter_warmup, iter_sampling = iter_sampling,
    adapt_delta = adapt_delta, max_treedepth = max_treedepth,
    refresh = 0, ...
  )

  diagnostics <- extract_diagnostics(fit)
  loo_obj <- compute_loo(fit)
  param_summary <- extract_param_summary(fit, config)

  base::structure(
    list(fit = fit, config = config, diagnostics = diagnostics,
         loo = loo_obj, param_summary = param_summary),
    class = "fit_result"
  )
}

#' Extract MCMC diagnostics
#' @param fit A CmdStanMCMC object.
#' @return A tibble with columns: metric, value, status.
#' @export
extract_diagnostics <- function(fit) {
  sampler_diag <- fit$diagnostic_summary()
  summary_df <- fit$summary()

  n_div <- sum(sampler_diag$num_divergent)
  max_rhat <- max(summary_df$rhat, na.rm = TRUE)
  min_bulk <- min(summary_df$ess_bulk, na.rm = TRUE)
  min_tail <- min(summary_df$ess_tail, na.rm = TRUE)

  tibble::tibble(
    metric = c("n_divergences", "max_rhat", "min_bulk_ess", "min_tail_ess"),
    value = c(n_div, max_rhat, min_bulk, min_tail),
    status = c(
      if (n_div == 0) "ok" else "critical",
      if (max_rhat < 1.01) "ok" else if (max_rhat < 1.05) "warning" else "critical",
      if (min_bulk > 400) "ok" else if (min_bulk > 100) "warning" else "critical",
      if (min_tail > 400) "ok" else if (min_tail > 100) "warning" else "critical"
    )
  )
}

#' Compute LOO-CV via PSIS-LOO
#' @param fit A CmdStanMCMC object with log_lik in generated quantities.
#' @return A loo object.
#' @export
compute_loo <- function(fit) {
  log_lik <- fit$draws("log_lik", format = "matrix")
  loo::loo(log_lik, r_eff = loo::relative_eff(exp(log_lik)))
}

#' Extract parameter summaries mapped to item/skill names
#' @param fit A CmdStanMCMC object.
#' @param config A model_config object.
#' @return A tibble with parameter summaries.
#' @export
extract_param_summary <- function(fit, config) {
  summary_df <- fit$summary() |> tibble::as_tibble()
  summary_df |>
    dplyr::mutate(
      param_type = dplyr::case_when(
        grepl("^alpha\\[", variable) ~ "alpha",
        grepl("^lambda_free\\[", variable) ~ "lambda",
        grepl("^theta\\[", variable) ~ "theta",
        TRUE ~ "other"
      )
    )
}

#' @export
print.fit_result <- function(x, ...) {
  elpd <- x$loo$estimates["elpd_loo", "Estimate"]
  n_critical <- sum(x$diagnostics$status == "critical")
  cat(sprintf("<fit_result>: ELPD = %.1f\n", elpd))
  if (n_critical == 0) cat("  All diagnostics OK\n")
  else cat(sprintf("  ! %d critical diagnostics\n", n_critical))
  invisible(x)
}
