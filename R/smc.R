#' Sequential Monte Carlo over structures (minimum viable)
#'
#' Particle-population structure search against any [structure_problem()].
#' K particles are initialised from `problem$propose_initial`, mutated
#' each step via `problem$propose_local_move`, scored by `problem$score`,
#' and reweighted by ELPD. Resampling kicks in when effective sample size
#' falls below K/2.
#'
#' This is the "minimum viable" SMC: bootstrap particle filter with a
#' static target. There is **no tempering and no MH acceptance**, so
#' particle weights are based on raw model fit rather than a properly
#' annealed posterior. Useful for population-based exploration of the
#' structure space and for surfacing multimodal posteriors that greedy
#' search collapses; full Bayesian SMC over structures will add a
#' tempering schedule and MH acceptance in a follow-up.
#'
#' Requires `problem$propose_local_move` to be a function (the mutation
#' kernel). Errors otherwise. The greedy `optimize_structure()` does not
#' need this slot.
#'
#' @param problem A `structure_problem` with a `propose_local_move` slot.
#' @param n_particles Number of particles (default 16).
#' @param n_steps Number of SMC steps (default 10).
#' @param ess_resample_threshold Resample when effective sample size
#'   falls below this fraction of `n_particles`. Default 0.5.
#' @param cache Optional `cachem` cache passed to `problem$score()`.
#' @param ... Extra args forwarded to `problem$score()`.
#' @return An S3 object of class `smc_result` (subclassing
#'   `optimization_result`).
#' @seealso [optimize_structure()], [structure_problem()]
#' @export
optimize_structure_smc <- function(problem,
                                   n_particles = 16L,
                                   n_steps = 10L,
                                   ess_resample_threshold = 0.5,
                                   cache = NULL,
                                   ...) {
  if (!is_structure_problem(problem)) {
    cli_abort("{.arg problem} must be a {.cls structure_problem} object.")
  }
  if (!is.function(problem$propose_local_move)) {
    cli_abort(c(
      "SMC requires {.field problem$propose_local_move} to be a function.",
      i = "Set this slot in your domain constructor (see {.fn causal_problem})."
    ))
  }
  n_particles <- as.integer(n_particles)
  n_steps <- as.integer(n_steps)
  ess_threshold <- ess_resample_threshold * n_particles

  # Initialise: K identical particles. They diversify as soon as the
  # first mutation step runs.
  initial <- problem$propose_initial(...)
  particles <- replicate(n_particles, initial, simplify = FALSE)
  fits <- lapply(particles, function(s) problem$score(s, cache = cache, ...))
  log_weights <- vapply(fits, fit_elpd, numeric(1L))
  if (any(!is.finite(log_weights))) {
    cli_abort("Initial particle score returned non-finite ELPD.")
  }
  log_weights <- normalize_log_weights(log_weights)

  history <- vector("list", n_steps)

  for (t in seq_len(n_steps)) {
    cli_inform("SMC step {t}/{n_steps}")

    ess <- effective_sample_size(log_weights)
    if (ess < ess_threshold) {
      idx <- systematic_resample(exp(log_weights))
      particles <- particles[idx]
      fits <- fits[idx]
      log_weights <- rep(-log(n_particles), n_particles)
      cli_inform("  resampled (ESS {round(ess, 1)} < {round(ess_threshold, 1)})")
    }

    for (i in seq_len(n_particles)) {
      candidate <- problem$propose_local_move(particles[[i]], ...)
      v <- problem$validate(candidate)
      if (!isTRUE(v$passed)) next  # rejection: keep particle as-is

      cand_fit <- problem$score(candidate, cache = cache, ...)
      cand_elpd <- fit_elpd(cand_fit)
      if (!is.finite(cand_elpd)) next  # bad fit; keep particle

      # Bootstrap particle filter: just replace and reweight.
      particles[[i]] <- candidate
      fits[[i]] <- cand_fit
      log_weights[i] <- cand_elpd
    }

    log_weights <- normalize_log_weights(log_weights)

    history[[t]] <- list(
      particles = particles,
      fits = fits,
      log_weights = log_weights,
      ess = effective_sample_size(log_weights)
    )

    cli_inform("  best ELPD = {round(max(vapply(fits, fit_elpd, numeric(1))), 1)}, ESS = {round(history[[t]]$ess, 1)}")
  }

  best_idx <- which.max(log_weights)
  weights <- exp(log_weights)

  base::structure(
    list(
      particles = particles,
      fits = fits,
      log_weights = log_weights,
      weights = weights,
      best = list(
        structure = particles[[best_idx]],
        fit = fits[[best_idx]],
        elpd = fit_elpd(fits[[best_idx]])
      ),
      history = history,
      n_particles = n_particles,
      n_steps = n_steps,
      problem = problem
    ),
    class = c("smc_result", "optimization_result")
  )
}

#' @noRd
fit_elpd <- function(fit) {
  fit$loo$estimates["elpd_loo", "Estimate"]
}

#' Numerically stable log-sum-exp normalisation
#'
#' Returns `log_weights - logSumExp(log_weights)` so that
#' `sum(exp(result)) == 1` to within floating-point.
#'
#' @param log_weights Numeric vector.
#' @return Numeric vector the same length as `log_weights`.
#' @export
normalize_log_weights <- function(log_weights) {
  m <- max(log_weights)
  log_weights - (m + log(sum(exp(log_weights - m))))
}

#' Effective sample size of a particle population
#'
#' `1 / sum(weights^2)` where `weights = exp(log_weights)` (assumed
#' normalised). Returns a value in `[1, n_particles]`.
#'
#' @param log_weights Normalised log-weights.
#' @return Numeric scalar.
#' @export
effective_sample_size <- function(log_weights) {
  weights <- exp(log_weights)
  1 / sum(weights^2)
}

#' Systematic resampling
#'
#' Samples `length(weights)` indices from `weights` using a single
#' uniform draw to seed `K` equally-spaced positions on `[0, 1)`.
#' Lower variance than multinomial resampling.
#'
#' @param weights Normalised weights, summing to 1.
#' @return Integer vector of indices, length `length(weights)`.
#' @export
systematic_resample <- function(weights) {
  K <- length(weights)
  cumw <- cumsum(weights / sum(weights))
  u <- ((seq_len(K) - 1) + stats::runif(1L)) / K
  pmin(findInterval(u, cumw) + 1L, K)
}

#' @export
print.smc_result <- function(x, ..., n_top = 5) {
  cat(sprintf("<smc_result>: %d particles, %d steps\n",
              x$n_particles, x$n_steps))
  cat(sprintf("  Best ELPD: %.1f\n", x$best$elpd))
  if (!is.null(x$problem)) {
    cat(sprintf("  Best: %s\n",
        x$problem$summarize_structure(x$best$structure)))
  }
  cat(sprintf("  Final ESS: %.1f / %d\n",
      effective_sample_size(x$log_weights), x$n_particles))

  ord <- order(x$weights, decreasing = TRUE)
  top <- utils::head(ord, n_top)
  cat("  Top particles:\n")
  for (i in top) {
    cat(sprintf("    [%d] w=%.3f  %s\n",
        i, x$weights[i],
        x$problem$summarize_structure(x$particles[[i]])))
  }
  if (length(ord) > n_top) {
    cat(sprintf("    (... %d more)\n", length(ord) - n_top))
  }
  invisible(x)
}
