#' Sequential Monte Carlo over structures
#'
#' K-particle population sampler against any [structure_problem()].
#' Targets the tempered posterior `π_t(S) ∝ p(S) · exp(γ_t · elpd(S))`
#' along a schedule `γ_t : 0 → 1`. Each step reweights particles by the
#' incremental log-target (closed form, no refit), resamples on low ESS,
#' and mutates via Barker acceptance with `problem$propose_local_move`
#' as the proposal kernel.
#'
#' Final particle weights approximate the posterior over visited
#' structures; use [structure_marginal()] / [edge_marginals()] to
#' summarise.
#'
#' Requires `problem$propose_local_move` to be a function. ELPD plays
#' the role of `log L(S)` in the tempered target — it is a consistent
#' estimator of expected log predictive density and serves as a
#' principled ranking-equivalent to the log marginal likelihood.
#'
#' @param problem A `structure_problem` with `propose_local_move`.
#' @param n_particles Number of particles (default 16).
#' @param n_steps Number of tempering steps (default 10).
#' @param schedule Optional length-`(n_steps + 1)` numeric vector of
#'   temperatures, increasing from 0 to 1. Defaults to a linear schedule.
#' @param n_mutations_per_step Number of Barker MH sweeps per particle
#'   per step (default 1). Increase to improve mixing.
#' @param ess_resample_threshold Resample when ESS falls below this
#'   fraction of `n_particles`. Default 0.5.
#' @param cache Optional `cachem` cache passed to `problem$score()`.
#' @param ... Extra args forwarded to `problem$score()`.
#' @return An S3 object of class `smc_result`.
#' @seealso [structure_marginal()], [edge_marginals()]
#' @export
optimize_structure_smc <- function(problem,
                                   n_particles = 16L,
                                   n_steps = 10L,
                                   schedule = NULL,
                                   n_mutations_per_step = 1L,
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

  if (is.null(schedule)) {
    schedule <- seq(0, 1, length.out = n_steps + 1L)
  }
  if (length(schedule) != n_steps + 1L) {
    cli_abort("{.arg schedule} must have length {.val {n_steps + 1L}}.")
  }
  if (schedule[1] != 0 || schedule[length(schedule)] != 1 ||
      any(diff(schedule) < 0)) {
    cli_abort("{.arg schedule} must increase from 0 to 1.")
  }

  # Initialise: K identical particles. Mutation under γ_0 ≈ 0 acts as a
  # random-walk diffuser before tempering bites.
  initial <- problem$propose_initial(...)
  particles <- replicate(n_particles, initial, simplify = FALSE)
  fits <- lapply(particles, function(s) problem$score(s, cache = cache, ...))
  log_priors <- vapply(particles, problem$log_prior, numeric(1L))
  elpds <- vapply(fits, fit_elpd, numeric(1L))

  if (any(!is.finite(elpds))) {
    cli_abort("Initial scoring returned non-finite ELPD.")
  }
  if (any(!is.finite(log_priors))) {
    cli_abort("Initial log_prior returned non-finite value.")
  }

  # Initial log-weights at γ_0
  log_weights <- log_priors + schedule[1L] * elpds
  log_weights <- normalize_log_weights(log_weights)

  history <- vector("list", n_steps)

  for (t in seq_len(n_steps)) {
    gamma_t <- schedule[t + 1L]
    delta_gamma <- gamma_t - schedule[t]
    cli_inform("SMC step {t}/{n_steps} (gamma {sprintf('%.3f', gamma_t)})")

    # Reweight from π_{t-1} to π_t (closed form, no refit)
    log_weights <- log_weights + delta_gamma * elpds
    log_weights <- normalize_log_weights(log_weights)

    # Resample if effective sample size is too low
    ess <- effective_sample_size(log_weights)
    if (ess < ess_threshold) {
      idx <- systematic_resample(exp(log_weights))
      particles <- particles[idx]
      fits <- fits[idx]
      elpds <- elpds[idx]
      log_priors <- log_priors[idx]
      log_weights <- rep(-log(n_particles), n_particles)
      cli_inform("  resampled (ESS {round(ess, 1)} < {round(ess_threshold, 1)})")
    }

    # Mutate via Barker MH under π_t. Mutation preserves invariance, so
    # weights are unchanged.
    for (sweep in seq_len(n_mutations_per_step)) {
      for (i in seq_len(n_particles)) {
        candidate <- problem$propose_local_move(particles[[i]], ...)
        v <- problem$validate(candidate)
        if (!isTRUE(v$passed)) next

        cand_fit <- problem$score(candidate, cache = cache, ...)
        cand_elpd <- fit_elpd(cand_fit)
        if (!is.finite(cand_elpd)) next
        cand_log_prior <- problem$log_prior(candidate)
        if (!is.finite(cand_log_prior)) next

        log_ratio <- (cand_log_prior - log_priors[i]) +
                     gamma_t * (cand_elpd - elpds[i])
        log_p_accept <- log_ratio - log_sum_exp(c(0, log_ratio))

        if (log(stats::runif(1L)) < log_p_accept) {
          particles[[i]] <- candidate
          fits[[i]] <- cand_fit
          elpds[i] <- cand_elpd
          log_priors[i] <- cand_log_prior
        }
      }
    }

    history[[t]] <- list(
      particles = particles,
      fits = fits,
      log_weights = log_weights,
      gamma = gamma_t,
      ess = effective_sample_size(log_weights)
    )

    cli_inform("  best ELPD = {round(max(elpds), 1)}, ESS = {round(history[[t]]$ess, 1)}")
  }

  best_idx <- which.max(log_weights)
  weights <- exp(log_weights)

  base::structure(
    list(
      particles = particles,
      fits = fits,
      log_weights = log_weights,
      weights = weights,
      schedule = schedule,
      best = list(
        structure = particles[[best_idx]],
        fit = fits[[best_idx]],
        elpd = elpds[best_idx]
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

#' Numerically stable log-sum-exp
#' @param x Numeric vector.
#' @return Numeric scalar `log(sum(exp(x)))`.
#' @export
log_sum_exp <- function(x) {
  m <- max(x)
  if (!is.finite(m)) return(m)
  m + log(sum(exp(x - m)))
}

#' Numerically stable log-sum-exp normalisation
#' @param log_weights Numeric vector.
#' @return Numeric vector the same length as input; `sum(exp(result)) == 1`.
#' @export
normalize_log_weights <- function(log_weights) {
  log_weights - log_sum_exp(log_weights)
}

#' Effective sample size of a particle population
#' @param log_weights Normalised log-weights.
#' @return Numeric scalar in `[1, length(log_weights)]`.
#' @export
effective_sample_size <- function(log_weights) {
  weights <- exp(log_weights)
  1 / sum(weights^2)
}

#' Systematic resampling
#' @param weights Non-negative weights; need not be exactly normalised.
#' @return Integer vector of `length(weights)` indices.
#' @export
systematic_resample <- function(weights) {
  K <- length(weights)
  cumw <- cumsum(weights / sum(weights))
  u <- ((seq_len(K) - 1) + stats::runif(1L)) / K
  pmin(findInterval(u, cumw) + 1L, K)
}

#' Marginal expectation of a structure feature under the SMC posterior
#'
#' Computes `Σ_i w_i · feature_fn(particle_i)` where `w_i` are the
#' particle weights from an [optimize_structure_smc()] run. Use for any
#' summary statistic over the structure posterior: edge inclusion
#' probabilities, expected number of edges/skills, posterior probability
#' that a feature holds, etc.
#'
#' @param smc_result An `smc_result` object.
#' @param feature_fn A function of a single particle, returning a numeric
#'   scalar (or numeric vector — but the same shape across particles).
#' @return Whatever shape `feature_fn` returns, weighted-averaged.
#' @export
structure_marginal <- function(smc_result, feature_fn) {
  if (!inherits(smc_result, "smc_result")) {
    cli_abort("{.arg smc_result} must be an {.cls smc_result} object.")
  }
  features <- lapply(smc_result$particles, feature_fn)
  weights <- smc_result$weights
  Reduce(`+`, Map(function(f, w) f * w, features, weights))
}

#' Marginal edge inclusion probabilities for a DAG-shaped SMC result
#'
#' Convenience wrapper around [structure_marginal()] for SMC runs whose
#' particles are `causal_dag` objects. Returns a K x K matrix where entry
#' `[i, j]` is the posterior probability of the edge `i -> j`.
#'
#' @param smc_result An `smc_result` whose particles are `causal_dag`.
#' @return A K x K numeric matrix with row/column names from the
#'   particles' variable list.
#' @export
edge_marginals <- function(smc_result) {
  if (!inherits(smc_result$particles[[1L]], "causal_dag")) {
    cli_abort("{.fn edge_marginals} requires {.cls causal_dag} particles.")
  }
  structure_marginal(smc_result, function(p) p$adj)
}

#' @export
print.smc_result <- function(x, ..., n_top = 5) {
  cat(sprintf("<smc_result>: %d particles, %d steps\n",
              x$n_particles, x$n_steps))
  cat(sprintf("  Schedule: gamma %g -> %g (%d temperatures)\n",
              x$schedule[1L], x$schedule[length(x$schedule)],
              length(x$schedule)))
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
