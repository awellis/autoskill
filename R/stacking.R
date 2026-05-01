#' Compute stacking weights over a set of fits
#'
#' Wraps [loo::loo_model_weights()] to produce stacking (or pseudo-BMA)
#' weights over a list of `fit_result` objects fitted to the **same data
#' in the same order**. Useful for Bayesian model averaging across the
#' configurations visited by [optimize_structure()] (or any other
#' user-supplied set of fits).
#'
#' Stacking (Yao et al. 2018) chooses weights that maximise the
#' leave-one-out predictive density of the ensemble. It distributes
#' weight more evenly than pseudo-BMA when several models are
#' statistically indistinguishable, which is the typical situation in
#' structure search where multiple latent-skill specifications give
#' near-equivalent predictive performance.
#'
#' All fits must share the observation set: `loo_model_weights()` aligns
#' models pointwise. If you combine fits from different data subsets the
#' result is meaningless.
#'
#' @param fits A list of `fit_result` objects, each with a `$loo` field.
#'   Names are preserved on the returned vector.
#' @param method One of `"stacking"` (default) or `"pseudobma"`.
#' @param ... Additional arguments to [loo::loo_model_weights()].
#' @return A named numeric vector of weights summing to 1. Length-zero
#'   input returns an empty vector; length-one input returns weight 1.
#' @references Yao, Vehtari, Simpson, Gelman (2018). Using stacking to
#'   average Bayesian predictive distributions. *Bayesian Analysis*,
#'   13(3), 917-1007.
#' @seealso [optimize_structure()], [loo::loo_model_weights()]
#' @export
compute_stacked_weights <- function(fits,
                                    method = c("stacking", "pseudobma"),
                                    ...) {
  method <- match.arg(method)

  if (length(fits) == 0L) {
    return(stats::setNames(numeric(0), character(0)))
  }

  if (is.null(names(fits))) {
    names(fits) <- paste0("model_", seq_along(fits))
  }

  if (length(fits) == 1L) {
    return(stats::setNames(1, names(fits)))
  }

  loos <- purrr::map(fits, "loo")
  if (any(vapply(loos, is.null, logical(1)))) {
    cli_abort("All {.arg fits} must have a {.field $loo} component.")
  }

  # Build a pointwise matrix [n_obs x n_models] from each fit's
  # $loo$pointwise[, "elpd_loo"]. This works whether the underlying
  # objects are full psis_loo (from real Stan fits) or synthetic loo
  # shells (e.g. from causal_problem's BIC scorer).
  pointwise_list <- lapply(loos, function(l) {
    if (is.null(l$pointwise)) {
      cli_abort("Each fit's $loo must have a {.field pointwise} matrix.")
    }
    as.numeric(l$pointwise[, "elpd_loo"])
  })
  obs_counts <- vapply(pointwise_list, length, integer(1L))
  if (length(unique(obs_counts)) > 1L) {
    cli_abort("Fits must be over the same observations; got pointwise lengths {.val {obs_counts}}.")
  }
  lpd_mat <- do.call(cbind, pointwise_list)

  weights <- if (method == "stacking") {
    loo::stacking_weights(lpd_mat, ...)
  } else {
    loo::pseudobma_weights(lpd_mat, ...)
  }

  stats::setNames(as.numeric(weights), names(fits))
}
