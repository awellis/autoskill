# Build a minimal fit_result wrapping a real loo object computed from a
# synthetic log-likelihood matrix. Avoids invoking Stan.
make_synthetic_fit <- function(mean_loglik = -1, sd = 0.05,
                                n_obs = 60, n_draws = 400, seed = 1) {
  withr::with_seed(seed, {
    log_lik <- matrix(rnorm(n_obs * n_draws, mean = mean_loglik, sd = sd),
                      nrow = n_draws, ncol = n_obs)
  })
  loo_obj <- suppressWarnings(loo::loo(log_lik))
  base::structure(
    list(fit = NULL, config = NULL,
         diagnostics = tibble::tibble(),
         loo = loo_obj,
         param_summary = tibble::tibble()),
    class = "fit_result"
  )
}

test_that("compute_stacked_weights returns empty vector for empty input", {
  out <- compute_stacked_weights(list())
  expect_length(out, 0)
  expect_named(out, character(0))
})

test_that("compute_stacked_weights returns weight 1 for a single fit", {
  fit <- make_synthetic_fit()
  out <- compute_stacked_weights(list(only = fit))
  expect_equal(unname(out), 1)
  expect_named(out, "only")
})

test_that("compute_stacked_weights returns valid weights for multiple fits", {
  good <- make_synthetic_fit(mean_loglik = -1.0, seed = 1)
  bad  <- make_synthetic_fit(mean_loglik = -2.0, seed = 2)

  out <- compute_stacked_weights(list(good = good, bad = bad))
  expect_length(out, 2)
  expect_named(out, c("good", "bad"))
  expect_equal(sum(out), 1, tolerance = 1e-6)
  expect_true(all(out >= 0))
  # The better-fitting model should get the larger weight
  expect_gt(out["good"], out["bad"])
})

test_that("compute_stacked_weights auto-names unnamed lists", {
  out <- compute_stacked_weights(list(make_synthetic_fit(seed = 1),
                                       make_synthetic_fit(seed = 2)))
  expect_named(out, c("model_1", "model_2"))
})

test_that("compute_stacked_weights errors when a fit lacks $loo", {
  good <- make_synthetic_fit()
  bad <- base::structure(list(loo = NULL), class = "fit_result")
  expect_error(
    compute_stacked_weights(list(good, bad)),
    "must have a"
  )
})

test_that("compute_stacked_weights honours method = 'pseudobma'", {
  good <- make_synthetic_fit(mean_loglik = -1.0, seed = 1)
  bad  <- make_synthetic_fit(mean_loglik = -2.0, seed = 2)

  out_stack <- compute_stacked_weights(list(good = good, bad = bad),
                                        method = "stacking")
  out_bma <- compute_stacked_weights(list(good = good, bad = bad),
                                      method = "pseudobma")
  expect_equal(sum(out_bma), 1, tolerance = 1e-6)
  # pseudobma is typically more concentrated than stacking when one model
  # dominates, so the better model gets even more weight.
  expect_gte(out_bma["good"], out_stack["good"] - 1e-6)
})

test_that("collect_stacked_weights handles iteration_results with NULL fits", {
  fit <- make_synthetic_fit()
  history <- list(
    list(fit_result = NULL),
    list(fit_result = fit),
    list(fit_result = NULL),
    list(fit_result = make_synthetic_fit(mean_loglik = -1.5, seed = 3))
  )
  out <- collect_stacked_weights(history)
  expect_length(out, 2)
  expect_named(out, c("iter_2", "iter_4"))
  expect_equal(sum(out), 1, tolerance = 1e-6)
})

test_that("collect_stacked_weights returns NULL when no successful fits", {
  history <- list(list(fit_result = NULL), list(fit_result = NULL))
  expect_null(collect_stacked_weights(history))
})
