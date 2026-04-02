test_that("fit_model returns fit_result with all components", {
  skip_if_no_cmdstan()
  skip_on_cran()
  config <- make_test_config()
  sim <- simulate_responses(config, n_students = 100, seed = 42)
  result <- fit_model(sim$responses, config, chains = 2, iter_warmup = 200, iter_sampling = 200)
  expect_s3_class(result, "fit_result")
  expect_true(!is.null(result$fit))
  expect_s3_class(result$config, "model_config")
  expect_s3_class(result$diagnostics, "tbl_df")
  expect_true(!is.null(result$loo))
  expect_s3_class(result$param_summary, "tbl_df")
})

test_that("extract_diagnostics returns expected metrics", {
  skip_if_no_cmdstan()
  skip_on_cran()
  config <- make_test_config()
  sim <- simulate_responses(config, n_students = 100, seed = 42)
  result <- fit_model(sim$responses, config, chains = 2, iter_warmup = 200, iter_sampling = 200)
  diag <- result$diagnostics
  expect_true("n_divergences" %in% diag$metric)
  expect_true("max_rhat" %in% diag$metric)
  expect_true("min_bulk_ess" %in% diag$metric)
})

test_that("compute_loo returns loo object", {
  skip_if_no_cmdstan()
  skip_on_cran()
  config <- make_test_config()
  sim <- simulate_responses(config, n_students = 100, seed = 42)
  result <- fit_model(sim$responses, config, chains = 2, iter_warmup = 200, iter_sampling = 200)
  expect_s3_class(result$loo, "loo")
})
