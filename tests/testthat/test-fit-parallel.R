test_that("fit_many returns empty list for empty input", {
  rd <- make_fake_responses()
  expect_identical(fit_many(rd, list()), list())
})

test_that("fit_many returns same length as input on cache-hit batch", {
  skip_if_not_installed("furrr")
  skip_if_not_installed("cachem")

  cache_dir <- withr::local_tempdir()
  cache <- fit_cache(dir = cache_dir)
  rd <- make_fake_responses()

  configs <- list(
    make_test_config(measurement = "linear"),
    make_test_config(measurement = "interaction")
  )

  # Prepopulate cache so no Stan fit is needed
  for (cfg in configs) {
    fake <- make_fake_fit_result(cfg, elpd = -100)
    cache$set(fit_cache_key(rd, cfg, list(chains = 2)), fake)
  }

  results <- fit_many(rd, configs, cache = cache, chains = 2)
  expect_length(results, 2)
  expect_true(all(vapply(results, inherits, logical(1), what = "fit_result")))
})

test_that("print.fit_error renders the error message", {
  err <- base::structure(list(config = NULL, error = "bad config"),
                         class = "fit_error")
  expect_output(print(err), "fit_error.*bad config")
})
