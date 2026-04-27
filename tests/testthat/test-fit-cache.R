test_that("fit_cache_key is stable for identical inputs", {
  config <- make_test_config()
  rd <- make_fake_responses()
  k1 <- fit_cache_key(rd, config, list(chains = 4, iter_warmup = 200))
  k2 <- fit_cache_key(rd, config, list(chains = 4, iter_warmup = 200))
  expect_identical(k1, k2)
})

test_that("fit_cache_key is invariant to fit_args order", {
  config <- make_test_config()
  rd <- make_fake_responses()
  k1 <- fit_cache_key(rd, config, list(chains = 4, iter_warmup = 200))
  k2 <- fit_cache_key(rd, config, list(iter_warmup = 200, chains = 4))
  expect_identical(k1, k2)
})

test_that("fit_cache_key changes when config changes", {
  rd <- make_fake_responses()
  k_linear <- fit_cache_key(rd, make_test_config(measurement = "linear"), list())
  k_inter  <- fit_cache_key(rd, make_test_config(measurement = "interaction"), list())
  expect_false(identical(k_linear, k_inter))
})

test_that("fit_cache_key changes when fit_args change", {
  config <- make_test_config()
  rd <- make_fake_responses()
  k1 <- fit_cache_key(rd, config, list(chains = 4))
  k2 <- fit_cache_key(rd, config, list(chains = 2))
  expect_false(identical(k1, k2))
})

test_that("fit_cache_key changes when responses change", {
  config <- make_test_config()
  k1 <- fit_cache_key(make_fake_responses(seed = 1), config, list())
  k2 <- fit_cache_key(make_fake_responses(seed = 2), config, list())
  expect_false(identical(k1, k2))
})

test_that("fit_cache_key ignores fit_args that don't affect inference", {
  config <- make_test_config()
  rd <- make_fake_responses()
  k1 <- fit_cache_key(rd, config, list(chains = 4, refresh = 0))
  k2 <- fit_cache_key(rd, config, list(chains = 4, refresh = 100))
  expect_identical(k1, k2)
})

test_that("fit_cached returns cached value without calling fit_model", {
  skip_if_not_installed("cachem")
  cache_dir <- withr::local_tempdir()
  cache <- fit_cache(dir = cache_dir)
  config <- make_test_config()
  rd <- make_fake_responses()

  fake <- make_fake_fit_result(config, elpd = -123.45)
  key <- fit_cache_key(rd, config, list(chains = 4))
  cache$set(key, fake)

  # If fit_model were called, it would error (no cmdstan in this code path)
  result <- fit_cached(rd, config, cache = cache, chains = 4)
  expect_s3_class(result, "fit_result")
  expect_equal(result$loo$estimates["elpd_loo", "Estimate"], -123.45)
})

test_that("fit_cached stores result on cache miss", {
  skip_if_not_installed("cachem")
  skip_if_no_cmdstan()
  skip_on_cran()

  cache_dir <- withr::local_tempdir()
  cache <- fit_cache(dir = cache_dir)
  config <- make_test_config()
  sim <- simulate_responses(config, n_students = 50, seed = 7)

  key <- fit_cache_key(sim$responses, config,
                       list(chains = 2, iter_warmup = 100, iter_sampling = 100))
  expect_true(cachem::is.key_missing(cache$get(key)))

  fit_cached(sim$responses, config, cache = cache,
             chains = 2, iter_warmup = 100, iter_sampling = 100)

  hit <- cache$get(key)
  expect_false(cachem::is.key_missing(hit))
  expect_s3_class(hit, "fit_result")
})
