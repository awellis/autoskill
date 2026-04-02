test_that("simulate_responses returns correct structure", {
  config <- make_test_config()
  sim <- simulate_responses(config, n_students = 200, seed = 42)
  expect_s3_class(sim$responses, "response_data")
  expect_equal(sim$responses$n_students, 200L)
  expect_equal(sim$responses$n_items, 8L)
  expect_true(all(sim$responses$Y %in% c(0L, 1L)))
  expect_equal(dim(sim$params$theta), c(200, 3))
  expect_equal(length(sim$params$alpha), 8)
})

test_that("simulate_responses is reproducible with seed", {
  config <- make_test_config()
  sim1 <- simulate_responses(config, n_students = 100, seed = 123)
  sim2 <- simulate_responses(config, n_students = 100, seed = 123)
  expect_identical(sim1$responses$Y, sim2$responses$Y)
})

test_that("simulate_responses works for correlated structural", {
  config <- make_test_config(structural = "correlated")
  sim <- simulate_responses(config, n_students = 500, seed = 42)
  cor_mat <- cor(sim$params$theta)
  expect_true(any(abs(cor_mat[upper.tri(cor_mat)]) > 0.1))
})

test_that("simulate_responses works for DAG structural", {
  config <- make_test_config(structural = "dag")
  sim <- simulate_responses(config, n_students = 200, seed = 42)
  expect_equal(dim(sim$params$theta), c(200, 3))
})

test_that("sbc_generator returns a function", {
  config <- make_test_config()
  gen <- sbc_generator(config, n_students = 100)
  expect_type(gen, "closure")
  result <- gen()
  expect_true("variables" %in% names(result))
  expect_true("generated" %in% names(result))
})
