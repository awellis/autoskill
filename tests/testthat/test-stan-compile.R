test_that("compile_model compiles Stan code", {
  skip_if_no_cmdstan()
  config <- make_test_config()
  model <- compile_model(config)
  expect_true(inherits(model, "CmdStanModel"))
})

test_that("compile_model caches by content hash", {
  skip_if_no_cmdstan()
  config <- make_test_config()
  t1 <- system.time(compile_model(config, force = TRUE))
  t2 <- system.time(compile_model(config))
  expect_true(t2["elapsed"] < t1["elapsed"])
})
