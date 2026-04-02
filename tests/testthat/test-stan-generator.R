test_that("generate_stan_code returns valid Stan string", {
  config <- make_test_config()
  code <- generate_stan_code(config)
  expect_type(code, "character")
  expect_length(code, 1)
  expect_match(code, "data \\{")
  expect_match(code, "parameters \\{")
  expect_match(code, "model \\{")
  expect_match(code, "generated quantities \\{")
})

test_that("generate_stan_code includes log_lik", {
  config <- make_test_config()
  code <- generate_stan_code(config)
  expect_match(code, "log_lik")
})

test_that("generate_stan_code works for all FA specs", {
  fa_specs <- list(
    list(m = "linear", s = "independent"),
    list(m = "linear", s = "correlated"),
    list(m = "interaction", s = "independent"),
    list(m = "interaction", s = "correlated")
  )
  for (sp in fa_specs) {
    config <- make_test_config(measurement = sp$m, structural = sp$s)
    code <- generate_stan_code(config)
    expect_type(code, "character")
  }
})

test_that("generate_stan_code works for DAG spec", {
  config <- make_test_config(structural = "dag")
  code <- generate_stan_code(config)
  expect_match(code, "B_free")
  expect_match(code, "topo_order")
})

test_that("generate_stan_code works with slip_guess", {
  config <- make_test_config(item = "slip_guess")
  code <- generate_stan_code(config)
  expect_match(code, "guess")
  expect_match(code, "slip")
})

test_that("all 64 block combinations produce Stan code", {
  specs <- all_valid_specs()
  for (i in seq_len(nrow(specs))) {
    row <- specs[i, ]
    config <- make_test_config(
      measurement = row$measurement,
      structural = row$structural,
      population = row$population,
      item = row$item,
      link = row$link
    )
    code <- generate_stan_code(config)
    expect_type(code, "character")
  }
})
