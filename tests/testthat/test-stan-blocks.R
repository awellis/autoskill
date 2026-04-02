# Helper to build a minimal config for testing blocks
make_test_config <- function(measurement = "linear",
                             structural = "independent",
                             population = "single",
                             item = "basic",
                             link = "logit") {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()
  struc <- loading_structure(taxonomy, assignments, items)
  spec <- model_spec(
    measurement = measurement,
    structural = structural,
    population = population,
    item = item,
    link = link
  )
  ep <- NULL
  if (structural == "dag") {
    ep <- edge_prior(
      from = c("skill_1", "skill_1"),
      to = c("skill_2", "skill_3"),
      prob = c(0.8, 0.6)
    )
  }
  model_config(spec, struc, edge_prior = ep)
}

# -- Fragment structure tests --

test_that("all block generators return complete fragment lists", {
  config <- make_test_config()
  expected_keys <- c(
    "functions", "data", "transformed_data", "parameters",
    "transformed_parameters", "model", "generated_quantities"
  )
  frag <- stan_measurement(config)
  expect_named(frag, expected_keys, ignore.order = TRUE)
  frag <- stan_structural(config)
  expect_named(frag, expected_keys, ignore.order = TRUE)
  frag <- stan_population(config)
  expect_named(frag, expected_keys, ignore.order = TRUE)
  frag <- stan_item(config)
  expect_named(frag, expected_keys, ignore.order = TRUE)
})

# -- Measurement block tests --

test_that("stan_measurement_linear declares loading arrays", {
  config <- make_test_config(measurement = "linear")
  frag <- stan_measurement(config)
  expect_match(frag$data, "N_loadings")
  expect_match(frag$data, "loading_item")
  expect_match(frag$data, "loading_skill")
  expect_match(frag$parameters, "lambda_free")
  expect_match(frag$parameters, "lower=0")
  expect_match(frag$parameters, "alpha")
})

test_that("stan_measurement_interaction adds interaction terms", {
  config <- make_test_config(measurement = "interaction")
  frag <- stan_measurement(config)
  expect_match(frag$data, "N_interactions")
  expect_match(frag$parameters, "gamma")
})

# -- Structural block tests --

test_that("stan_structural_independent uses std_normal", {
  config <- make_test_config(structural = "independent")
  frag <- stan_structural(config)
  expect_match(frag$model, "std_normal")
})

test_that("stan_structural_correlated uses LKJ and Cholesky", {
  config <- make_test_config(structural = "correlated")
  frag <- stan_structural(config)
  expect_match(frag$parameters, "L_Omega")
  expect_match(frag$parameters, "cholesky_factor_corr")
  expect_match(frag$model, "lkj_corr_cholesky")
})

test_that("stan_structural_dag declares edge arrays", {
  config <- make_test_config(structural = "dag")
  frag <- stan_structural(config)
  expect_match(frag$data, "N_edges")
  expect_match(frag$data, "edge_from")
  expect_match(frag$data, "edge_to")
  expect_match(frag$parameters, "B_free")
})

# -- Population block tests --

test_that("stan_population_single adds nothing to data", {
  config <- make_test_config(population = "single")
  frag <- stan_population(config)
  expect_equal(frag$data, "")
})

test_that("stan_population_grouped declares group arrays", {
  config <- make_test_config(population = "grouped")
  frag <- stan_population(config)
  expect_match(frag$data, "N_groups")
  expect_match(frag$data, "group")
})

# -- Item block tests --

test_that("stan_item_basic adds no extra parameters", {
  config <- make_test_config(item = "basic")
  frag <- stan_item(config)
  expect_false(grepl("guess", frag$parameters))
  expect_false(grepl("slip", frag$parameters))
})

test_that("stan_item_slip_guess adds guess and slip parameters", {
  config <- make_test_config(item = "slip_guess")
  frag <- stan_item(config)
  expect_match(frag$parameters, "guess")
  expect_match(frag$parameters, "slip")
})

# -- Link function tests --

test_that("logit link uses inv_logit", {
  config <- make_test_config(link = "logit")
  frag <- stan_measurement(config)
  expect_match(frag$generated_quantities, "inv_logit")
})

test_that("probit link uses Phi", {
  config <- make_test_config(link = "probit")
  frag <- stan_measurement(config)
  expect_match(frag$generated_quantities, "Phi")
})
