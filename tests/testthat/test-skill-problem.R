test_that("skill_problem constructs a structure_problem of subclass skill_problem", {
  rd <- make_fake_responses()
  problem <- skill_problem(items = make_test_items(), responses = rd)
  expect_s3_class(problem, c("skill_problem", "structure_problem"))
  expect_true(is_structure_problem(problem))
})

test_that("skill_problem does not eagerly call ellmer::chat_anthropic", {
  # No ANTHROPIC_API_KEY needed; we only construct, never invoke LLM slots.
  rd <- make_fake_responses()
  expect_no_error(
    skill_problem(items = make_test_items(), responses = rd)
  )
})

test_that("skill_problem closures see the bound responses and items", {
  rd <- make_fake_responses()
  items <- make_test_items()
  problem <- skill_problem(items = items, responses = rd)

  expect_identical(problem$data$items, items)
  expect_identical(problem$data$responses, rd)
})

test_that("skill_problem$summarize_structure renders the spec slash-label", {
  rd <- make_fake_responses()
  problem <- skill_problem(items = make_test_items(), responses = rd)
  config <- make_test_config(measurement = "interaction",
                             structural = "correlated")
  expect_equal(
    problem$summarize_structure(config),
    "interaction/correlated/single/basic"
  )
})

test_that("skill_problem$validate flags non-identifiable structures", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()
  struc <- loading_structure(taxonomy, assignments, items)
  struc$lambda_mask[, "skill_3"] <- FALSE
  struc$lambda_mask["item_4", "skill_3"] <- TRUE

  config <- base::structure(
    list(spec = model_spec(), structure = struc, edge_prior = NULL),
    class = "model_config"
  )

  rd <- make_fake_responses()
  problem <- skill_problem(items = items, responses = rd)
  v <- problem$validate(config)
  expect_false(v$passed)
})

test_that("skill_problem$cache_key matches fit_cache_key", {
  rd <- make_fake_responses()
  config <- make_test_config()
  problem <- skill_problem(items = make_test_items(), responses = rd)

  expect_identical(
    problem$cache_key(config, list(chains = 4)),
    fit_cache_key(rd, config, list(chains = 4))
  )
})
