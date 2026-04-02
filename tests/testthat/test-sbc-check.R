test_that("check_identifiability passes for valid config", {
  config <- make_test_config()
  result <- check_identifiability(config)
  expect_true(result$passed)
  expect_length(result$problems, 0)
})

test_that("check_identifiability catches skills with < 2 items", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()
  struc <- loading_structure(taxonomy, assignments, items)
  # Manually break: set skill_3 to only have 1 item
  struc$lambda_mask[, "skill_3"] <- FALSE
  struc$lambda_mask["item_4", "skill_3"] <- TRUE

  config <- base::structure(
    list(spec = model_spec(), structure = struc, edge_prior = NULL),
    class = "model_config"
  )

  result <- check_identifiability(config)
  expect_false(result$passed)
  expect_true(any(grepl("skill_3", result$problems)))
})

test_that("check_identifiability catches cyclic DAG", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()
  struc <- loading_structure(taxonomy, assignments, items)

  ep <- edge_prior(
    from = c("skill_1", "skill_2", "skill_3"),
    to = c("skill_2", "skill_3", "skill_1"),
    prob = c(0.8, 0.8, 0.8)
  )

  config <- base::structure(
    list(spec = model_spec(structural = "dag"), structure = struc, edge_prior = ep),
    class = "model_config"
  )

  result <- check_identifiability(config)
  expect_false(result$passed)
  expect_true(any(grepl("cycle", result$problems, ignore.case = TRUE)))
})
