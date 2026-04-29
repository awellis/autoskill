test_that("run_iteration returns iteration_result", {
  skip_if_no_cmdstan()
  skip_on_cran()
  config <- make_test_config()
  sim <- simulate_responses(config, n_students = 100, seed = 42)
  problem <- skill_problem(items = make_test_items(),
                           responses = sim$responses)

  result <- run_iteration(problem, config,
                          chains = 2, iter_warmup = 200, iter_sampling = 200)
  expect_s3_class(result, "iteration_result")
  expect_true(result$identifiable)
  expect_s3_class(result$fit_result, "fit_result")
})

test_that("run_iteration rejects non-identifiable structure", {
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

  Y <- matrix(sample(0:1, 100 * 8, replace = TRUE), 100, 8)
  rownames(Y) <- paste0("s_", 1:100)
  colnames(Y) <- paste0("item_", 1:8)
  rd <- response_data(Y)

  problem <- skill_problem(items = items, responses = rd)
  result <- run_iteration(problem, config)

  expect_s3_class(result, "iteration_result")
  expect_false(result$identifiable)
  expect_null(result$fit_result)
})

test_that("run_iteration rejects non-structure_problem first arg", {
  expect_error(
    run_iteration(list(), make_test_config()),
    "must be a"
  )
})

test_that("optimize_structure rejects non-structure_problem first arg", {
  expect_error(
    optimize_structure(list()),
    "must be a"
  )
})

test_that("log_iteration writes JSONL", {
  tmp <- tempfile(fileext = ".jsonl")
  on.exit(unlink(tmp))

  log_iteration(
    iter = 1, elpd = -150.3, improved = TRUE,
    diagnostics = tibble::tibble(metric = "n_divergences", value = 0, status = "ok"),
    label = "linear/independent/single/basic",
    rationale = "initial model", log_file = tmp
  )

  lines <- readLines(tmp)
  expect_length(lines, 1)
  parsed <- jsonlite::fromJSON(lines[1])
  expect_equal(parsed$iter, 1)
  expect_equal(parsed$elpd, -150.3)
  expect_equal(parsed$structure, "linear/independent/single/basic")
})
