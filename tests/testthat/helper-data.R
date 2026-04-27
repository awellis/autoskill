# Shared test fixtures for autoskill tests

make_test_items <- function() {
  tibble::tibble(
    item_id = paste0("item_", 1:8),
    text = c(
      "Solve: 3x + 5 = 20",
      "Simplify: 2(x + 3) - x",
      "Compute: 1/2 + 1/3",
      "A train goes x km/h for 3 hours, covering 180 km. Find x.",
      "Factor: x^2 - 9",
      "Convert 3/4 to a decimal",
      "Two numbers sum to 20 and differ by 4. Find them.",
      "Compute: 3/8 + 5/8"
    )
  )
}

make_test_taxonomy <- function() {
  tibble::tibble(
    skill_id = c("skill_1", "skill_2", "skill_3"),
    name = c("Linear equations", "Fraction arithmetic", "Equation setup"),
    description = c(
      "Solving and rearranging equations with one unknown",
      "Adding, converting, and computing with fractions and decimals",
      "Translating a problem description into a solvable equation"
    ),
    is_new = c(TRUE, TRUE, TRUE)
  )
}

make_test_assignments <- function() {
  tibble::tibble(
    item_id = c(
      "item_1", "item_2", "item_3", "item_4", "item_4",
      "item_5", "item_6", "item_7", "item_7", "item_8"
    ),
    skill_id = c(
      "skill_1", "skill_1", "skill_2", "skill_1", "skill_3",
      "skill_1", "skill_2", "skill_1", "skill_3", "skill_2"
    ),
    skill_name = c(
      "Linear equations", "Linear equations", "Fraction arithmetic",
      "Linear equations", "Equation setup",
      "Linear equations", "Fraction arithmetic",
      "Linear equations", "Equation setup", "Fraction arithmetic"
    )
  )
}

# Skip helper for tests requiring cmdstan
skip_if_no_cmdstan <- function() {
  testthat::skip_if_not(
    cmdstanr::cmdstan_version() >= "2.26.0",
    "CmdStan not available"
  )
}

# Lightweight fake response_data for cache/parallel tests (no Stan needed)
make_fake_responses <- function(n_students = 10, n_items = 4, seed = 1) {
  withr::with_seed(seed, {
    Y <- matrix(sample(0:1, n_students * n_items, replace = TRUE),
                n_students, n_items)
  })
  rownames(Y) <- paste0("s_", seq_len(n_students))
  colnames(Y) <- paste0("item_", seq_len(n_items))
  response_data(Y)
}

# Minimal fit_result with just the fields the optimizer reads
make_fake_fit_result <- function(config, elpd = -100) {
  est <- matrix(c(elpd, 5), nrow = 1,
                dimnames = list("elpd_loo", c("Estimate", "SE")))
  loo_obj <- structure(list(estimates = est), class = "loo")
  base::structure(
    list(fit = NULL, config = config,
         diagnostics = tibble::tibble(metric = "n_divergences",
                                      value = 0, status = "ok"),
         loo = loo_obj,
         param_summary = tibble::tibble()),
    class = "fit_result"
  )
}

# Helper to build a minimal config for testing blocks and generator
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
