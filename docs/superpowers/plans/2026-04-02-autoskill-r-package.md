# autoskill R Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a state-of-the-art R package for LLM-driven discovery of latent knowledge component structures in student response data, with composable Stan code generation, Bayesian model comparison, and an LLM-in-the-loop refinement cycle.

**Architecture:** Two-phase design: (1) skill proposer (LLM reads items, proposes KCs) and (2) structure optimizer (composable Stan blocks, MCMC fitting, LOO comparison, LLM reflection loop). Stan code is generated programmatically from block configurations using brms's named-list fragment pattern. Every valid block combination (64 total: 2 measurement x 4 structural x 2 population x 2 item x 2 link) is guaranteed to compile.

**Tech Stack:** R, tidyverse, cmdstanr, loo, posterior, ellmer, SBC, bayesplot, priorsense, testthat

---

## File Structure

```
autoskill/
  DESCRIPTION
  NAMESPACE                        # roxygen2-generated
  LICENSE
  LICENSE.md
  .Rbuildignore
  R/
    autoskill-package.R            # package docs, re-exports, global imports
    utils.R                        # collapse_stan_lists(), topological_sort(), helpers
    model-spec.R                   # model_spec S3 class + validation
    loading-structure.R            # loading_structure + edge_prior S3 classes
    response-data.R                # response_data S3 class
    model-config.R                 # model_config = spec + structure + edge_prior
    stan-blocks.R                  # block-specific Stan code fragment generators
    stan-generator.R               # compose fragments into complete Stan programs
    stan-data.R                    # prepare_stan_data()
    stan-compile.R                 # compile_model() with content-hash caching
    simulate-data.R                # simulate_responses(), sbc_generator()
    sbc-check.R                    # check_identifiability(), run_sbc()
    model-fit.R                    # fit_model(), extract_diagnostics(), compute_loo()
    model-compare.R                # compare_models(), flag_problem_items()
    reflection.R                   # format_reflection_prompt(), propose_refinement()
    skill-proposer.R               # refactored existing code
    structure-optimizer.R          # optimize_structure() outer loop
  tests/
    testthat.R
    testthat/
      helper-data.R                # shared test fixtures
      test-model-spec.R
      test-loading-structure.R
      test-response-data.R
      test-model-config.R
      test-utils.R
      test-stan-blocks.R
      test-stan-generator.R
      test-stan-data.R
      test-stan-compile.R
      test-simulate-data.R
      test-sbc-check.R
      test-model-fit.R
      test-model-compare.R
      test-skill-proposer.R
      test-structure-optimizer.R
  examples/
    propose-skills.R               # updated to use package
    optimize-structure.R           # end-to-end demo
  notes/                           # .Rbuildignore'd, existing design docs
```

---

## Task 1: Package Skeleton

**Files:**
- Create: `DESCRIPTION`
- Create: `NAMESPACE`
- Create: `LICENSE`
- Create: `LICENSE.md`
- Create: `.Rbuildignore`
- Create: `R/autoskill-package.R`
- Create: `tests/testthat.R`
- Create: `tests/testthat/helper-data.R`

- [ ] **Step 1: Create DESCRIPTION**

```
Package: autoskill
Title: LLM-Driven Discovery of Latent Knowledge Component Structure
Version: 0.1.0
Authors@R: person("Andrew", "Ellis", role = c("aut", "cre"),
                  email = "andrew.ellis@unibe.ch")
Description: Discovers latent knowledge component (KC) structures in student
    response data using LLM-driven hypothesis generation and Bayesian model
    comparison. Combines cognitive task analysis via large language models with
    multilevel IRT evaluation via Stan.
License: MIT + file LICENSE
Encoding: UTF-8
Roxygen: list(markdown = TRUE)
RoxygenNote: 7.3.2
Depends:
    R (>= 4.1.0)
Imports:
    tibble,
    dplyr,
    tidyr,
    purrr,
    stringr,
    glue,
    rlang (>= 1.0.0),
    cli,
    ellmer,
    cmdstanr,
    posterior,
    loo,
    jsonlite
Suggests:
    testthat (>= 3.0.0),
    SBC,
    bayesplot,
    priorsense,
    brms,
    withr,
    knitr,
    rmarkdown
Config/testthat/edition: 3
Additional_repositories: https://stan-dev.r-universe.dev/
```

- [ ] **Step 2: Create LICENSE and LICENSE.md**

`LICENSE`:
```
YEAR: 2026
COPYRIGHT HOLDER: Andrew Ellis
```

`LICENSE.md`:
```
# MIT License

Copyright (c) 2026 Andrew Ellis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Create .Rbuildignore**

```
^\.claude$
^\.remember$
^\.env$
^\.Renviron$
^notes$
^examples$
^LICENSE\.md$
^\.github$
^docs$
^_pkgdown\.yml$
```

- [ ] **Step 4: Create R/autoskill-package.R**

```r
#' @keywords internal
"_PACKAGE"

#' @importFrom rlang %||% .data := abort warn inform
#' @importFrom cli cli_abort cli_warn cli_inform
#' @importFrom glue glue
#' @importFrom stats setNames
NULL
```

- [ ] **Step 5: Create tests/testthat.R**

```r
library(testthat)
library(autoskill)

test_check("autoskill")
```

- [ ] **Step 6: Create tests/testthat/helper-data.R**

This file provides shared test fixtures used across multiple test files.

```r
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
```

- [ ] **Step 7: Create empty NAMESPACE and generate with roxygen**

Create an empty `NAMESPACE` file (will be overwritten by roxygen2):
```
# Generated by roxygen2: do not edit by hand
```

- [ ] **Step 8: Verify package loads**

Run: `Rscript -e "devtools::load_all(); cat('Package loaded successfully\n')"`
Expected: "Package loaded successfully"

- [ ] **Step 9: Commit**

```bash
git add DESCRIPTION NAMESPACE LICENSE LICENSE.md .Rbuildignore R/autoskill-package.R tests/testthat.R tests/testthat/helper-data.R
git commit -m "Init R package skeleton with DESCRIPTION, test harness, and shared fixtures"
```

---

## Task 2: Utility Functions

**Files:**
- Create: `R/utils.R`
- Create: `tests/testthat/test-utils.R`

- [ ] **Step 1: Write the failing tests**

`tests/testthat/test-utils.R`:

```r
test_that("collapse_stan_lists merges same-named elements", {
  a <- list(data = "int N;\n", parameters = "real mu;\n", model = "")
  b <- list(data = "int K;\n", parameters = "real sigma;\n", model = "mu ~ normal(0, 1);\n")

  result <- collapse_stan_lists(a, b)

  expect_equal(result$data, "int N;\nint K;\n")
  expect_equal(result$parameters, "real mu;\nreal sigma;\n")
  expect_equal(result$model, "mu ~ normal(0, 1);\n")
})

test_that("collapse_stan_lists handles missing keys", {
  a <- list(data = "int N;\n")
  b <- list(parameters = "real mu;\n")

  result <- collapse_stan_lists(a, b)

  expect_equal(result$data, "int N;\n")
  expect_equal(result$parameters, "real mu;\n")
})

test_that("topological_sort orders DAG correctly", {
  # A -> B -> C
  from <- c("A", "B")
  to <- c("B", "C")
  nodes <- c("A", "B", "C")

  result <- topological_sort(from, to, nodes)

  # A must come before B, B before C
  expect_true(which(result == "A") < which(result == "B"))
  expect_true(which(result == "B") < which(result == "C"))
})

test_that("topological_sort handles diamond DAG", {
  # A -> B, A -> C, B -> D, C -> D
  from <- c("A", "A", "B", "C")
  to <- c("B", "C", "D", "D")
  nodes <- c("A", "B", "C", "D")

  result <- topological_sort(from, to, nodes)

  expect_true(which(result == "A") < which(result == "B"))
  expect_true(which(result == "A") < which(result == "C"))
  expect_true(which(result == "B") < which(result == "D"))
  expect_true(which(result == "C") < which(result == "D"))
})

test_that("validate_dag rejects cycles", {
  from <- c("A", "B", "C")
  to <- c("B", "C", "A")
  nodes <- c("A", "B", "C")

  expect_false(validate_dag(from, to, nodes))
})

test_that("validate_dag accepts valid DAGs", {
  from <- c("A", "A")
  to <- c("B", "C")
  nodes <- c("A", "B", "C")

  expect_true(validate_dag(from, to, nodes))
})

test_that("validate_dag accepts empty DAG", {
  expect_true(validate_dag(character(0), character(0), c("A", "B")))
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-utils.R')"`
Expected: All tests FAIL (functions not found)

- [ ] **Step 3: Implement R/utils.R**

```r
#' Merge named lists of Stan code fragments
#'
#' Concatenates same-named elements across multiple fragment lists.
#' This is the core composition mechanism for the block-based Stan
#' code generator (analogous to brms's internal collapse approach).
#'
#' @param ... Named lists of Stan code fragments.
#' @return A named list with all unique keys, values concatenated.
#' @noRd
collapse_stan_lists <- function(...) {
  ls <- list(...)
  all_keys <- unique(unlist(lapply(ls, names)))
  out <- lapply(all_keys, function(key) {
    parts <- vapply(ls, function(x) x[[key]] %||% "", character(1))
    paste0(parts, collapse = "")
  })
  setNames(out, all_keys)
}


#' Topological sort of a DAG
#'
#' Returns nodes in topological order (parents before children).
#' Uses Kahn's algorithm.
#'
#' @param from Character vector of parent node IDs.
#' @param to Character vector of child node IDs (same length as `from`).
#' @param nodes Character vector of all node IDs.
#' @return Character vector of nodes in topological order.
#' @noRd
topological_sort <- function(from, to, nodes) {
  n <- length(nodes)
  if (length(from) == 0L) return(nodes)

  # Build adjacency list and in-degree count
  children <- setNames(vector("list", n), nodes)
  in_degree <- setNames(integer(n), nodes)

  for (i in seq_along(from)) {
    children[[from[i]]] <- c(children[[from[i]]], to[i])
    in_degree[[to[i]]] <- in_degree[[to[i]]] + 1L
  }

  # Kahn's algorithm
  queue <- nodes[in_degree == 0L]
  result <- character(0)

  while (length(queue) > 0L) {
    node <- queue[1]
    queue <- queue[-1]
    result <- c(result, node)

    for (child in children[[node]]) {
      in_degree[[child]] <- in_degree[[child]] - 1L
      if (in_degree[[child]] == 0L) {
        queue <- c(queue, child)
      }
    }
  }

  if (length(result) != n) {
    cli_abort("DAG contains a cycle: topological sort failed.")
  }

  result
}


#' Validate that edges form a DAG (no cycles)
#'
#' @param from Character vector of parent node IDs.
#' @param to Character vector of child node IDs.
#' @param nodes Character vector of all node IDs.
#' @return `TRUE` if acyclic, `FALSE` if cyclic.
#' @noRd
validate_dag <- function(from, to, nodes) {
  if (length(from) == 0L) return(TRUE)
  tryCatch(
    {
      topological_sort(from, to, nodes)
      TRUE
    },
    error = function(e) FALSE
  )
}


#' Create an empty Stan fragment list
#'
#' Returns a named list with all 7 Stan block keys set to empty strings.
#' Block generators should start from this and fill in their pieces.
#'
#' @return Named list with keys: functions, data, transformed_data,
#'   parameters, transformed_parameters, model, generated_quantities.
#' @noRd
empty_stan_fragment <- function() {
  list(
    functions = "",
    data = "",
    transformed_data = "",
    parameters = "",
    transformed_parameters = "",
    model = "",
    generated_quantities = ""
  )
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-utils.R')"`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add R/utils.R tests/testthat/test-utils.R
git commit -m "Add utility functions: collapse_stan_lists, topological_sort, validate_dag"
```

---

## Task 3: model_spec S3 Class

**Files:**
- Create: `R/model-spec.R`
- Create: `tests/testthat/test-model-spec.R`

- [ ] **Step 1: Write the failing tests**

`tests/testthat/test-model-spec.R`:

```r
test_that("model_spec creates valid object with defaults", {
  spec <- model_spec()

  expect_s3_class(spec, "model_spec")
  expect_equal(spec$measurement, "linear")
  expect_equal(spec$structural, "independent")
  expect_equal(spec$population, "single")
  expect_equal(spec$item, "basic")
  expect_equal(spec$link, "logit")
})

test_that("model_spec accepts all valid block options", {
  spec <- model_spec(
    measurement = "interaction",
    structural = "dag",
    population = "grouped",
    item = "slip_guess",
    link = "probit"
  )

  expect_s3_class(spec, "model_spec")
  expect_equal(spec$structural, "dag")
})

test_that("model_spec rejects invalid block options", {
  expect_error(model_spec(measurement = "banana"), "measurement")
  expect_error(model_spec(structural = "nope"), "structural")
  expect_error(model_spec(population = "bad"), "population")
  expect_error(model_spec(item = "wrong"), "item")
  expect_error(model_spec(link = "identity"), "link")
})

test_that("is_sem_mode detects SEM configurations", {
  expect_false(is_sem_mode(model_spec()))
  expect_false(is_sem_mode(model_spec(structural = "correlated")))
  expect_true(is_sem_mode(model_spec(structural = "dag")))
  expect_true(is_sem_mode(model_spec(structural = "hierarchical")))
})

test_that("is_fa_mode is the complement of is_sem_mode", {
  expect_true(is_fa_mode(model_spec()))
  expect_false(is_fa_mode(model_spec(structural = "dag")))
})

test_that("print.model_spec produces output", {
  spec <- model_spec()
  expect_output(print(spec), "model_spec")
})

test_that("all_valid_specs enumerates all 64 combinations", {
  specs <- all_valid_specs()
  expect_equal(nrow(specs), 64L)
  expect_named(specs, c("measurement", "structural", "population", "item", "link"))
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-model-spec.R')"`
Expected: All tests FAIL

- [ ] **Step 3: Implement R/model-spec.R**

```r
# Valid options for each block
MEASUREMENT_OPTIONS <- c("linear", "interaction")
STRUCTURAL_OPTIONS <- c("independent", "correlated", "dag", "hierarchical")
POPULATION_OPTIONS <- c("single", "grouped")
ITEM_OPTIONS <- c("basic", "slip_guess")
LINK_OPTIONS <- c("logit", "probit")


#' Create a model specification
#'
#' A model specification defines which composable block is selected for each
#' component of the IRT model. Every valid combination produces compilable
#' Stan code.
#'
#' @param measurement One of "linear" or "interaction".
#' @param structural One of "independent", "correlated", "dag", "hierarchical".
#' @param population One of "single" or "grouped".
#' @param item One of "basic" or "slip_guess".
#' @param link One of "logit" or "probit".
#' @return An S3 object of class `model_spec`.
#' @export
model_spec <- function(measurement = "linear",
                       structural = "independent",
                       population = "single",
                       item = "basic",
                       link = "logit") {
  spec <- structure(
    list(
      measurement = measurement,
      structural = structural,
      population = population,
      item = item,
      link = link
    ),
    class = "model_spec"
  )
  validate_model_spec(spec)
  spec
}


#' Validate a model_spec object
#' @noRd
validate_model_spec <- function(x) {
  if (!x$measurement %in% MEASUREMENT_OPTIONS) {
    cli_abort("{.arg measurement} must be one of {.or {.val {MEASUREMENT_OPTIONS}}}, not {.val {x$measurement}}.")
  }
  if (!x$structural %in% STRUCTURAL_OPTIONS) {
    cli_abort("{.arg structural} must be one of {.or {.val {STRUCTURAL_OPTIONS}}}, not {.val {x$structural}}.")
  }
  if (!x$population %in% POPULATION_OPTIONS) {
    cli_abort("{.arg population} must be one of {.or {.val {POPULATION_OPTIONS}}}, not {.val {x$population}}.")
  }
  if (!x$item %in% ITEM_OPTIONS) {
    cli_abort("{.arg item} must be one of {.or {.val {ITEM_OPTIONS}}}, not {.val {x$item}}.")
  }
  if (!x$link %in% LINK_OPTIONS) {
    cli_abort("{.arg link} must be one of {.or {.val {LINK_OPTIONS}}}, not {.val {x$link}}.")
  }
  invisible(x)
}


#' @export
print.model_spec <- function(x, ...) {
  cli_inform(c(
    "{.cls model_spec}",
    "*" = "measurement: {.val {x$measurement}}",
    "*" = "structural:  {.val {x$structural}}",
    "*" = "population:  {.val {x$population}}",
    "*" = "item:        {.val {x$item}}",
    "*" = "link:        {.val {x$link}}"
  ))
  invisible(x)
}


#' Check if a model spec uses SEM mode
#'
#' SEM mode means the structural block is "dag" or "hierarchical",
#' implying directed dependencies among skills.
#'
#' @param spec A `model_spec` object.
#' @return Logical.
#' @export
is_sem_mode <- function(spec) {
  spec$structural %in% c("dag", "hierarchical")
}


#' Check if a model spec uses factor analysis mode
#'
#' @param spec A `model_spec` object.
#' @return Logical.
#' @export
is_fa_mode <- function(spec) {
  !is_sem_mode(spec)
}


#' Enumerate all valid model spec combinations
#'
#' Returns a tibble with one row per valid combination (64 total).
#'
#' @return A tibble with columns: measurement, structural, population, item, link.
#' @export
all_valid_specs <- function() {
  tidyr::crossing(
    measurement = MEASUREMENT_OPTIONS,
    structural = STRUCTURAL_OPTIONS,
    population = POPULATION_OPTIONS,
    item = ITEM_OPTIONS,
    link = LINK_OPTIONS
  )
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-model-spec.R')"`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add R/model-spec.R tests/testthat/test-model-spec.R
git commit -m "Add model_spec S3 class with validation, query helpers, and enumeration"
```

---

## Task 4: loading_structure and edge_prior S3 Classes

**Files:**
- Create: `R/loading-structure.R`
- Create: `tests/testthat/test-loading-structure.R`

- [ ] **Step 1: Write the failing tests**

`tests/testthat/test-loading-structure.R`:

```r
test_that("loading_structure creates valid object", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()

  ls <- loading_structure(taxonomy, assignments, items)

  expect_s3_class(ls, "loading_structure")
  expect_equal(nrow(ls$taxonomy), 3)
  expect_equal(nrow(ls$items), 8)
  expect_equal(dim(ls$lambda_mask), c(8, 3))
})

test_that("lambda_mask has correct entries", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()

  ls <- loading_structure(taxonomy, assignments, items)

  # item_1 loads on skill_1 only

  expect_true(ls$lambda_mask["item_1", "skill_1"])
  expect_false(ls$lambda_mask["item_1", "skill_2"])
  expect_false(ls$lambda_mask["item_1", "skill_3"])

  # item_4 cross-loads on skill_1 and skill_3
  expect_true(ls$lambda_mask["item_4", "skill_1"])
  expect_false(ls$lambda_mask["item_4", "skill_2"])
  expect_true(ls$lambda_mask["item_4", "skill_3"])
})

test_that("loading_structure rejects orphan items", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  # Only assign skills to items 1-7, leave item_8 orphaned
  assignments <- make_test_assignments() |>
    dplyr::filter(item_id != "item_8")

  expect_error(loading_structure(taxonomy, assignments, items), "item_8")
})

test_that("loading_structure rejects skills with fewer than 2 items", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  # Remove all but one assignment for skill_3
  assignments <- make_test_assignments() |>
    dplyr::filter(!(skill_id == "skill_3" & item_id == "item_7"))

  expect_error(loading_structure(taxonomy, assignments, items), "skill_3")
})

test_that("n_loadings counts nonzero entries", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()

  ls <- loading_structure(taxonomy, assignments, items)

  # 10 assignments total (items 4 and 7 each have 2)
  expect_equal(ls$n_loadings, 10L)
})

test_that("print.loading_structure produces output", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()

  ls <- loading_structure(taxonomy, assignments, items)

  expect_output(print(ls), "loading_structure")
})

test_that("edge_prior creates valid object", {
  ep <- edge_prior(
    from = c("skill_1", "skill_1"),
    to = c("skill_2", "skill_3"),
    prob = c(0.8, 0.6)
  )

  expect_s3_class(ep, "edge_prior")
  expect_equal(nrow(ep$edges), 2)
})

test_that("edge_prior rejects probabilities outside [0, 1]", {
  expect_error(
    edge_prior(from = "A", to = "B", prob = 1.5),
    "prob"
  )
})

test_that("edge_prior rejects self-loops", {
  expect_error(
    edge_prior(from = "A", to = "A", prob = 0.5),
    "self-loop"
  )
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-loading-structure.R')"`
Expected: All tests FAIL

- [ ] **Step 3: Implement R/loading-structure.R**

```r
#' Create a loading structure
#'
#' Wraps taxonomy, assignments, and items from the skill proposer into
#' a validated structure with a precomputed lambda mask (logical matrix
#' indicating which items load on which skills).
#'
#' @param taxonomy A tibble with columns `skill_id`, `name`, `description`.
#' @param assignments A tibble with columns `item_id`, `skill_id`, `skill_name`.
#' @param items A tibble with columns `item_id`, `text`.
#' @return An S3 object of class `loading_structure`.
#' @export
loading_structure <- function(taxonomy, assignments, items) {
  mask <- build_lambda_mask(assignments, items$item_id, taxonomy$skill_id)
  n_loadings <- sum(mask)

  x <- structure(
    list(
      taxonomy = taxonomy,
      assignments = assignments,
      items = items,
      lambda_mask = mask,
      n_loadings = as.integer(n_loadings)
    ),
    class = "loading_structure"
  )
  validate_loading_structure(x)
  x
}


#' Build the lambda mask (logical I x K matrix)
#'
#' @param assignments Tibble with `item_id` and `skill_id` columns.
#' @param item_ids Character vector of item IDs (defines rows).
#' @param skill_ids Character vector of skill IDs (defines columns).
#' @return A logical matrix with rownames = item_ids, colnames = skill_ids.
#' @noRd
build_lambda_mask <- function(assignments, item_ids, skill_ids) {
  mask <- matrix(
    FALSE,
    nrow = length(item_ids),
    ncol = length(skill_ids),
    dimnames = list(item_ids, skill_ids)
  )
  for (i in seq_len(nrow(assignments))) {
    mask[assignments$item_id[i], assignments$skill_id[i]] <- TRUE
  }
  mask
}


#' @noRd
validate_loading_structure <- function(x) {
  mask <- x$lambda_mask

  # Every item must load on at least one skill
  orphans <- rownames(mask)[rowSums(mask) == 0L]
  if (length(orphans) > 0) {
    cli_abort("Items have no skill assignments: {.val {orphans}}.")
  }

  # Every skill must have at least 2 items (for identifiability)
  thin_skills <- colnames(mask)[colSums(mask) < 2L]
  if (length(thin_skills) > 0) {
    names <- x$taxonomy$name[x$taxonomy$skill_id %in% thin_skills]
    cli_abort("Skills have fewer than 2 items (not identifiable): {.val {thin_skills}} ({names}).")
  }

  invisible(x)
}


#' @export
print.loading_structure <- function(x, ...) {
  n_items <- nrow(x$lambda_mask)
  n_skills <- ncol(x$lambda_mask)
  cli_inform(c(
    "{.cls loading_structure}: {n_items} items, {n_skills} skills, {x$n_loadings} loadings",
    "*" = "Skills: {paste(x$taxonomy$name, collapse = ', ')}"
  ))
  invisible(x)
}


#' Create an edge prior for SEM mode
#'
#' Specifies prior probabilities for directed edges among skills.
#' Each edge has a probability of existing. Used to break Markov
#' equivalence ambiguity with domain knowledge.
#'
#' @param from Character vector of parent skill IDs.
#' @param to Character vector of child skill IDs.
#' @param prob Numeric vector of prior probabilities in [0, 1].
#' @return An S3 object of class `edge_prior`.
#' @export
edge_prior <- function(from, to, prob) {
  if (length(from) != length(to) || length(from) != length(prob)) {
    cli_abort("{.arg from}, {.arg to}, and {.arg prob} must have the same length.")
  }
  if (any(prob < 0 | prob > 1)) {
    cli_abort("All {.arg prob} values must be in [0, 1].")
  }
  if (any(from == to)) {
    cli_abort("Edge prior contains a self-loop (from == to).")
  }

  edges <- tibble::tibble(from = from, to = to, prob = prob)

  structure(
    list(edges = edges),
    class = "edge_prior"
  )
}


#' @export
print.edge_prior <- function(x, ...) {
  cli_inform("{.cls edge_prior}: {nrow(x$edges)} edges")
  print(x$edges)
  invisible(x)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-loading-structure.R')"`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add R/loading-structure.R tests/testthat/test-loading-structure.R
git commit -m "Add loading_structure and edge_prior S3 classes with validation"
```

---

## Task 5: response_data S3 Class

**Files:**
- Create: `R/response-data.R`
- Create: `tests/testthat/test-response-data.R`

- [ ] **Step 1: Write the failing tests**

`tests/testthat/test-response-data.R`:

```r
test_that("response_data accepts a wide matrix", {
  Y <- matrix(
    sample(0:1, 100 * 8, replace = TRUE),
    nrow = 100, ncol = 8,
    dimnames = list(paste0("s_", 1:100), paste0("item_", 1:8))
  )

  rd <- response_data(Y)

  expect_s3_class(rd, "response_data")
  expect_equal(rd$n_students, 100L)
  expect_equal(rd$n_items, 8L)
  expect_equal(dim(rd$Y), c(100, 8))
})

test_that("response_data accepts a long tibble", {
  long <- tidyr::crossing(
    student_id = paste0("s_", 1:50),
    item_id = paste0("item_", 1:6)
  ) |>
    dplyr::mutate(correct = sample(0:1, dplyr::n(), replace = TRUE))

  rd <- response_data(long)

  expect_s3_class(rd, "response_data")
  expect_equal(rd$n_students, 50L)
  expect_equal(rd$n_items, 6L)
})

test_that("response_data rejects non-binary responses", {
  Y <- matrix(c(0, 1, 2, 1), nrow = 2, ncol = 2)
  expect_error(response_data(Y), "binary")
})

test_that("response_data handles missing values", {
  Y <- matrix(c(0, 1, NA, 1, 0, 1), nrow = 3, ncol = 2)
  rd <- response_data(Y)
  expect_true(any(is.na(rd$Y)))
})

test_that("print.response_data produces output", {
  Y <- matrix(sample(0:1, 40, replace = TRUE), nrow = 10, ncol = 4)
  rd <- response_data(Y)
  expect_output(print(rd), "response_data")
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-response-data.R')"`
Expected: All tests FAIL

- [ ] **Step 3: Implement R/response-data.R**

```r
#' Create a response data object
#'
#' Accepts either a wide matrix (students x items, binary 0/1) or a long
#' tibble with columns `student_id`, `item_id`, `correct`.
#'
#' @param responses A matrix or tibble of student responses.
#' @return An S3 object of class `response_data`.
#' @export
response_data <- function(responses) {
  if (is.data.frame(responses)) {
    rd <- response_data_from_long(responses)
  } else if (is.matrix(responses)) {
    rd <- response_data_from_wide(responses)
  } else {
    cli_abort("{.arg responses} must be a matrix or data frame.")
  }
  validate_response_data(rd)
  rd
}


#' @noRd
response_data_from_wide <- function(Y) {
  if (is.null(rownames(Y))) {
    rownames(Y) <- paste0("student_", seq_len(nrow(Y)))
  }
  if (is.null(colnames(Y))) {
    colnames(Y) <- paste0("item_", seq_len(ncol(Y)))
  }
  storage.mode(Y) <- "integer"

  structure(
    list(
      Y = Y,
      item_ids = colnames(Y),
      student_ids = rownames(Y),
      n_students = nrow(Y),
      n_items = ncol(Y)
    ),
    class = "response_data"
  )
}


#' @noRd
response_data_from_long <- function(df) {
  required <- c("student_id", "item_id", "correct")
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) {
    cli_abort("Long format requires columns: {.val {missing}}.")
  }

  wide <- df |>
    tidyr::pivot_wider(
      id_cols = "student_id",
      names_from = "item_id",
      values_from = "correct"
    )

  student_ids <- wide$student_id
  Y <- as.matrix(wide[, -1])
  rownames(Y) <- student_ids
  storage.mode(Y) <- "integer"

  structure(
    list(
      Y = Y,
      item_ids = colnames(Y),
      student_ids = student_ids,
      n_students = nrow(Y),
      n_items = ncol(Y)
    ),
    class = "response_data"
  )
}


#' @noRd
validate_response_data <- function(x) {
  vals <- x$Y[!is.na(x$Y)]
  if (!all(vals %in% c(0L, 1L))) {
    cli_abort("Responses must be binary (0 or 1).")
  }
  invisible(x)
}


#' @export
print.response_data <- function(x, ...) {
  n_missing <- sum(is.na(x$Y))
  cli_inform(c(
    "{.cls response_data}: {x$n_students} students, {x$n_items} items",
    if (n_missing > 0) c("!" = "{n_missing} missing values")
  ))
  invisible(x)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-response-data.R')"`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add R/response-data.R tests/testthat/test-response-data.R
git commit -m "Add response_data S3 class with wide/long input support"
```

---

## Task 6: model_config S3 Class

**Files:**
- Create: `R/model-config.R`
- Create: `tests/testthat/test-model-config.R`

- [ ] **Step 1: Write the failing tests**

`tests/testthat/test-model-config.R`:

```r
test_that("model_config bundles spec and structure", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()

  spec <- model_spec()
  structure <- loading_structure(taxonomy, assignments, items)
  config <- model_config(spec, structure)

  expect_s3_class(config, "model_config")
  expect_s3_class(config$spec, "model_spec")
  expect_s3_class(config$structure, "loading_structure")
  expect_null(config$edge_prior)
})

test_that("model_config requires edge_prior for dag mode", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()

  spec <- model_spec(structural = "dag")
  structure <- loading_structure(taxonomy, assignments, items)

  expect_error(model_config(spec, structure), "edge_prior")
})

test_that("model_config accepts edge_prior for dag mode", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()

  spec <- model_spec(structural = "dag")
  structure <- loading_structure(taxonomy, assignments, items)
  ep <- edge_prior(
    from = c("skill_1", "skill_1"),
    to = c("skill_2", "skill_3"),
    prob = c(0.8, 0.6)
  )

  config <- model_config(spec, structure, edge_prior = ep)
  expect_s3_class(config, "model_config")
  expect_s3_class(config$edge_prior, "edge_prior")
})

test_that("config_hash is stable", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()

  spec <- model_spec()
  structure <- loading_structure(taxonomy, assignments, items)
  config <- model_config(spec, structure)

  h1 <- config_hash(config)
  h2 <- config_hash(config)
  expect_equal(h1, h2)
})

test_that("config_hash differs for different specs", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()
  structure <- loading_structure(taxonomy, assignments, items)

  c1 <- model_config(model_spec(structural = "independent"), structure)
  c2 <- model_config(model_spec(structural = "correlated"), structure)

  expect_false(config_hash(c1) == config_hash(c2))
})

test_that("print.model_config produces output", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()

  config <- model_config(model_spec(), loading_structure(taxonomy, assignments, items))
  expect_output(print(config), "model_config")
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-model-config.R')"`
Expected: All tests FAIL

- [ ] **Step 3: Implement R/model-config.R**

```r
#' Create a model configuration
#'
#' Bundles a model specification (block choices) with a loading structure
#' (which items load on which skills). This is the complete hypothesis
#' that drives Stan code generation and fitting.
#'
#' @param spec A `model_spec` object.
#' @param structure A `loading_structure` object.
#' @param edge_prior An `edge_prior` object (required for `structural = "dag"`).
#' @return An S3 object of class `model_config`.
#' @export
model_config <- function(spec, structure, edge_prior = NULL) {
  config <- structure(
    list(
      spec = spec,
      structure = structure,
      edge_prior = edge_prior
    ),
    class = "model_config"
  )
  validate_model_config(config)
  config
}


#' @noRd
validate_model_config <- function(x) {
  if (!inherits(x$spec, "model_spec")) {
    cli_abort("{.arg spec} must be a {.cls model_spec} object.")
  }
  if (!inherits(x$structure, "loading_structure")) {
    cli_abort("{.arg structure} must be a {.cls loading_structure} object.")
  }

  # DAG mode requires an edge_prior

  if (x$spec$structural == "dag" && is.null(x$edge_prior)) {
    cli_abort("Structural model {.val dag} requires an {.arg edge_prior}.")
  }
  if (!is.null(x$edge_prior) && !inherits(x$edge_prior, "edge_prior")) {
    cli_abort("{.arg edge_prior} must be an {.cls edge_prior} object.")
  }

  invisible(x)
}


#' Compute a content hash for a model configuration
#'
#' Used for caching compiled Stan models. Two configs with the same hash
#' produce identical Stan code.
#'
#' @param config A `model_config` object.
#' @return A character string (hex digest).
#' @export
config_hash <- function(config) {
  # Hash the parts that affect Stan code generation
  content <- list(
    spec = unclass(config$spec),
    lambda_mask = config$structure$lambda_mask,
    edge_prior = if (!is.null(config$edge_prior)) config$edge_prior$edges else NULL
  )
  rlang::hash(content)
}


#' @export
print.model_config <- function(x, ...) {
  n_items <- nrow(x$structure$lambda_mask)
  n_skills <- ncol(x$structure$lambda_mask)
  mode <- if (is_sem_mode(x$spec)) "SEM" else "Factor analysis"

  cli_inform(c(
    "{.cls model_config} ({mode}): {n_items} items, {n_skills} skills",
    "*" = "measurement: {.val {x$spec$measurement}}",
    "*" = "structural:  {.val {x$spec$structural}}",
    "*" = "population:  {.val {x$spec$population}}",
    "*" = "item:        {.val {x$spec$item}}",
    "*" = "link:        {.val {x$spec$link}}"
  ))
  invisible(x)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-model-config.R')"`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add R/model-config.R tests/testthat/test-model-config.R
git commit -m "Add model_config S3 class with cross-validation and content hashing"
```

---

## Task 7: Stan Block Generators

This is the core technical module. Each function generates Stan code fragments for one block option. This is the largest task and should be implemented incrementally: measurement blocks first, then structural, then population and item.

**Files:**
- Create: `R/stan-blocks.R`
- Create: `tests/testthat/test-stan-blocks.R`

- [ ] **Step 1: Write the failing tests**

`tests/testthat/test-stan-blocks.R`:

```r
# Helper to build a minimal config for testing blocks
make_test_config <- function(measurement = "linear",
                             structural = "independent",
                             population = "single",
                             item = "basic",
                             link = "logit") {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()
  structure <- loading_structure(taxonomy, assignments, items)
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

  model_config(spec, structure, edge_prior = ep)
}


# -- Fragment structure tests --

test_that("all block generators return complete fragment lists", {
  config <- make_test_config()
  expected_keys <- c(
    "functions", "data", "transformed_data", "parameters",
    "transformed_parameters", "model", "generated_quantities"
  )

  # Measurement
  frag <- stan_measurement(config)
  expect_named(frag, expected_keys, ignore.order = TRUE)

  # Structural
  frag <- stan_structural(config)
  expect_named(frag, expected_keys, ignore.order = TRUE)

  # Population
  frag <- stan_population(config)
  expect_named(frag, expected_keys, ignore.order = TRUE)

  # Item
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

  # Single population needs no group data
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-stan-blocks.R')"`
Expected: All tests FAIL

- [ ] **Step 3: Implement R/stan-blocks.R (measurement blocks)**

```r
# --- Dispatchers ---

#' Generate Stan fragments for the measurement block
#' @noRd
stan_measurement <- function(config) {
  switch(config$spec$measurement,
    linear = stan_measurement_linear(config),
    interaction = stan_measurement_interaction(config)
  )
}

#' Generate Stan fragments for the structural block
#' @noRd
stan_structural <- function(config) {
  switch(config$spec$structural,
    independent = stan_structural_independent(config),
    correlated = stan_structural_correlated(config),
    dag = stan_structural_dag(config),
    hierarchical = stan_structural_hierarchical(config)
  )
}

#' Generate Stan fragments for the population block
#' @noRd
stan_population <- function(config) {
  switch(config$spec$population,
    single = stan_population_single(config),
    grouped = stan_population_grouped(config)
  )
}

#' Generate Stan fragments for the item block
#' @noRd
stan_item <- function(config) {
  switch(config$spec$item,
    basic = stan_item_basic(config),
    slip_guess = stan_item_slip_guess(config)
  )
}


# --- Measurement: linear ---

#' @noRd
stan_measurement_linear <- function(config) {
  link_inv <- if (config$spec$link == "logit") "inv_logit" else "Phi"

  frag <- empty_stan_fragment()

  # compute_prob() function: composable hook for slip-guess wrapping
  frag$functions <- glue("
  real compute_prob(int i, row_vector theta_j, matrix Lambda, vector alpha) {{
    return {link_inv}(alpha[i] + Lambda[i] * theta_j');
  }}
")

  frag$data <- glue("
  int<lower=1> N_obs;
  int<lower=1> I;
  int<lower=1> J;
  int<lower=1> K;
  int<lower=1> N_loadings;
  array[N_loadings] int<lower=1,upper=I> loading_item;
  array[N_loadings] int<lower=1,upper=K> loading_skill;
  array[N_obs] int<lower=1,upper=I> ii;
  array[N_obs] int<lower=1,upper=J> jj;
  array[N_obs] int<lower=0,upper=1> y;
")

  frag$parameters <- glue("
  vector<lower=0>[N_loadings] lambda_free;
  vector[I] alpha;
")

  frag$transformed_parameters <- glue("
  matrix[I, K] Lambda = rep_matrix(0, I, K);
  for (n in 1:N_loadings)
    Lambda[loading_item[n], loading_skill[n]] = lambda_free[n];
")

  frag$model <- glue("
  // Measurement priors
  lambda_free ~ normal(0, 1);
  alpha ~ normal(0, 2);

  // Likelihood
  for (n in 1:N_obs)
    y[n] ~ bernoulli(compute_prob(ii[n], theta[jj[n]], Lambda, alpha));
")

  frag$generated_quantities <- glue("
  vector[N_obs] log_lik;
  for (n in 1:N_obs)
    log_lik[n] = bernoulli_lpmf(y[n] | compute_prob(ii[n], theta[jj[n]], Lambda, alpha));
")

  frag
}


# --- Measurement: interaction ---

#' @noRd
stan_measurement_interaction <- function(config) {
  link_inv <- if (config$spec$link == "logit") "inv_logit" else "Phi"
  mask <- config$structure$lambda_mask

  # Find items that cross-load on 2+ skills: these get interaction terms
  cross_items <- which(rowSums(mask) >= 2)
  n_interactions <- 0L
  for (ci in cross_items) {
    skills <- which(mask[ci, ])
    n_interactions <- n_interactions + choose(length(skills), 2)
  }

  frag <- stan_measurement_linear(config)

  if (n_interactions > 0) {
    frag$data <- paste0(frag$data, glue("
  int<lower=0> N_interactions;
  array[N_interactions] int<lower=1,upper=I> interact_item;
  array[N_interactions] int<lower=1,upper=K> interact_skill1;
  array[N_interactions] int<lower=1,upper=K> interact_skill2;
"))

    frag$parameters <- paste0(frag$parameters, glue("
  vector[N_interactions] gamma;
"))

    frag$transformed_parameters <- paste0(frag$transformed_parameters, glue("
  // Interaction terms stored per item
  vector[I] interact_contrib = rep_vector(0, I);
  // (populated per-observation in model block)
"))

    # Override model and gen quant to include interaction
    frag$model <- glue("
  // Measurement priors
  lambda_free ~ normal(0, 1);
  alpha ~ normal(0, 2);
  gamma ~ normal(0, 0.5);

  // Likelihood with interactions
  for (n in 1:N_obs) {{
    real eta = alpha[ii[n]] + Lambda[ii[n]] * theta[jj[n]]';
    for (m in 1:N_interactions) {{
      if (interact_item[m] == ii[n])
        eta += gamma[m] * theta[jj[n], interact_skill1[m]] * theta[jj[n], interact_skill2[m]];
    }}
    y[n] ~ bernoulli({link_inv}(eta));
  }}
")

    frag$generated_quantities <- glue("
  vector[N_obs] log_lik;
  for (n in 1:N_obs) {{
    real eta = alpha[ii[n]] + Lambda[ii[n]] * theta[jj[n]]';
    for (m in 1:N_interactions) {{
      if (interact_item[m] == ii[n])
        eta += gamma[m] * theta[jj[n], interact_skill1[m]] * theta[jj[n], interact_skill2[m]];
    }}
    log_lik[n] = bernoulli_lpmf(y[n] | {link_inv}(eta));
  }}
")
  }

  frag
}


# --- Structural: independent ---

#' @noRd
stan_structural_independent <- function(config) {
  K <- ncol(config$structure$lambda_mask)
  frag <- empty_stan_fragment()

  frag$parameters <- glue("
  matrix[J, {K}] theta;
")

  frag$model <- glue("
  // Structural: independent standard normal
  to_vector(theta) ~ std_normal();
")

  frag
}


# --- Structural: correlated (non-centered parameterization) ---

#' @noRd
stan_structural_correlated <- function(config) {
  K <- ncol(config$structure$lambda_mask)
  frag <- empty_stan_fragment()

  frag$parameters <- glue("
  cholesky_factor_corr[{K}] L_Omega;
  vector<lower=0>[{K}] sigma_theta;
  matrix[{K}, J] z_theta;
")

  frag$transformed_parameters <- glue("
  matrix[J, {K}] theta;
  {{
    matrix[{K}, {K}] L_Sigma = diag_pre_multiply(sigma_theta, L_Omega);
    theta = (L_Sigma * z_theta)';
  }}
")

  frag$model <- glue("
  // Structural: correlated (NCP)
  L_Omega ~ lkj_corr_cholesky(2.0);
  sigma_theta ~ normal(0, 1);
  to_vector(z_theta) ~ std_normal();
")

  frag
}


# --- Structural: DAG ---

#' @noRd
stan_structural_dag <- function(config) {
  K <- ncol(config$structure$lambda_mask)
  ep <- config$edge_prior
  n_edges <- nrow(ep$edges)

  frag <- empty_stan_fragment()

  frag$data <- glue("
  int<lower=0> N_edges;
  array[N_edges] int<lower=1,upper=K> edge_from;
  array[N_edges] int<lower=1,upper=K> edge_to;
  array[K] int<lower=1,upper=K> topo_order;
")

  frag$parameters <- glue("
  vector[N_edges] B_free;
  matrix[{K}, J] z_epsilon;
")

  frag$transformed_parameters <- glue("
  matrix[J, {K}] theta;
  {{
    matrix[{K}, {K}] B = rep_matrix(0, {K}, {K});
    for (n in 1:N_edges)
      B[edge_to[n], edge_from[n]] = B_free[n];
    for (j in 1:J) {{
      for (k_idx in 1:K) {{
        int k = topo_order[k_idx];
        real parent_sum = 0;
        for (l in 1:{K})
          parent_sum += B[k, l] * theta[j, l];
        theta[j, k] = parent_sum + z_epsilon[k, j];
      }}
    }}
  }}
")

  frag$model <- glue("
  // Structural: DAG
  B_free ~ normal(0, 1);
  to_vector(z_epsilon) ~ std_normal();
")

  frag
}


# --- Structural: hierarchical ---

#' @noRd
stan_structural_hierarchical <- function(config) {
  K <- ncol(config$structure$lambda_mask)
  frag <- empty_stan_fragment()

  # Hierarchical: one higher-order factor feeding into K domain factors
  frag$parameters <- glue("
  vector[J] phi;
  vector<lower=0>[{K}] beta_hier;
  matrix[{K}, J] z_epsilon_hier;
")

  frag$transformed_parameters <- glue("
  matrix[J, {K}] theta;
  for (j in 1:J)
    for (k in 1:{K})
      theta[j, k] = beta_hier[k] * phi[j] + z_epsilon_hier[k, j];
")

  frag$model <- glue("
  // Structural: hierarchical (single higher-order factor)
  phi ~ std_normal();
  beta_hier ~ normal(0, 1);
  to_vector(z_epsilon_hier) ~ std_normal();
")

  frag
}


# --- Population: single ---

#' @noRd
stan_population_single <- function(config) {
  empty_stan_fragment()
}


# --- Population: grouped ---

#' @noRd
stan_population_grouped <- function(config) {
  K <- ncol(config$structure$lambda_mask)
  frag <- empty_stan_fragment()

  frag$data <- glue("
  int<lower=1> N_groups;
  array[J] int<lower=1,upper=N_groups> group;
")

  frag$parameters <- glue("
  matrix[N_groups, {K}] mu_group;
  real<lower=0> sigma_group;
")

  frag$model <- glue("
  // Population: grouped
  to_vector(mu_group) ~ normal(0, 1);
  sigma_group ~ normal(0, 0.5);
  // Group-level centering applied to theta via shifted prior
  for (j in 1:J)
    theta[j] ~ normal(mu_group[group[j]], sigma_group);
")

  frag
}


# --- Item: basic ---

#' @noRd
stan_item_basic <- function(config) {
  empty_stan_fragment()
}


# --- Item: slip-guess ---

#' @noRd
stan_item_slip_guess <- function(config) {
  link_inv <- if (config$spec$link == "logit") "inv_logit" else "Phi"

  frag <- empty_stan_fragment()

  # Override compute_prob() to wrap with slip-guess asymptotes
  frag$functions <- glue("
  real compute_prob(int i, row_vector theta_j, matrix Lambda, vector alpha,
                    vector guess, vector slip) {{
    real base_prob = {link_inv}(alpha[i] + Lambda[i] * theta_j');
    return guess[i] + (1 - guess[i] - slip[i]) * base_prob;
  }}
")

  frag$parameters <- glue("
  vector<lower=0,upper=0.5>[I] guess;
  vector<lower=0,upper=0.5>[I] slip;
")

  frag$model <- glue("
  // Item: slip-guess priors
  guess ~ beta(1, 9);
  slip ~ beta(1, 9);
")

  frag
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-stan-blocks.R')"`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add R/stan-blocks.R tests/testthat/test-stan-blocks.R
git commit -m "Add Stan block generators for all measurement, structural, population, and item options"
```

---

## Task 8: Stan Code Generator (Assembly)

**Files:**
- Create: `R/stan-generator.R`
- Create: `tests/testthat/test-stan-generator.R`

- [ ] **Step 1: Write the failing tests**

`tests/testthat/test-stan-generator.R`:

```r
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
    expect_type(code, "character", label = paste(sp$m, sp$s))
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
    expect_type(code, "character",
      label = paste(row$measurement, row$structural, row$population, row$item, row$link))
  }
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-stan-generator.R')"`
Expected: All tests FAIL

- [ ] **Step 3: Implement R/stan-generator.R**

```r
#' Generate complete Stan code from a model configuration
#'
#' Calls each block generator, merges the fragments, and assembles
#' a complete Stan program. Every valid `model_spec` combination
#' produces compilable Stan code.
#'
#' @param config A `model_config` object.
#' @return A character string containing the complete Stan program.
#' @export
generate_stan_code <- function(config) {
  # Collect fragments from each block
  frag_structural <- stan_structural(config)
  frag_measurement <- stan_measurement(config)
  frag_population <- stan_population(config)
  frag_item <- stan_item(config)

  # For slip-guess: the item block provides its own compute_prob() that
  # overrides the measurement block's version. We use the item block's
  # functions fragment INSTEAD of the measurement block's.
  if (config$spec$item == "slip_guess") {
    frag_measurement$functions <- ""
    # Also update model/gen quant to pass guess, slip to compute_prob
    frag_measurement$model <- gsub(
      "compute_prob\\(ii\\[n\\], theta\\[jj\\[n\\]\\], Lambda, alpha\\)",
      "compute_prob(ii[n], theta[jj[n]], Lambda, alpha, guess, slip)",
      frag_measurement$model
    )
    frag_measurement$generated_quantities <- gsub(
      "compute_prob\\(ii\\[n\\], theta\\[jj\\[n\\]\\], Lambda, alpha\\)",
      "compute_prob(ii[n], theta[jj[n]], Lambda, alpha, guess, slip)",
      frag_measurement$generated_quantities
    )
  }

  # Merge all fragments
  merged <- collapse_stan_lists(
    frag_structural,
    frag_measurement,
    frag_population,
    frag_item
  )

  assemble_stan_program(merged, config)
}


#' Assemble Stan fragments into a complete program
#' @noRd
assemble_stan_program <- function(fragments, config) {
  glue("
// autoskill {utils::packageVersion('autoskill') %||% '0.1.0'}
// {config$spec$measurement} / {config$spec$structural} / {config$spec$population} / {config$spec$item} / {config$spec$link}

data {{
{fragments$data}
}}

transformed data {{
{fragments$transformed_data}
}}

parameters {{
{fragments$parameters}
}}

transformed parameters {{
{fragments$transformed_parameters}
}}

model {{
{fragments$model}
}}

generated quantities {{
{fragments$generated_quantities}
}}
")
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-stan-generator.R')"`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add R/stan-generator.R tests/testthat/test-stan-generator.R
git commit -m "Add Stan code generator: assemble block fragments into complete programs"
```

---

## Task 9: Stan Data Preparation and Compilation

**Files:**
- Create: `R/stan-data.R`
- Create: `R/stan-compile.R`
- Create: `tests/testthat/test-stan-data.R`
- Create: `tests/testthat/test-stan-compile.R`

- [ ] **Step 1: Write the failing tests for stan-data**

`tests/testthat/test-stan-data.R`:

```r
test_that("prepare_stan_data returns correct base fields", {
  config <- make_test_config()
  Y <- matrix(sample(0:1, 100 * 8, replace = TRUE), nrow = 100, ncol = 8)
  rownames(Y) <- paste0("s_", 1:100)
  colnames(Y) <- paste0("item_", 1:8)
  rd <- response_data(Y)

  stan_data <- prepare_stan_data(rd, config)

  expect_equal(stan_data$I, 8L)
  expect_equal(stan_data$J, 100L)
  expect_equal(stan_data$K, 3L)
  expect_equal(stan_data$N_loadings, 10L)
  expect_equal(length(stan_data$loading_item), 10L)
  expect_equal(length(stan_data$loading_skill), 10L)
  expect_true(stan_data$N_obs > 0)
})

test_that("prepare_stan_data adds DAG fields for dag structural", {
  config <- make_test_config(structural = "dag")
  Y <- matrix(sample(0:1, 100 * 8, replace = TRUE), nrow = 100, ncol = 8)
  rownames(Y) <- paste0("s_", 1:100)
  colnames(Y) <- paste0("item_", 1:8)
  rd <- response_data(Y)

  stan_data <- prepare_stan_data(rd, config)

  expect_true("N_edges" %in% names(stan_data))
  expect_true("edge_from" %in% names(stan_data))
  expect_true("edge_to" %in% names(stan_data))
  expect_true("topo_order" %in% names(stan_data))
})

test_that("prepare_stan_data adds interaction fields", {
  config <- make_test_config(measurement = "interaction")
  Y <- matrix(sample(0:1, 100 * 8, replace = TRUE), nrow = 100, ncol = 8)
  rownames(Y) <- paste0("s_", 1:100)
  colnames(Y) <- paste0("item_", 1:8)
  rd <- response_data(Y)

  stan_data <- prepare_stan_data(rd, config)

  expect_true("N_interactions" %in% names(stan_data))
})

test_that("prepare_stan_data handles missing data by dropping observations", {
  Y <- matrix(sample(0:1, 50 * 8, replace = TRUE), nrow = 50, ncol = 8)
  Y[1, 1] <- NA
  Y[2, 3] <- NA
  rownames(Y) <- paste0("s_", 1:50)
  colnames(Y) <- paste0("item_", 1:8)
  rd <- response_data(Y)
  config <- make_test_config()

  stan_data <- prepare_stan_data(rd, config)

  # 2 missing observations dropped
  expect_equal(stan_data$N_obs, 50 * 8 - 2)
})
```

- [ ] **Step 2: Write the failing tests for stan-compile**

`tests/testthat/test-stan-compile.R`:

```r
test_that("compile_model compiles Stan code", {
  skip_if_no_cmdstan()

  config <- make_test_config()
  model <- compile_model(config)

  expect_s4_class(model, "CmdStanModel")
})

test_that("compile_model caches by content hash", {
  skip_if_no_cmdstan()

  config <- make_test_config()

  t1 <- system.time(compile_model(config, force = TRUE))
  t2 <- system.time(compile_model(config))

  # Second call should be much faster (cache hit)
  expect_true(t2["elapsed"] < t1["elapsed"])
})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-stan-data.R')"`
Expected: All tests FAIL

- [ ] **Step 4: Implement R/stan-data.R**

```r
#' Prepare Stan data list from response data and model config
#'
#' Converts the response matrix and model configuration into the named
#' list expected by the generated Stan program.
#'
#' @param responses A `response_data` object.
#' @param config A `model_config` object.
#' @return A named list suitable for `cmdstanr::CmdStanModel$sample(data = ...)`.
#' @export
prepare_stan_data <- function(responses, config) {
  mask <- config$structure$lambda_mask
  Y <- responses$Y
  item_ids <- responses$item_ids
  skill_ids <- colnames(mask)

  # Match item ordering between responses and config
  config_items <- rownames(mask)
  item_order <- match(config_items, item_ids)
  if (any(is.na(item_order))) {
    missing <- config_items[is.na(item_order)]
    cli_abort("Items in config not found in responses: {.val {missing}}.")
  }

  I <- ncol(Y)
  J <- nrow(Y)
  K <- ncol(mask)

  # Long-format observation triplets (student, item, response), dropping NAs
  obs <- expand.grid(j = seq_len(J), i = seq_len(I))
  obs$y <- as.integer(Y[cbind(obs$j, obs$i)])
  obs <- obs[!is.na(obs$y), ]

  # Loading index arrays (sparse representation of lambda_mask)
  loading_pairs <- which(mask, arr.ind = TRUE)
  loading_item <- as.integer(loading_pairs[, 1])
  loading_skill <- as.integer(loading_pairs[, 2])

  stan_data <- list(
    N_obs = nrow(obs),
    I = I,
    J = J,
    K = K,
    N_loadings = length(loading_item),
    loading_item = loading_item,
    loading_skill = loading_skill,
    ii = as.integer(obs$i),
    jj = as.integer(obs$j),
    y = obs$y
  )

  # DAG-specific data
  if (config$spec$structural == "dag") {
    ep <- config$edge_prior
    skill_idx <- setNames(seq_along(skill_ids), skill_ids)
    topo <- topological_sort(ep$edges$from, ep$edges$to, skill_ids)

    stan_data$N_edges <- nrow(ep$edges)
    stan_data$edge_from <- as.integer(skill_idx[ep$edges$from])
    stan_data$edge_to <- as.integer(skill_idx[ep$edges$to])
    stan_data$topo_order <- as.integer(skill_idx[topo])
  }

  # Interaction-specific data
  if (config$spec$measurement == "interaction") {
    interact <- compute_interaction_indices(mask)
    stan_data$N_interactions <- nrow(interact)
    stan_data$interact_item <- as.integer(interact$item)
    stan_data$interact_skill1 <- as.integer(interact$skill1)
    stan_data$interact_skill2 <- as.integer(interact$skill2)
  }

  # Grouped population data
  if (config$spec$population == "grouped") {
    # Expects responses to have a $groups field; fall back to single group
    if (!is.null(responses$groups)) {
      stan_data$N_groups <- length(unique(responses$groups))
      stan_data$group <- as.integer(as.factor(responses$groups))
    } else {
      cli_warn("No group information in response data; using single group.")
      stan_data$N_groups <- 1L
      stan_data$group <- rep(1L, J)
    }
  }

  stan_data
}


#' Compute interaction index arrays for cross-loading items
#' @noRd
compute_interaction_indices <- function(mask) {
  result <- list(item = integer(0), skill1 = integer(0), skill2 = integer(0))

  for (i in seq_len(nrow(mask))) {
    skills <- which(mask[i, ])
    if (length(skills) >= 2) {
      pairs <- utils::combn(skills, 2)
      for (p in seq_len(ncol(pairs))) {
        result$item <- c(result$item, i)
        result$skill1 <- c(result$skill1, pairs[1, p])
        result$skill2 <- c(result$skill2, pairs[2, p])
      }
    }
  }

  tibble::tibble(
    item = result$item,
    skill1 = result$skill1,
    skill2 = result$skill2
  )
}
```

- [ ] **Step 5: Implement R/stan-compile.R**

```r
#' Compile a Stan model from a model configuration
#'
#' Generates Stan code, writes it to a file, and compiles via cmdstanr.
#' Results are cached by content hash so identical configs are not
#' recompiled.
#'
#' @param config A `model_config` object.
#' @param dir Cache directory. Defaults to `tools::R_user_dir("autoskill", "cache")`.
#' @param force If `TRUE`, recompile even if cached.
#' @return A `CmdStanModel` object.
#' @export
compile_model <- function(config, dir = NULL, force = FALSE) {
  dir <- dir %||% file.path(tools::R_user_dir("autoskill", "cache"), "stan")
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)

  code <- generate_stan_code(config)
  hash <- rlang::hash(code)
  stan_file <- file.path(dir, paste0("autoskill_", hash, ".stan"))

  if (!file.exists(stan_file) || force) {
    writeLines(code, stan_file)
  }

  cmdstanr::cmdstan_model(stan_file, compile = TRUE)
}


#' Clear the autoskill Stan model cache
#'
#' @param dir Cache directory. Defaults to the standard cache location.
#' @export
clear_stan_cache <- function(dir = NULL) {
  dir <- dir %||% file.path(tools::R_user_dir("autoskill", "cache"), "stan")
  if (dir.exists(dir)) {
    files <- list.files(dir, full.names = TRUE)
    file.remove(files)
    cli_inform("Removed {length(files)} cached Stan files.")
  }
  invisible()
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-stan-data.R')"`
Expected: All tests PASS

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-stan-compile.R')"`
Expected: Compilation test PASSES (may be slow first time)

- [ ] **Step 7: Commit**

```bash
git add R/stan-data.R R/stan-compile.R tests/testthat/test-stan-data.R tests/testthat/test-stan-compile.R
git commit -m "Add Stan data preparation and model compilation with content-hash caching"
```

---

## Task 10: Compilation Test for All 32 Block Combinations

This is the critical correctness gate: every valid block combination must compile.

**Files:**
- Modify: `tests/testthat/test-stan-compile.R`

- [ ] **Step 1: Add the compilation matrix test**

Append to `tests/testthat/test-stan-compile.R`:

```r
test_that("all 64 block combinations compile", {
  skip_if_no_cmdstan()
  skip_on_cran()

  specs <- all_valid_specs()
  failures <- character(0)

  for (i in seq_len(nrow(specs))) {
    row <- specs[i, ]
    label <- paste(row$measurement, row$structural, row$population, row$item, row$link,
                   sep = "/")

    result <- tryCatch(
      {
        config <- make_test_config(
          measurement = row$measurement,
          structural = row$structural,
          population = row$population,
          item = row$item,
          link = row$link
        )
        compile_model(config)
        NULL
      },
      error = function(e) conditionMessage(e)
    )

    if (!is.null(result)) {
      failures <- c(failures, paste0(label, ": ", result))
    }
  }

  if (length(failures) > 0) {
    fail(paste0(
      length(failures), " of 64 block combinations failed to compile:\n",
      paste("  -", failures, collapse = "\n")
    ))
  }

  expect_length(failures, 0)
})
```

- [ ] **Step 2: Run the compilation matrix**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-stan-compile.R')"`
Expected: All 64 combinations compile. If any fail, fix the Stan fragments in `R/stan-blocks.R` and rerun until all pass.

- [ ] **Step 3: Fix any compilation failures and rerun**

Common issues to watch for:
- Mismatched braces in glue templates (use `{{` and `}}` for literal braces)
- Missing semicolons in Stan code
- Type mismatches (e.g., `vector` vs `row_vector` in dot products)
- Undeclared variables across block boundaries (e.g., `theta` must be declared by structural before measurement uses it)

- [ ] **Step 4: Commit**

```bash
git add tests/testthat/test-stan-compile.R
git commit -m "Add compilation matrix test: all 64 block combinations compile"
```

---

## Task 11: Data Simulation

**Files:**
- Create: `R/simulate-data.R`
- Create: `tests/testthat/test-simulate-data.R`

- [ ] **Step 1: Write the failing tests**

`tests/testthat/test-simulate-data.R`:

```r
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

  # theta should show correlation structure
  cor_mat <- cor(sim$params$theta)
  # Off-diagonal should not all be zero

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

  # Call it once to verify output structure
  result <- gen()
  expect_true("variables" %in% names(result))
  expect_true("generated" %in% names(result))
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-simulate-data.R')"`
Expected: All tests FAIL

- [ ] **Step 3: Implement R/simulate-data.R**

```r
#' Simulate student response data from a model configuration
#'
#' Generates synthetic data by following the generative process defined
#' by the model configuration: draw skill vectors, compute response
#' probabilities, sample binary responses.
#'
#' @param config A `model_config` object.
#' @param n_students Number of students to simulate.
#' @param seed Random seed for reproducibility.
#' @param params Optional named list of true parameter values. If `NULL`,
#'   parameters are drawn from the prior.
#' @return A list with components:
#'   - `responses`: A `response_data` object.
#'   - `params`: Named list of true parameter values.
#'   - `config`: The input config.
#' @export
simulate_responses <- function(config, n_students = 200, seed = NULL,
                               params = NULL) {
  if (!is.null(seed)) set.seed(seed)

  mask <- config$structure$lambda_mask
  I <- nrow(mask)
  K <- ncol(mask)
  J <- n_students

  # Draw or use provided parameters
  if (is.null(params)) params <- list()

  # Item intercepts
  alpha <- params$alpha %||% stats::rnorm(I, 0, 1)

  # Loadings (positive, ~0.8 on average)
  n_load <- sum(mask)
  lambda_vals <- params$lambda_free %||% abs(stats::rnorm(n_load, 0.8, 0.3))
  Lambda <- matrix(0, I, K)
  Lambda[mask] <- lambda_vals

  # Skill vectors (dispatch by structural block)
  theta <- params$theta %||% simulate_theta(config, J, K)

  # Linear predictor
  eta <- matrix(NA_real_, J, I)
  for (j in seq_len(J)) {
    eta[j, ] <- alpha + as.numeric(Lambda %*% theta[j, ])
  }

  # Apply link function
  link_inv <- if (config$spec$link == "logit") stats::plogis else stats::pnorm
  prob <- link_inv(eta)

  # Apply slip-guess if needed
  if (config$spec$item == "slip_guess") {
    guess <- params$guess %||% stats::rbeta(I, 1, 9)
    slip <- params$slip %||% stats::rbeta(I, 1, 9)
    prob <- sweep(prob, 2, guess, "+") -
      sweep(sweep(prob, 2, guess, "+"), 2, slip + guess, "*") +
      sweep(prob, 2, 1 - slip - guess, "*")
    # Cleaner: p = guess + (1 - guess - slip) * base_prob
    prob <- t(guess + (1 - guess - slip) * t(link_inv(eta)))
    params$guess <- guess
    params$slip <- slip
  }

  # Sample responses
  Y <- matrix(
    stats::rbinom(J * I, 1, as.numeric(prob)),
    nrow = J, ncol = I
  )
  rownames(Y) <- paste0("student_", seq_len(J))
  colnames(Y) <- rownames(mask)

  params$alpha <- alpha
  params$lambda_free <- lambda_vals
  params$Lambda <- Lambda
  params$theta <- theta

  list(
    responses = response_data(Y),
    params = params,
    config = config
  )
}


#' Simulate theta based on structural block
#' @noRd
simulate_theta <- function(config, J, K) {
  switch(config$spec$structural,
    independent = matrix(stats::rnorm(J * K), J, K),
    correlated = simulate_theta_correlated(J, K),
    dag = simulate_theta_dag(config, J, K),
    hierarchical = simulate_theta_hierarchical(J, K)
  )
}


#' @noRd
simulate_theta_correlated <- function(J, K, Sigma = NULL) {
  if (is.null(Sigma)) {
    # Draw a random correlation matrix via LKJ(2)
    # Simple approach: random Cholesky factor
    L <- matrix(0, K, K)
    L[1, 1] <- 1
    for (i in 2:K) {
      for (j in 1:(i - 1)) {
        L[i, j] <- stats::rnorm(1, 0, 0.3)
      }
      L[i, i] <- sqrt(max(0.01, 1 - sum(L[i, 1:(i - 1)]^2)))
    }
    Sigma <- L %*% t(L)
  }
  MASS_mvrnorm <- function(n, mu, Sigma) {
    # Minimal MVN sampler without MASS dependency
    K <- length(mu)
    L <- chol(Sigma)
    Z <- matrix(stats::rnorm(n * K), n, K)
    sweep(Z %*% L, 2, mu, "+")
  }
  MASS_mvrnorm(J, rep(0, K), Sigma)
}


#' @noRd
simulate_theta_dag <- function(config, J, K) {
  ep <- config$edge_prior
  skill_ids <- colnames(config$structure$lambda_mask)
  topo <- topological_sort(ep$edges$from, ep$edges$to, skill_ids)
  skill_idx <- setNames(seq_along(skill_ids), skill_ids)

  # Draw structural coefficients
  B <- matrix(0, K, K)
  for (i in seq_len(nrow(ep$edges))) {
    from_idx <- skill_idx[ep$edges$from[i]]
    to_idx <- skill_idx[ep$edges$to[i]]
    B[to_idx, from_idx] <- stats::rnorm(1, 0.5, 0.3)
  }

  theta <- matrix(0, J, K)
  for (node_name in topo) {
    k <- skill_idx[node_name]
    parent_contribution <- theta %*% B[k, ]
    theta[, k] <- as.numeric(parent_contribution) + stats::rnorm(J)
  }
  theta
}


#' @noRd
simulate_theta_hierarchical <- function(J, K) {
  phi <- stats::rnorm(J)
  beta_hier <- abs(stats::rnorm(K, 0.7, 0.2))
  theta <- outer(phi, beta_hier) + matrix(stats::rnorm(J * K), J, K)
  theta
}


#' Create an SBC generator function
#'
#' Returns a function compatible with `SBC::SBC_generator_function()`.
#' Each call draws parameters from the prior and simulates data.
#'
#' @param config A `model_config` object.
#' @param n_students Number of students per simulation.
#' @return A function that returns a list with `variables` and `generated`.
#' @export
sbc_generator <- function(config, n_students = 200) {
  function() {
    sim <- simulate_responses(config, n_students = n_students)
    stan_data <- prepare_stan_data(sim$responses, config)

    # Flatten params into a named vector for SBC
    variables <- list(
      alpha = sim$params$alpha,
      lambda_free = sim$params$lambda_free
    )

    list(
      variables = posterior::draws_matrix(
        !!!setNames(
          as.list(c(sim$params$alpha, sim$params$lambda_free)),
          c(paste0("alpha[", seq_along(sim$params$alpha), "]"),
            paste0("lambda_free[", seq_along(sim$params$lambda_free), "]"))
        )
      ),
      generated = stan_data
    )
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-simulate-data.R')"`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add R/simulate-data.R tests/testthat/test-simulate-data.R
git commit -m "Add data simulation engine with per-block theta generation and SBC generator"
```

---

## Task 12: Identifiability Check and SBC

**Files:**
- Create: `R/sbc-check.R`
- Create: `tests/testthat/test-sbc-check.R`

- [ ] **Step 1: Write the failing tests**

`tests/testthat/test-sbc-check.R`:

```r
test_that("check_identifiability passes for valid config", {
  config <- make_test_config()
  result <- check_identifiability(config)

  expect_true(result$passed)
  expect_length(result$problems, 0)
})

test_that("check_identifiability catches skills with < 2 items", {
  # This is already caught by loading_structure validation,
  # but check_identifiability should also flag it
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()

  # Manually create a bad config (bypass validation for testing)
  structure <- loading_structure(taxonomy, assignments, items)
  # Simulate a skill with only 1 item by zeroing out
  structure$lambda_mask[, "skill_3"] <- FALSE
  structure$lambda_mask["item_4", "skill_3"] <- TRUE

  config <- structure(
    list(spec = model_spec(), structure = structure, edge_prior = NULL),
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
  structure <- loading_structure(taxonomy, assignments, items)

  ep <- edge_prior(
    from = c("skill_1", "skill_2", "skill_3"),
    to = c("skill_2", "skill_3", "skill_1"),
    prob = c(0.8, 0.8, 0.8)
  )

  config <- structure(
    list(
      spec = model_spec(structural = "dag"),
      structure = structure,
      edge_prior = ep
    ),
    class = "model_config"
  )

  result <- check_identifiability(config)
  expect_false(result$passed)
  expect_true(any(grepl("cycle", result$problems, ignore.case = TRUE)))
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-sbc-check.R')"`
Expected: All tests FAIL

- [ ] **Step 3: Implement R/sbc-check.R**

```r
#' Quick identifiability pre-check
#'
#' Checks structural properties of the model configuration without
#' running any inference. Catches common non-identifiability issues
#' before expensive MCMC or SBC.
#'
#' @param config A `model_config` object.
#' @return A list with `$passed` (logical) and `$problems` (character vector).
#' @export
check_identifiability <- function(config) {
  problems <- character(0)
  mask <- config$structure$lambda_mask

  # Each skill needs >= 2 items
  items_per_skill <- colSums(mask)
  thin <- names(items_per_skill[items_per_skill < 2])
  if (length(thin) > 0) {
    problems <- c(problems, glue("Skills with < 2 items (not identifiable): {paste(thin, collapse = ', ')}"))
  }

  # Each item needs >= 1 skill
  skills_per_item <- rowSums(mask)
  orphans <- names(skills_per_item[skills_per_item == 0])
  if (length(orphans) > 0) {
    problems <- c(problems, glue("Items with no skills: {paste(orphans, collapse = ', ')}"))
  }

  # No duplicate loading patterns (two skills with identical item sets)
  patterns <- apply(mask, 2, paste, collapse = "")
  dups <- names(which(duplicated(patterns)))
  if (length(dups) > 0) {
    problems <- c(problems, glue("Duplicate loading patterns (not identifiable): {paste(dups, collapse = ', ')}"))
  }

  # DAG-specific: must be acyclic

  if (config$spec$structural == "dag" && !is.null(config$edge_prior)) {
    ep <- config$edge_prior
    skill_ids <- colnames(mask)
    if (!validate_dag(ep$edges$from, ep$edges$to, skill_ids)) {
      problems <- c(problems, "DAG contains a cycle.")
    }
  }

  list(passed = length(problems) == 0, problems = problems)
}


#' Run simulation-based calibration
#'
#' Wraps the SBC package to check that the model can recover known
#' parameters from simulated data.
#'
#' @param config A `model_config` object.
#' @param n_sims Number of SBC simulations.
#' @param n_students Number of students per simulation.
#' @param ... Additional arguments passed to `cmdstanr::CmdStanModel$sample()`.
#' @return An S3 object of class `sbc_result`.
#' @export
run_sbc <- function(config, n_sims = 100, n_students = 200, ...) {
  if (!requireNamespace("SBC", quietly = TRUE)) {
    cli_abort("Package {.pkg SBC} is required for SBC checks. Install with: {.code install.packages('SBC')}")
  }

  # Pre-check identifiability
  id_check <- check_identifiability(config)
  if (!id_check$passed) {
    return(structure(
      list(passed = FALSE, problems = id_check$problems, rank_stats = NULL),
      class = "sbc_result"
    ))
  }

  gen <- sbc_generator(config, n_students = n_students)
  model <- compile_model(config)

  # Run SBC
  datasets <- SBC::SBC_generator_function(gen, N = n_sims)
  backend <- SBC::SBC_backend_cmdstan_sample(model, ...)
  results <- SBC::compute_SBC(datasets, backend)

  passed <- !any(results$stats$z_score_warning)

  structure(
    list(
      passed = passed,
      problems = if (!passed) "SBC rank statistics show calibration issues" else character(0),
      rank_stats = results
    ),
    class = "sbc_result"
  )
}


#' @export
print.sbc_result <- function(x, ...) {
  status <- if (x$passed) "PASSED" else "FAILED"
  cli_inform("{.cls sbc_result}: {status}")
  if (length(x$problems) > 0) {
    cli_inform(c("!" = "{x$problems}"))
  }
  invisible(x)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-sbc-check.R')"`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add R/sbc-check.R tests/testthat/test-sbc-check.R
git commit -m "Add identifiability pre-check and SBC wrapper"
```

---

## Task 13: Model Fitting and Diagnostics

**Files:**
- Create: `R/model-fit.R`
- Create: `tests/testthat/test-model-fit.R`

- [ ] **Step 1: Write the failing tests**

`tests/testthat/test-model-fit.R`:

```r
test_that("fit_model returns fit_result with all components", {
  skip_if_no_cmdstan()
  skip_on_cran()

  config <- make_test_config()
  sim <- simulate_responses(config, n_students = 100, seed = 42)

  result <- fit_model(sim$responses, config,
    chains = 2, iter_warmup = 200, iter_sampling = 200
  )

  expect_s3_class(result, "fit_result")
  expect_true(!is.null(result$fit))
  expect_s3_class(result$config, "model_config")
  expect_s3_class(result$diagnostics, "tbl_df")
  expect_true(!is.null(result$loo))
  expect_s3_class(result$param_summary, "tbl_df")
})

test_that("extract_diagnostics returns expected metrics", {
  skip_if_no_cmdstan()
  skip_on_cran()

  config <- make_test_config()
  sim <- simulate_responses(config, n_students = 100, seed = 42)
  result <- fit_model(sim$responses, config,
    chains = 2, iter_warmup = 200, iter_sampling = 200
  )

  diag <- result$diagnostics
  expect_true("n_divergences" %in% diag$metric)
  expect_true("max_rhat" %in% diag$metric)
  expect_true("min_bulk_ess" %in% diag$metric)
})

test_that("compute_loo returns loo object", {
  skip_if_no_cmdstan()
  skip_on_cran()

  config <- make_test_config()
  sim <- simulate_responses(config, n_students = 100, seed = 42)
  result <- fit_model(sim$responses, config,
    chains = 2, iter_warmup = 200, iter_sampling = 200
  )

  expect_s3_class(result$loo, "loo")
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-model-fit.R')"`
Expected: All tests FAIL

- [ ] **Step 3: Implement R/model-fit.R**

```r
#' Fit a model to student response data
#'
#' Compiles the Stan model, prepares data, runs MCMC via cmdstanr,
#' and extracts diagnostics and LOO-CV.
#'
#' @param responses A `response_data` object.
#' @param config A `model_config` object.
#' @param chains Number of MCMC chains.
#' @param iter_warmup Number of warmup iterations per chain.
#' @param iter_sampling Number of sampling iterations per chain.
#' @param adapt_delta Target acceptance rate.
#' @param max_treedepth Maximum tree depth for NUTS.
#' @param ... Additional arguments passed to `CmdStanModel$sample()`.
#' @return An S3 object of class `fit_result`.
#' @export
fit_model <- function(responses, config,
                      chains = 4,
                      iter_warmup = 1000,
                      iter_sampling = 1000,
                      adapt_delta = 0.95,
                      max_treedepth = 12,
                      ...) {
  model <- compile_model(config)
  stan_data <- prepare_stan_data(responses, config)

  fit <- model$sample(
    data = stan_data,
    chains = chains,
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    adapt_delta = adapt_delta,
    max_treedepth = max_treedepth,
    refresh = 0,
    ...
  )

  diagnostics <- extract_diagnostics(fit)
  loo_obj <- compute_loo(fit)
  param_summary <- extract_param_summary(fit, config)

  structure(
    list(
      fit = fit,
      config = config,
      diagnostics = diagnostics,
      loo = loo_obj,
      param_summary = param_summary
    ),
    class = "fit_result"
  )
}


#' Extract MCMC diagnostics
#'
#' @param fit A `CmdStanMCMC` object.
#' @return A tibble with columns: metric, value, status.
#' @export
extract_diagnostics <- function(fit) {
  sampler_diag <- fit$diagnostic_summary()
  summary_df <- fit$summary()

  n_div <- sum(sampler_diag$num_divergent)
  max_rhat <- max(summary_df$rhat, na.rm = TRUE)
  min_bulk <- min(summary_df$ess_bulk, na.rm = TRUE)
  min_tail <- min(summary_df$ess_tail, na.rm = TRUE)

  tibble::tibble(
    metric = c("n_divergences", "max_rhat", "min_bulk_ess", "min_tail_ess"),
    value = c(n_div, max_rhat, min_bulk, min_tail),
    status = c(
      if (n_div == 0) "ok" else "critical",
      if (max_rhat < 1.01) "ok" else if (max_rhat < 1.05) "warning" else "critical",
      if (min_bulk > 400) "ok" else if (min_bulk > 100) "warning" else "critical",
      if (min_tail > 400) "ok" else if (min_tail > 100) "warning" else "critical"
    )
  )
}


#' Compute LOO-CV via PSIS-LOO
#'
#' @param fit A `CmdStanMCMC` object with `log_lik` in generated quantities.
#' @return A `loo` object.
#' @export
compute_loo <- function(fit) {
  log_lik <- fit$draws("log_lik", format = "matrix")
  loo::loo(log_lik, r_eff = loo::relative_eff(exp(log_lik)))
}


#' Extract parameter summaries mapped to item/skill names
#'
#' @param fit A `CmdStanMCMC` object.
#' @param config A `model_config` object.
#' @return A tibble with parameter summaries.
#' @export
extract_param_summary <- function(fit, config) {
  summary_df <- fit$summary() |>
    tibble::as_tibble()

  # Map alpha indices to item names
  item_ids <- rownames(config$structure$lambda_mask)
  skill_ids <- colnames(config$structure$lambda_mask)

  summary_df |>
    dplyr::mutate(
      param_type = dplyr::case_when(
        grepl("^alpha\\[", variable) ~ "alpha",
        grepl("^lambda_free\\[", variable) ~ "lambda",
        grepl("^theta\\[", variable) ~ "theta",
        TRUE ~ "other"
      )
    )
}


#' @export
print.fit_result <- function(x, ...) {
  elpd <- x$loo$estimates["elpd_loo", "Estimate"]
  n_critical <- sum(x$diagnostics$status == "critical")

  cli_inform(c(
    "{.cls fit_result}: ELPD = {round(elpd, 1)}",
    if (n_critical == 0) c("v" = "All diagnostics OK") else c("!" = "{n_critical} critical diagnostics")
  ))
  invisible(x)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-model-fit.R')"`
Expected: All tests PASS (may take 1-2 minutes for MCMC)

- [ ] **Step 5: Commit**

```bash
git add R/model-fit.R tests/testthat/test-model-fit.R
git commit -m "Add model fitting with cmdstanr, diagnostics extraction, and LOO-CV"
```

---

## Task 14: Model Comparison

**Files:**
- Create: `R/model-compare.R`
- Create: `tests/testthat/test-model-compare.R`

- [ ] **Step 1: Write the failing tests**

`tests/testthat/test-model-compare.R`:

```r
test_that("compare_models returns loo comparison", {
  skip_if_no_cmdstan()
  skip_on_cran()

  config1 <- make_test_config(structural = "independent")
  config2 <- make_test_config(structural = "correlated")

  # Simulate from correlated model
  sim <- simulate_responses(config2, n_students = 200, seed = 42)

  r1 <- fit_model(sim$responses, config1, chains = 2, iter_warmup = 200, iter_sampling = 200)
  r2 <- fit_model(sim$responses, config2, chains = 2, iter_warmup = 200, iter_sampling = 200)

  comp <- compare_models(independent = r1, correlated = r2)

  expect_s3_class(comp, "tbl_df")
  expect_true("model" %in% names(comp))
  expect_true("elpd_diff" %in% names(comp))
})

test_that("flag_problem_items identifies high pareto k", {
  skip_if_no_cmdstan()
  skip_on_cran()

  config <- make_test_config()
  sim <- simulate_responses(config, n_students = 100, seed = 42)
  result <- fit_model(sim$responses, config, chains = 2, iter_warmup = 200, iter_sampling = 200)

  flagged <- flag_problem_items(result)
  expect_s3_class(flagged, "tbl_df")
  expect_true("item_id" %in% names(flagged))
  expect_true("pareto_k" %in% names(flagged))
})

test_that("format_reflection_prompt returns a character string", {
  skip_if_no_cmdstan()
  skip_on_cran()

  config <- make_test_config()
  sim <- simulate_responses(config, n_students = 100, seed = 42)
  result <- fit_model(sim$responses, config, chains = 2, iter_warmup = 200, iter_sampling = 200)

  prompt <- format_reflection_prompt(result, items = make_test_items())
  expect_type(prompt, "character")
  expect_true(nchar(prompt) > 100)
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-model-compare.R')"`
Expected: All tests FAIL

- [ ] **Step 3: Implement R/model-compare.R**

```r
#' Compare multiple fitted models by LOO-CV
#'
#' @param ... Named `fit_result` objects.
#' @return A tibble with LOO comparison statistics.
#' @export
compare_models <- function(...) {
  results <- list(...)
  loo_list <- lapply(results, function(r) r$loo)
  comp <- loo::loo_compare(loo_list)

  tibble::tibble(
    model = rownames(comp),
    elpd_diff = comp[, "elpd_diff"],
    se_diff = comp[, "se_diff"],
    elpd_loo = comp[, "elpd_loo"],
    se_elpd_loo = comp[, "se_elpd_loo"],
    p_loo = comp[, "p_loo"]
  )
}


#' Flag items with high Pareto k diagnostic
#'
#' Items with Pareto k > threshold indicate observations that are
#' poorly predicted by the model and may be driving poor LOO estimates.
#'
#' @param fit_result A `fit_result` object.
#' @param k_threshold Pareto k threshold (default 0.7).
#' @return A tibble with flagged items.
#' @export
flag_problem_items <- function(fit_result, k_threshold = 0.7) {
  k_values <- loo::pareto_k_values(fit_result$loo)
  item_ids <- fit_result$config$structure$items$item_id

  # Map observation-level k back to items
  stan_data <- prepare_stan_data(
    response_data(matrix(0L, 1, length(item_ids),
      dimnames = list(NULL, item_ids))),
    fit_result$config
  )

  # Aggregate k by item (take max per item)
  obs_items <- stan_data$ii
  k_by_obs <- tibble::tibble(
    obs = seq_along(k_values),
    pareto_k = k_values
  )

  # Since we may not have the exact mapping, report observation-level

  tibble::tibble(
    obs_id = seq_along(k_values),
    pareto_k = k_values,
    status = dplyr::case_when(
      k_values > 1.0 ~ "very_bad",
      k_values > k_threshold ~ "bad",
      k_values > 0.5 ~ "ok",
      TRUE ~ "good"
    )
  ) |>
    dplyr::filter(pareto_k > k_threshold)
}


#' Format diagnostics as a structured reflection prompt
#'
#' Creates a text summary of model performance for LLM reflection.
#'
#' @param fit_result A `fit_result` object.
#' @param comparison Optional tibble from `compare_models()`.
#' @param previous_configs Optional list of previously tried configs.
#' @param items Tibble of items with `item_id` and `text`.
#' @return A character string.
#' @export
format_reflection_prompt <- function(fit_result,
                                     comparison = NULL,
                                     previous_configs = NULL,
                                     items = NULL) {
  config <- fit_result$config
  diag <- fit_result$diagnostics
  elpd <- fit_result$loo$estimates["elpd_loo", "Estimate"]

  parts <- c(
    "## Current Model",
    glue("- Measurement: {config$spec$measurement}"),
    glue("- Structural: {config$spec$structural}"),
    glue("- Population: {config$spec$population}"),
    glue("- Item: {config$spec$item}"),
    glue("- Link: {config$spec$link}"),
    glue("- Skills: {paste(config$structure$taxonomy$name, collapse = ', ')}"),
    "",
    "## Loading Matrix",
    format_mask_for_prompt(config$structure, items),
    "",
    "## Diagnostics",
    glue("- ELPD (LOO): {round(elpd, 1)}"),
    paste(glue("- {diag$metric}: {round(diag$value, 3)} ({diag$status})"), collapse = "\n")
  )

  # Problem items
  problems <- flag_problem_items(fit_result)
  if (nrow(problems) > 0) {
    parts <- c(parts, "",
      glue("## Problem Observations ({nrow(problems)} with Pareto k > 0.7)"))
  }

  # Comparison
  if (!is.null(comparison)) {
    parts <- c(parts, "",
      "## Model Comparison (LOO)",
      paste(utils::capture.output(print(comparison)), collapse = "\n"))
  }

  # History
  if (!is.null(previous_configs) && length(previous_configs) > 0) {
    parts <- c(parts, "",
      "## Previously Tried Configurations",
      paste(vapply(previous_configs, function(pc) {
        glue("- {pc$spec$measurement}/{pc$spec$structural}/{pc$spec$population}/{pc$spec$item}")
      }, character(1)), collapse = "\n"))
  }

  paste(parts, collapse = "\n")
}


#' @noRd
format_mask_for_prompt <- function(structure, items = NULL) {
  mask <- structure$lambda_mask
  skill_names <- structure$taxonomy$name

  header <- paste(c("Item", skill_names), collapse = " | ")
  separator <- paste(rep("---", length(skill_names) + 1), collapse = " | ")

  rows <- vapply(seq_len(nrow(mask)), function(i) {
    item_label <- if (!is.null(items)) {
      items$text[items$item_id == rownames(mask)[i]]
    } else {
      rownames(mask)[i]
    }
    marks <- ifelse(mask[i, ], "X", ".")
    paste(c(item_label, marks), collapse = " | ")
  }, character(1))

  paste(c(header, separator, rows), collapse = "\n")
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-model-compare.R')"`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add R/model-compare.R tests/testthat/test-model-compare.R
git commit -m "Add model comparison, problem flagging, and LLM reflection prompt formatting"
```

---

## Task 15: LLM Reflection (propose_refinement)

**Files:**
- Create: `R/reflection.R`

- [ ] **Step 1: Implement R/reflection.R**

```r
#' Propose a refined model configuration via LLM
#'
#' Sends the current model diagnostics to an LLM and receives a
#' structured proposal for a refined configuration.
#'
#' @param chat An ellmer chat object.
#' @param fit_result A `fit_result` object.
#' @param items Tibble with `item_id` and `text`.
#' @param comparison Optional tibble from `compare_models()`.
#' @param previous_configs Optional list of previously tried configs.
#' @return A `model_config` object.
#' @export
propose_refinement <- function(chat, fit_result, items,
                               comparison = NULL,
                               previous_configs = NULL) {
  prompt <- format_reflection_prompt(
    fit_result,
    comparison = comparison,
    previous_configs = previous_configs,
    items = items
  )

  prompt <- paste0(
    prompt,
    "\n\n## Your Task\n",
    "Based on the diagnostics above, propose a refined model configuration.\n",
    "You may change:\n",
    "1. The block configuration (measurement, structural, population, item)\n",
    "2. The loading matrix (which items load on which skills)\n",
    "3. The skill taxonomy (add, remove, or rename skills)\n\n",
    "Explain your rationale.\n"
  )

  skill_ids <- fit_result$config$structure$taxonomy$skill_id
  item_ids <- fit_result$config$structure$items$item_id

  response_type <- refinement_output_type(skill_ids, item_ids)

  result <- chat$chat_structured(prompt, type = response_type)

  # Build new config from LLM response
  build_config_from_refinement(result, items)
}


#' Define structured output type for refinement
#' @noRd
refinement_output_type <- function(skill_ids, item_ids) {
  ellmer::type_object(
    measurement = ellmer::type_enum(
      values = MEASUREMENT_OPTIONS,
      description = "Measurement block choice"
    ),
    structural = ellmer::type_enum(
      values = STRUCTURAL_OPTIONS,
      description = "Structural block choice"
    ),
    population = ellmer::type_enum(
      values = POPULATION_OPTIONS,
      description = "Population block choice"
    ),
    item = ellmer::type_enum(
      values = ITEM_OPTIONS,
      description = "Item block choice"
    ),
    skills = ellmer::type_array(
      items = ellmer::type_object(
        skill_id = ellmer::type_string("Skill identifier"),
        name = ellmer::type_string("Skill name"),
        description = ellmer::type_string("Skill description")
      ),
      description = "Updated skill taxonomy"
    ),
    assignments = ellmer::type_array(
      items = ellmer::type_object(
        item_id = ellmer::type_string("Item identifier"),
        skills = ellmer::type_array(
          items = ellmer::type_string("Skill ID"),
          description = "Skills this item requires"
        )
      ),
      description = "Updated skill assignments per item"
    ),
    edge_prior = ellmer::type_array(
      items = ellmer::type_object(
        from = ellmer::type_string("Parent skill ID"),
        to = ellmer::type_string("Child skill ID"),
        prob = ellmer::type_number("Prior probability of this edge")
      ),
      description = "Edge prior for DAG mode (empty array if not DAG)"
    ),
    rationale = ellmer::type_string("Explanation of proposed changes")
  )
}


#' Build model_config from LLM refinement response
#' @noRd
build_config_from_refinement <- function(result, items) {
  taxonomy <- tibble::tibble(
    skill_id = result$skills$skill_id,
    name = result$skills$name,
    description = result$skills$description,
    is_new = TRUE
  )

  # Unnest assignments
  assignments_raw <- result$assignments
  assignments <- purrr::map_dfr(seq_len(nrow(assignments_raw)), function(i) {
    row <- assignments_raw[i, ]
    skill_list <- row$skills[[1]]
    tibble::tibble(
      item_id = row$item_id,
      skill_id = skill_list,
      skill_name = taxonomy$name[match(skill_list, taxonomy$skill_id)]
    )
  })

  structure <- loading_structure(taxonomy, assignments, items)

  spec <- model_spec(
    measurement = result$measurement,
    structural = result$structural,
    population = result$population,
    item = result$item
  )

  ep <- NULL
  if (result$structural == "dag" && nrow(result$edge_prior) > 0) {
    ep <- edge_prior(
      from = result$edge_prior$from,
      to = result$edge_prior$to,
      prob = result$edge_prior$prob
    )
  }

  model_config(spec, structure, edge_prior = ep)
}


#' Build the reflection system prompt
#' @noRd
build_reflection_system_prompt <- function() {
  paste(
    "You are an expert psychometrician and Bayesian modeler.",
    "You are helping refine a latent knowledge component model.",
    "You understand IRT, factor analysis, and structural equation modeling.",
    "You use diagnostic information (LOO-CV, Pareto k, R-hat, ESS) to guide model refinement.",
    "When proposing changes, explain your reasoning and be specific.",
    sep = "\n"
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add R/reflection.R
git commit -m "Add LLM reflection module with structured output for model refinement"
```

---

## Task 16: Refactor Skill Proposer for Package

**Files:**
- Modify: `R/skill-proposer.R`
- Create: `tests/testthat/test-skill-proposer.R`

- [ ] **Step 1: Write tests with mocked LLM**

`tests/testthat/test-skill-proposer.R`:

```r
test_that("propose_skills returns a loading_structure", {
  # Mock the chat object
  mock_chat <- list(
    chat_structured = function(prompt, type) {
      if (grepl("Analyze these test items", prompt)) {
        # Taxonomy response
        tibble::tibble(
          name = c("Linear equations", "Fraction arithmetic", "Equation setup"),
          description = c("Solving equations", "Fraction operations", "Setting up equations"),
          is_new = c(TRUE, TRUE, TRUE)
        )
      } else {
        # Assignment response
        tibble::tibble(
          item_id = c("item_1", "item_2", "item_3", "item_4", "item_4",
                      "item_5", "item_6", "item_7", "item_7", "item_8"),
          skills = list("Linear equations", "Linear equations", "Fraction arithmetic",
                       "Linear equations", "Equation setup",
                       "Linear equations", "Fraction arithmetic",
                       "Linear equations", "Equation setup", "Fraction arithmetic")
        )
      }
    }
  )

  items <- make_test_items()

  # Test the two-stage pipeline with our mock
  taxonomy <- propose_taxonomy(mock_chat, items)
  expect_s3_class(taxonomy, "tbl_df")
  expect_true("skill_id" %in% names(taxonomy))

  assignments <- assign_skills(mock_chat, items, taxonomy)
  expect_s3_class(assignments, "tbl_df")
})

test_that("build_taxonomy_prompt includes items", {
  items <- make_test_items()
  prompt <- build_taxonomy_prompt(items)
  expect_match(prompt, "3x \\+ 5 = 20")
})

test_that("build_taxonomy_prompt includes known skills", {
  items <- make_test_items()
  known <- tibble::tibble(
    name = "Algebra",
    description = "Basic algebra"
  )
  prompt <- build_taxonomy_prompt(items, known_skills = known)
  expect_match(prompt, "Algebra")
})
```

- [ ] **Step 2: Refactor R/skill-proposer.R**

Remove `library()` calls, add roxygen2 docs, use package-qualified imports:

```r
# Type definitions --------------------------------------------------------

skill_type <- ellmer::type_object(
  name = ellmer::type_string("Short descriptive name for the skill"),
  description = ellmer::type_string("What this skill involves, in one sentence"),
  is_new = ellmer::type_boolean("TRUE if this skill was newly proposed, FALSE if from the known set")
)

assignment_type <- ellmer::type_object(
  item_id = ellmer::type_string("The item identifier"),
  skills = ellmer::type_array(
    items = ellmer::type_string("Name of a required skill"),
    description = "Skills this item requires"
  )
)


# Prompt construction -----------------------------------------------------

#' @noRd
build_taxonomy_prompt <- function(items, known_skills = NULL,
                                  context = NULL, n_skills = NULL) {
  item_list <- items |>
    glue::glue_data("- {item_id}: {text}") |>
    stringr::str_c(collapse = "\n")

  has_known <- !is.null(known_skills) && nrow(known_skills) > 0

  known_block <- if (has_known) {
    skill_list <- known_skills |>
      glue::glue_data("- {name}: {description}") |>
      stringr::str_c(collapse = "\n")

    stringr::str_c(
      "The following skills are already known. Use these wherever they apply.",
      "Only propose a new skill if an item genuinely requires something not covered by the known set.",
      "Mark each skill with is_new = FALSE if from this list, is_new = TRUE if newly proposed.",
      "",
      "Known skills:",
      skill_list,
      sep = "\n"
    )
  }

  parts <- c(
    "Analyze these test items and identify the distinct latent skills required to solve them.",
    if (!is.null(context)) glue("Context: {context}"),
    if (!is.null(n_skills) && !has_known) glue("Propose approximately {n_skills} skills."),
    known_block,
    if (!has_known) c(
      "Skills should be at a consistent level of granularity.",
      "Too broad (e.g. 'math') is useless. Too narrow (e.g. 'multiplying 3-digit numbers') overfits."
    ),
    "Each item may require multiple skills.",
    "",
    "Items:",
    item_list
  )

  stringr::str_c(parts, collapse = "\n")
}


#' @noRd
build_assignment_prompt <- function(items, taxonomy) {
  item_list <- items |>
    glue::glue_data("- {item_id}: {text}") |>
    stringr::str_c(collapse = "\n")

  skill_list <- taxonomy |>
    glue::glue_data("- {name}: {description}") |>
    stringr::str_c(collapse = "\n")

  glue("
    For each item, identify which of the following skills are required to solve it.
    An item can require multiple skills. Only assign a skill if it is genuinely needed.
    Use the exact skill names provided.

    Skills:
    {skill_list}

    Items:
    {item_list}
  ")
}


# Core functions ----------------------------------------------------------

#' Propose a skill taxonomy from item text
#'
#' @param chat An ellmer chat object.
#' @param items Tibble with `item_id` and `text`.
#' @param known_skills Optional tibble of known skills.
#' @param context Optional domain context string.
#' @param n_skills Optional target number of skills.
#' @return A tibble with `skill_id`, `name`, `description`, `is_new`.
#' @export
propose_taxonomy <- function(chat, items, known_skills = NULL,
                             context = NULL, n_skills = NULL) {
  prompt <- build_taxonomy_prompt(items, known_skills, context, n_skills)

  chat$chat_structured(
    prompt,
    type = ellmer::type_array(items = skill_type, description = "List of identified skills")
  ) |>
    dplyr::mutate(skill_id = stringr::str_c("skill_", dplyr::row_number()), .before = 1)
}


#' Assign skills to items
#'
#' @param chat An ellmer chat object.
#' @param items Tibble with `item_id` and `text`.
#' @param taxonomy Tibble from `propose_taxonomy()`.
#' @return A tibble with `item_id`, `skill_id`, `skill_name`.
#' @export
assign_skills <- function(chat, items, taxonomy) {
  prompt <- build_assignment_prompt(items, taxonomy)

  chat$chat_structured(
    prompt,
    type = ellmer::type_array(items = assignment_type, description = "Skill assignments per item")
  ) |>
    tidyr::unnest(skills) |>
    dplyr::rename(skill_name = skills) |>
    dplyr::left_join(taxonomy |> dplyr::select(skill_id, name), by = c("skill_name" = "name")) |>
    dplyr::select(item_id, skill_id, skill_name)
}


#' Propose skills for a set of items
#'
#' Two-stage pipeline: first proposes a taxonomy, then assigns skills to items.
#'
#' @param items Tibble with `item_id` and `text`.
#' @param known_skills Optional tibble of known skills.
#' @param context Optional domain context string.
#' @param n_skills Optional target number of skills.
#' @param chat Optional pre-configured chat object.
#' @param model LLM model name (default: claude-sonnet-4-20250514).
#' @return A `loading_structure` object.
#' @export
propose_skills <- function(items, known_skills = NULL, context = NULL,
                           n_skills = NULL, chat = NULL,
                           model = "claude-sonnet-4-20250514") {
  if (is.null(chat)) {
    chat <- ellmer::chat_anthropic(
      system_prompt = "You are an expert in cognitive task analysis and psychometrics.",
      model = model
    )
  }

  taxonomy <- propose_taxonomy(chat, items, known_skills, context, n_skills)
  assignments <- assign_skills(chat, items, taxonomy)

  loading_structure(taxonomy, assignments, items)
}
```

- [ ] **Step 3: Run tests**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-skill-proposer.R')"`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add R/skill-proposer.R tests/testthat/test-skill-proposer.R
git commit -m "Refactor skill proposer for package: roxygen docs, no library() calls, returns loading_structure"
```

---

## Task 17: Structure Optimizer (Outer Loop)

**Files:**
- Create: `R/structure-optimizer.R`
- Create: `tests/testthat/test-structure-optimizer.R`

- [ ] **Step 1: Write the failing tests**

`tests/testthat/test-structure-optimizer.R`:

```r
test_that("run_iteration returns iteration_result", {
  skip_if_no_cmdstan()
  skip_on_cran()

  config <- make_test_config()
  sim <- simulate_responses(config, n_students = 100, seed = 42)

  result <- run_iteration(
    sim$responses, config,
    chains = 2, iter_warmup = 200, iter_sampling = 200
  )

  expect_s3_class(result, "iteration_result")
  expect_true(result$identifiable)
  expect_s3_class(result$fit_result, "fit_result")
})

test_that("run_iteration rejects non-identifiable config", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()
  structure <- loading_structure(taxonomy, assignments, items)

  # Manually break identifiability
  structure$lambda_mask[, "skill_3"] <- FALSE
  structure$lambda_mask["item_4", "skill_3"] <- TRUE

  config <- structure(
    list(spec = model_spec(), structure = structure, edge_prior = NULL),
    class = "model_config"
  )

  Y <- matrix(sample(0:1, 100 * 8, replace = TRUE), 100, 8)
  rownames(Y) <- paste0("s_", 1:100)
  colnames(Y) <- paste0("item_", 1:8)
  rd <- response_data(Y)

  result <- run_iteration(rd, config)

  expect_s3_class(result, "iteration_result")
  expect_false(result$identifiable)
  expect_null(result$fit_result)
})

test_that("log_iteration writes JSONL", {
  tmp <- tempfile(fileext = ".jsonl")
  on.exit(unlink(tmp))

  log_iteration(
    iter = 1,
    elpd = -150.3,
    improved = TRUE,
    diagnostics = tibble::tibble(
      metric = "n_divergences", value = 0, status = "ok"
    ),
    config_label = "linear/independent/single/basic",
    rationale = "initial model",
    log_file = tmp
  )

  lines <- readLines(tmp)
  expect_length(lines, 1)
  parsed <- jsonlite::fromJSON(lines[1])
  expect_equal(parsed$iter, 1)
  expect_equal(parsed$elpd, -150.3)
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-structure-optimizer.R')"`
Expected: All tests FAIL

- [ ] **Step 3: Implement R/structure-optimizer.R**

```r
#' Run one iteration of the optimization loop
#'
#' Checks identifiability, optionally runs SBC, then fits the model
#' and computes LOO.
#'
#' @param responses A `response_data` object.
#' @param config A `model_config` object.
#' @param previous_results List of previous `iteration_result` objects.
#' @param skip_sbc If `TRUE`, skip SBC check (default `TRUE` for speed).
#' @param ... Additional arguments passed to `fit_model()`.
#' @return An S3 object of class `iteration_result`.
#' @export
run_iteration <- function(responses, config,
                          previous_results = list(),
                          skip_sbc = TRUE,
                          ...) {
  # Pre-check identifiability
  id_check <- check_identifiability(config)
  if (!id_check$passed) {
    return(structure(
      list(
        config = config,
        fit_result = NULL,
        identifiable = FALSE,
        sbc_passed = NA,
        elpd = NA_real_,
        problems = id_check$problems
      ),
      class = "iteration_result"
    ))
  }

  # Optional SBC
  sbc_passed <- NA
  if (!skip_sbc) {
    sbc <- run_sbc(config, n_sims = 20, n_students = 100)
    sbc_passed <- sbc$passed
    if (!sbc$passed) {
      return(structure(
        list(
          config = config,
          fit_result = NULL,
          identifiable = TRUE,
          sbc_passed = FALSE,
          elpd = NA_real_,
          problems = sbc$problems
        ),
        class = "iteration_result"
      ))
    }
  }

  # Fit model
  fit_result <- fit_model(responses, config, ...)

  elpd <- fit_result$loo$estimates["elpd_loo", "Estimate"]

  structure(
    list(
      config = config,
      fit_result = fit_result,
      identifiable = TRUE,
      sbc_passed = sbc_passed,
      elpd = elpd,
      problems = character(0)
    ),
    class = "iteration_result"
  )
}


#' @export
print.iteration_result <- function(x, ...) {
  if (!x$identifiable) {
    cli_inform("{.cls iteration_result}: rejected (not identifiable)")
  } else if (identical(x$sbc_passed, FALSE)) {
    cli_inform("{.cls iteration_result}: rejected (SBC failed)")
  } else {
    cli_inform("{.cls iteration_result}: ELPD = {round(x$elpd, 1)}")
  }
  invisible(x)
}


#' Run the full optimization loop
#'
#' Iteratively fits models, compares by LOO, and uses an LLM to
#' propose refined configurations.
#'
#' @param responses A `response_data` object.
#' @param items Tibble with `item_id` and `text`.
#' @param initial_config A `model_config` to start from. If `NULL`,
#'   runs `propose_skills()` to create one.
#' @param max_iter Maximum number of iterations.
#' @param patience Stop after this many consecutive non-improving iterations.
#' @param skip_sbc If `TRUE`, skip SBC checks for speed.
#' @param chat Optional pre-configured ellmer chat object.
#' @param model LLM model name for reflection.
#' @param log_file Optional file path for JSONL logging.
#' @param ... Additional arguments passed to `fit_model()`.
#' @return An S3 object of class `optimization_result`.
#' @export
optimize_structure <- function(responses, items,
                               initial_config = NULL,
                               max_iter = 10,
                               patience = 3,
                               interactive = FALSE,
                               skip_sbc = TRUE,
                               chat = NULL,
                               model = "claude-sonnet-4-20250514",
                               log_file = NULL,
                               ...) {
  if (is.null(chat)) {
    chat <- ellmer::chat_anthropic(
      system_prompt = build_reflection_system_prompt(),
      model = model
    )
  }

  # Initial config from skill proposer if not provided
  if (is.null(initial_config)) {
    cli_inform("No initial config provided; running skill proposer...")
    structure <- propose_skills(items, chat = chat)
    initial_config <- model_config(model_spec(), structure)
  }

  history <- list()
  best_elpd <- -Inf
  best_result <- NULL
  non_improving <- 0

  for (iter in seq_len(max_iter)) {
    config <- if (iter == 1) initial_config else history[[iter - 1]]$proposed_config

    cli_inform("Iteration {iter}/{max_iter}: {config$spec$measurement}/{config$spec$structural}/{config$spec$population}/{config$spec$item}")

    # Run iteration
    iter_result <- run_iteration(responses, config, skip_sbc = skip_sbc, ...)

    # Check improvement
    improved <- FALSE
    if (!is.na(iter_result$elpd) && iter_result$elpd > best_elpd) {
      best_elpd <- iter_result$elpd
      best_result <- iter_result
      improved <- TRUE
      non_improving <- 0
    } else {
      non_improving <- non_improving + 1
    }

    # Log
    config_label <- paste(
      config$spec$measurement, config$spec$structural,
      config$spec$population, config$spec$item,
      sep = "/"
    )

    if (!is.null(log_file)) {
      log_iteration(
        iter = iter,
        elpd = iter_result$elpd,
        improved = improved,
        diagnostics = if (!is.null(iter_result$fit_result)) iter_result$fit_result$diagnostics else NULL,
        config_label = config_label,
        rationale = if (iter == 1) "initial" else "LLM refinement",
        log_file = log_file
      )
    }

    cli_inform("  ELPD = {round(iter_result$elpd, 1)}, improved = {improved}, patience = {patience - non_improving}")

    # Check stopping
    if (non_improving >= patience) {
      cli_inform("Stopping: {patience} consecutive non-improving iterations.")
      break
    }

    # Propose refinement for next iteration
    if (iter < max_iter && !is.null(iter_result$fit_result)) {
      previous_configs <- lapply(history, function(h) h$config)
      proposed <- tryCatch(
        propose_refinement(
          chat, iter_result$fit_result, items,
          previous_configs = previous_configs
        ),
        error = function(e) {
          cli_warn("LLM refinement failed: {conditionMessage(e)}")
          NULL
        }
      )

      # Interactive mode: present proposal and wait for approval
      if (interactive && !is.null(proposed)) {
        cli_inform(c(
          "",
          "Proposed refinement:",
          "*" = "{proposed$spec$measurement}/{proposed$spec$structural}/{proposed$spec$population}/{proposed$spec$item}",
          "*" = "Skills: {paste(proposed$structure$taxonomy$name, collapse = ', ')}"
        ))
        response <- readline("Accept (a), modify (m), or reject (r)? ")
        if (response == "r") {
          proposed <- NULL
        } else if (response == "m") {
          cli_inform("Modify the config manually and pass it back.")
          # In interactive mode, user can modify the proposed config
          # For now, we just keep the proposal
        }
      }

      iter_result$proposed_config <- proposed %||% config
    } else {
      iter_result$proposed_config <- config
    }

    history[[iter]] <- iter_result
  }

  structure(
    list(
      best = best_result,
      history = history,
      n_iterations = length(history),
      best_elpd = best_elpd
    ),
    class = "optimization_result"
  )
}


#' @export
print.optimization_result <- function(x, ...) {
  cli_inform(c(
    "{.cls optimization_result}: {x$n_iterations} iterations",
    "*" = "Best ELPD: {round(x$best_elpd, 1)}",
    "*" = "Best config: {x$best$config$spec$measurement}/{x$best$config$spec$structural}/{x$best$config$spec$population}/{x$best$config$spec$item}"
  ))
  invisible(x)
}


#' Log one iteration to JSONL
#' @noRd
log_iteration <- function(iter, elpd, improved, diagnostics,
                          config_label, rationale, log_file) {
  entry <- list(
    iter = iter,
    elpd = elpd,
    improved = improved,
    config = config_label,
    rationale = rationale,
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
  )

  if (!is.null(diagnostics)) {
    for (i in seq_len(nrow(diagnostics))) {
      entry[[diagnostics$metric[i]]] <- diagnostics$value[i]
    }
  }

  line <- jsonlite::toJSON(entry, auto_unbox = TRUE)
  cat(line, "\n", file = log_file, append = TRUE)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `Rscript -e "devtools::load_all(); testthat::test_file('tests/testthat/test-structure-optimizer.R')"`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add R/structure-optimizer.R tests/testthat/test-structure-optimizer.R
git commit -m "Add structure optimizer: iteration loop, LLM refinement, JSONL logging, patience stopping"
```

---

## Task 18: Update .gitignore and .Rbuildignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Update .gitignore for R package**

Add R-package-specific patterns:

```
# R package artifacts
*.Rcheck/
*.tar.gz
src/*.o
src/*.so
src/*.dll
man/
inst/doc/

# Stan compilation cache
*.stan.exe
*.stan.hpp
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore .Rbuildignore
git commit -m "Update gitignore and Rbuildignore for R package structure"
```

---

## Task 19: Run Full Test Suite and Fix Issues

- [ ] **Step 1: Run devtools::check()**

Run: `Rscript -e "devtools::check()"`

This will catch: missing imports, documentation issues, test failures, NAMESPACE problems.

- [ ] **Step 2: Generate NAMESPACE with roxygen2**

Run: `Rscript -e "devtools::document()"`

- [ ] **Step 3: Fix any issues found by check**

Common issues to fix:
- Missing `@importFrom` directives
- Undocumented exported functions
- Missing package dependencies in DESCRIPTION
- NAMESPACE conflicts

- [ ] **Step 4: Run full test suite**

Run: `Rscript -e "devtools::test()"`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Fix R CMD check issues, generate NAMESPACE with roxygen2"
```

---

## Task 20: End-to-End Example

**Files:**
- Modify: `examples/propose-skills.R`
- Create: `examples/optimize-structure.R`

- [ ] **Step 1: Update examples/propose-skills.R to use package**

```r
library(autoskill)
library(tibble)

items <- tibble(
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

# Open discovery
structure <- propose_skills(
  items,
  context = "Middle school mathematics assessment",
  n_skills = 3
)

print(structure)
```

- [ ] **Step 2: Create examples/optimize-structure.R**

```r
library(autoskill)
library(tibble)

# --- 1. Define items ---
items <- tibble(
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

# --- 2. Set up a known "true" model (correlated 3-factor) ---
taxonomy <- tibble(
  skill_id = c("skill_1", "skill_2", "skill_3"),
  name = c("Linear equations", "Fraction arithmetic", "Equation setup"),
  description = c(
    "Solving and rearranging equations",
    "Operations with fractions and decimals",
    "Translating word problems into equations"
  ),
  is_new = c(TRUE, TRUE, TRUE)
)

assignments <- tibble(
  item_id = c("item_1", "item_2", "item_3", "item_4", "item_4",
              "item_5", "item_6", "item_7", "item_7", "item_8"),
  skill_id = c("skill_1", "skill_1", "skill_2", "skill_1", "skill_3",
               "skill_1", "skill_2", "skill_1", "skill_3", "skill_2"),
  skill_name = c("Linear equations", "Linear equations", "Fraction arithmetic",
                 "Linear equations", "Equation setup",
                 "Linear equations", "Fraction arithmetic",
                 "Linear equations", "Equation setup", "Fraction arithmetic")
)

true_structure <- loading_structure(taxonomy, assignments, items)
true_config <- model_config(
  model_spec(structural = "correlated"),
  true_structure
)

# --- 3. Simulate data from the true model ---
sim <- simulate_responses(true_config, n_students = 500, seed = 42)

# --- 4. Start optimization from a simpler model (independent) ---
initial_config <- model_config(model_spec(structural = "independent"), true_structure)

# --- 5. Run the optimizer ---
result <- optimize_structure(
  sim$responses,
  items,
  initial_config = initial_config,
  max_iter = 5,
  patience = 3,
  log_file = "optimization-log.jsonl"
)

print(result)
```

- [ ] **Step 3: Commit**

```bash
git add examples/propose-skills.R examples/optimize-structure.R
git commit -m "Add end-to-end examples for skill proposer and structure optimizer"
```
