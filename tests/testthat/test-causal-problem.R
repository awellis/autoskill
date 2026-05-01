# Helper: simulate from a known linear-Gaussian DAG
# A -> B -> C, A -> C (chain with a confound)
simulate_chain_dag <- function(n = 200, seed = 1) {
  withr::with_seed(seed, {
    A <- rnorm(n)
    B <- 0.7 * A + rnorm(n, sd = 0.5)
    C <- 0.3 * A + 0.5 * B + rnorm(n, sd = 0.4)
  })
  data.frame(A = A, B = B, C = C)
}

test_that("causal_dag rejects malformed adjacency", {
  expect_error(causal_dag("not a matrix"), "matrix")
  expect_error(causal_dag(matrix(1, 2, 3), variables = c("x", "y", "z")),
               "square")
  expect_error(causal_dag(matrix(2, 2, 2), variables = c("x", "y")),
               "0 or 1")
})

test_that("causal_dag forces zero diagonal", {
  adj <- matrix(c(1, 1, 0, 1), 2, 2)
  dag <- causal_dag(adj, c("a", "b"))
  expect_true(all(diag(dag$adj) == 0L))
})

test_that("empty_dag has zero edges", {
  dag <- empty_dag(c("x", "y", "z"))
  expect_equal(sum(dag$adj), 0L)
  expect_equal(length(dag$variables), 3L)
})

test_that("is_acyclic detects DAGs and cycles", {
  vars <- c("a", "b", "c")

  # Chain a -> b -> c: acyclic
  adj <- matrix(0L, 3, 3, dimnames = list(vars, vars))
  adj["a", "b"] <- 1L; adj["b", "c"] <- 1L
  expect_true(is_acyclic(causal_dag(adj, vars)))

  # 2-cycle a -> b -> a: cyclic
  adj <- matrix(0L, 3, 3, dimnames = list(vars, vars))
  adj["a", "b"] <- 1L; adj["b", "a"] <- 1L
  expect_false(is_acyclic(causal_dag(adj, vars)))

  # 3-cycle a -> b -> c -> a: cyclic
  adj <- matrix(0L, 3, 3, dimnames = list(vars, vars))
  adj["a", "b"] <- 1L; adj["b", "c"] <- 1L; adj["c", "a"] <- 1L
  expect_false(is_acyclic(causal_dag(adj, vars)))

  # Empty: vacuously acyclic
  expect_true(is_acyclic(empty_dag(vars)))
})

test_that("score_causal_dag returns fit_result with valid loo shape", {
  dat <- simulate_chain_dag(n = 100)
  dag <- empty_dag(c("A", "B", "C"))

  fit <- score_causal_dag(dat, dag)
  expect_s3_class(fit, "fit_result")
  expect_s3_class(fit$loo, "loo")
  expect_true(is.finite(fit$loo$estimates["elpd_loo", "Estimate"]))
  expect_equal(nrow(fit$loo$pointwise), 100L)
})

test_that("score_causal_dag prefers correct DAG over reversed edges", {
  dat <- simulate_chain_dag(n = 500, seed = 2)
  vars <- c("A", "B", "C")

  # True structure: A -> B, A -> C, B -> C
  true_adj <- matrix(0L, 3, 3, dimnames = list(vars, vars))
  true_adj["A", "B"] <- 1L
  true_adj["A", "C"] <- 1L
  true_adj["B", "C"] <- 1L
  true_dag <- causal_dag(true_adj, vars)

  empty <- empty_dag(vars)

  elpd_true <- score_causal_dag(dat, true_dag)$loo$estimates["elpd_loo", "Estimate"]
  elpd_empty <- score_causal_dag(dat, empty)$loo$estimates["elpd_loo", "Estimate"]

  # The true DAG should fit substantially better than the empty graph
  expect_gt(elpd_true, elpd_empty)
})

test_that("random_local_move returns an acyclic DAG", {
  dag <- empty_dag(c("a", "b", "c", "d", "e"))
  withr::with_seed(7, {
    for (i in 1:50) {
      dag <- random_local_move(dag)
      expect_true(is_acyclic(dag),
                  info = sprintf("non-acyclic after %d moves", i))
    }
  })
})

test_that("causal_problem constructs structure_problem with correct subclass", {
  dat <- simulate_chain_dag()
  problem <- causal_problem(dat)
  expect_s3_class(problem, c("causal_problem", "structure_problem"))
  expect_true(is_structure_problem(problem))
})

test_that("causal_problem$summarize_structure formats node and edge count", {
  dat <- simulate_chain_dag()
  problem <- causal_problem(dat)
  expect_equal(problem$summarize_structure(empty_dag(c("A", "B", "C"))),
               "DAG[3 nodes, 0 edges]")
})

test_that("causal_problem$validate rejects cycles", {
  dat <- simulate_chain_dag()
  problem <- causal_problem(dat)

  vars <- c("A", "B", "C")
  cyclic_adj <- matrix(0L, 3, 3, dimnames = list(vars, vars))
  cyclic_adj["A", "B"] <- 1L; cyclic_adj["B", "A"] <- 1L
  cyclic <- causal_dag(cyclic_adj, vars)

  v <- problem$validate(cyclic)
  expect_false(v$passed)
})

test_that("causal_problem auto-selects numeric columns", {
  mixed <- data.frame(
    A = rnorm(50), B = rnorm(50),
    label = letters[1:50],  # non-numeric, should be excluded
    C = rnorm(50)
  )
  problem <- causal_problem(mixed)
  expect_equal(problem$data$variables, c("A", "B", "C"))
})

test_that("optimize_structure on causal_problem recovers a non-empty DAG", {
  withr::with_seed(123, {
    dat <- simulate_chain_dag(n = 300)
    problem <- causal_problem(dat)
    result <- optimize_structure(problem,
                                  max_iter = 40, patience = 15)

    expect_s3_class(result, "optimization_result")
    expect_true(!is.null(result$best))
    expect_gt(sum(result$best$structure$adj), 0L)
    expect_gt(result$best_elpd,
              score_causal_dag(dat, empty_dag(c("A", "B", "C")))$loo$estimates["elpd_loo", "Estimate"])
  })
})
