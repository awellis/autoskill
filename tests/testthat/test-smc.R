# Helper from test-causal-problem.R is not visible here; redefine.
simulate_chain_dag <- function(n = 200, seed = 1) {
  withr::with_seed(seed, {
    A <- rnorm(n)
    B <- 0.7 * A + rnorm(n, sd = 0.5)
    C <- 0.3 * A + 0.5 * B + rnorm(n, sd = 0.4)
  })
  data.frame(A = A, B = B, C = C)
}

test_that("normalize_log_weights produces weights summing to 1", {
  lw <- c(-100, -101, -99)
  norm <- normalize_log_weights(lw)
  expect_equal(sum(exp(norm)), 1, tolerance = 1e-10)
})

test_that("normalize_log_weights handles extreme values without overflow", {
  lw <- c(1e6, 1e6 - 1, 1e6 - 2)
  norm <- normalize_log_weights(lw)
  expect_equal(sum(exp(norm)), 1, tolerance = 1e-10)
  expect_true(all(is.finite(norm)))
})

test_that("normalize_log_weights of uniform input returns uniform", {
  lw <- rep(0, 4)
  norm <- normalize_log_weights(lw)
  expect_equal(exp(norm), rep(0.25, 4), tolerance = 1e-10)
})

test_that("effective_sample_size at uniform weights equals K", {
  lw <- normalize_log_weights(rep(0, 10))
  expect_equal(effective_sample_size(lw), 10, tolerance = 1e-10)
})

test_that("effective_sample_size at degenerate weights equals 1", {
  lw <- c(0, rep(-1e6, 9))
  lw <- normalize_log_weights(lw)
  expect_equal(effective_sample_size(lw), 1, tolerance = 1e-6)
})

test_that("systematic_resample returns K indices in [1, K]", {
  w <- normalize_log_weights(rnorm(20))
  idx <- systematic_resample(exp(w))
  expect_length(idx, 20L)
  expect_true(all(idx >= 1L & idx <= 20L))
})

test_that("systematic_resample concentrates on high-weight particles", {
  weights <- c(0.97, rep(0.001, 30))
  weights <- weights / sum(weights)
  withr::with_seed(1, {
    idx <- systematic_resample(weights)
  })
  # First particle should be sampled many times, others rarely
  expect_gt(sum(idx == 1L), 25)
})

test_that("optimize_structure_smc errors when propose_local_move missing", {
  bare_problem <- structure_problem(
    data = list(),
    propose_initial = function(...) "x",
    propose_refinement = function(...) "x",
    score = function(...) {
      list(loo = list(estimates = matrix(0,
        dimnames = list("elpd_loo", "Estimate"))))
    },
    log_prior = function(...) 0,
    validate = function(...) list(passed = TRUE, problems = character(0)),
    cache_key = function(...) "k",
    summarize_structure = function(...) "x",
    summarize_fit = function(...) list()
  )
  expect_error(
    optimize_structure_smc(bare_problem, n_particles = 4, n_steps = 2),
    "propose_local_move"
  )
})

test_that("optimize_structure_smc on causal_problem returns smc_result", {
  withr::with_seed(2, {
    dat <- simulate_chain_dag(n = 200)
    problem <- causal_problem(dat)
    result <- optimize_structure_smc(problem,
                                     n_particles = 8L, n_steps = 6L)
  })

  expect_s3_class(result, c("smc_result", "optimization_result"))
  expect_length(result$particles, 8L)
  expect_length(result$weights, 8L)
  expect_equal(sum(result$weights), 1, tolerance = 1e-8)
})

test_that("SMC on simulated chain DAG beats the empty graph", {
  withr::with_seed(3, {
    dat <- simulate_chain_dag(n = 300)
    problem <- causal_problem(dat)

    # Run SMC
    result <- optimize_structure_smc(problem,
                                     n_particles = 16L, n_steps = 10L)

    # Compare best ELPD to empty graph
    empty_elpd <- score_causal_dag(
      dat, empty_dag(c("A", "B", "C"))
    )$loo$estimates["elpd_loo", "Estimate"]
  })

  expect_gt(result$best$elpd, empty_elpd)
  # Best particle should be non-empty
  expect_gt(sum(result$best$structure$adj), 0L)
})

test_that("SMC particle weights sum to 1 throughout history", {
  withr::with_seed(5, {
    dat <- simulate_chain_dag(n = 100)
    problem <- causal_problem(dat)
    result <- optimize_structure_smc(problem,
                                     n_particles = 8L, n_steps = 4L)
  })

  for (h in result$history) {
    expect_equal(sum(exp(h$log_weights)), 1, tolerance = 1e-8)
  }
})

test_that("log_sum_exp matches the naive computation", {
  x <- c(-2, 1, 3, -10)
  expect_equal(log_sum_exp(x), log(sum(exp(x))), tolerance = 1e-12)
})

test_that("log_sum_exp is overflow-safe", {
  x <- c(1e6, 1e6 + 1)
  # Naive log(sum(exp(x))) would overflow; ours should not
  out <- log_sum_exp(x)
  expect_true(is.finite(out))
  expect_equal(out, 1e6 + log(1 + exp(1)), tolerance = 1e-9)
})

test_that("optimize_structure_smc rejects malformed schedule", {
  problem <- causal_problem(simulate_chain_dag(n = 50))
  expect_error(
    optimize_structure_smc(problem, n_steps = 4L,
                            schedule = c(0.1, 0.5, 1.0)),
    "length"
  )
  expect_error(
    optimize_structure_smc(problem, n_steps = 4L,
                            schedule = c(0.1, 0.5, 1.0, 1.0, 0.9)),
    "increase"
  )
  expect_error(
    optimize_structure_smc(problem, n_steps = 4L,
                            schedule = c(0, 0.5, 1.0, 1.0, 0.5)),
    "increase"
  )
})

test_that("optimize_structure_smc records the schedule on the result", {
  withr::with_seed(11, {
    problem <- causal_problem(simulate_chain_dag(n = 100))
    result <- optimize_structure_smc(problem, n_particles = 4L, n_steps = 5L)
  })
  expect_length(result$schedule, 6L)
  expect_equal(result$schedule[1], 0)
  expect_equal(result$schedule[6], 1)
})

test_that("structure_marginal weights features by particle weights", {
  # Hand-construct a minimal smc_result with two particles and known weights
  fake <- base::structure(
    list(
      particles = list(
        list(value = 1),
        list(value = 3)
      ),
      weights = c(0.25, 0.75),
      log_weights = log(c(0.25, 0.75)),
      schedule = c(0, 1)
    ),
    class = c("smc_result", "optimization_result")
  )
  out <- structure_marginal(fake, function(p) p$value)
  # 0.25 * 1 + 0.75 * 3 = 2.5
  expect_equal(out, 2.5)
})

test_that("structure_marginal works on matrix-valued features", {
  fake <- base::structure(
    list(
      particles = list(
        list(M = matrix(1, 2, 2)),
        list(M = matrix(3, 2, 2))
      ),
      weights = c(0.5, 0.5)
    ),
    class = c("smc_result", "optimization_result")
  )
  out <- structure_marginal(fake, function(p) p$M)
  expect_equal(out, matrix(2, 2, 2))
})

test_that("edge_marginals returns a K x K matrix in [0, 1]", {
  withr::with_seed(13, {
    problem <- causal_problem(simulate_chain_dag(n = 200))
    result <- optimize_structure_smc(problem,
                                      n_particles = 8L, n_steps = 6L)
  })

  P <- edge_marginals(result)
  expect_equal(dim(P), c(3L, 3L))
  expect_true(all(P >= 0 & P <= 1))
  expect_true(all(diag(P) == 0))  # diagonal forced to zero in causal_dag
})

test_that("edge_marginals on chain DAG concentrates on the truth", {
  # On A -> B -> C with confound A -> C, the truth has 3 edges (A,B), (A,C), (B,C).
  # SMC should give those three edges high marginal probability.
  withr::with_seed(17, {
    problem <- causal_problem(simulate_chain_dag(n = 500))
    result <- optimize_structure_smc(problem,
                                      n_particles = 16L, n_steps = 12L,
                                      n_mutations_per_step = 2L)
  })

  P <- edge_marginals(result)
  # The directed edges A -> B and B -> C in particular should be common
  # (these are the "spine" of the chain).
  expect_gt(P["A", "B"] + P["B", "C"], 0.5)
})

test_that("edge_marginals rejects non-causal_dag particles", {
  fake <- base::structure(
    list(
      particles = list(list(x = 1)),
      weights = 1
    ),
    class = c("smc_result", "optimization_result")
  )
  expect_error(edge_marginals(fake), "causal_dag")
})
