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
  from <- c("A", "B")
  to <- c("B", "C")
  nodes <- c("A", "B", "C")

  result <- topological_sort(from, to, nodes)

  expect_true(which(result == "A") < which(result == "B"))
  expect_true(which(result == "B") < which(result == "C"))
})

test_that("topological_sort handles diamond DAG", {
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
