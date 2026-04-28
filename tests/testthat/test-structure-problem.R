noop_fn <- function(...) NULL
trivial_validate <- function(...) list(passed = TRUE, problems = character(0))

minimal_problem <- function(...) {
  structure_problem(
    data = list(),
    propose_initial = function(...) "structure",
    propose_refinement = noop_fn,
    score = function(...) list(loo = NULL),
    log_prior = function(...) 0,
    validate = trivial_validate,
    cache_key = function(...) "key",
    summarize_structure = function(...) "label",
    summarize_fit = function(...) list(),
    ...
  )
}

test_that("structure_problem constructs an inheriting object", {
  p <- minimal_problem()
  expect_s3_class(p, "structure_problem")
  expect_true(is_structure_problem(p))
})

test_that("structure_problem accepts subclasses via class arg", {
  p <- minimal_problem(class = "skill_problem")
  expect_s3_class(p, c("skill_problem", "structure_problem"))
})

test_that("structure_problem rejects non-function required slot", {
  expect_error(
    structure_problem(
      data = list(),
      propose_initial = "not a function",
      propose_refinement = noop_fn, score = noop_fn, log_prior = noop_fn,
      validate = trivial_validate, cache_key = noop_fn,
      summarize_structure = noop_fn, summarize_fit = noop_fn
    ),
    "must be a function"
  )
})

test_that("propose_local_move is optional and may be NULL", {
  p <- minimal_problem(propose_local_move = NULL)
  expect_null(p$propose_local_move)
})

test_that("propose_local_move accepts a function", {
  p <- minimal_problem(propose_local_move = function(...) "moved")
  expect_true(is.function(p$propose_local_move))
})

test_that("propose_local_move rejects non-function non-NULL value", {
  expect_error(
    minimal_problem(propose_local_move = 42),
    "must be NULL or a function"
  )
})

test_that("is_structure_problem returns FALSE for non-problem inputs", {
  expect_false(is_structure_problem(list()))
  expect_false(is_structure_problem(NULL))
  expect_false(is_structure_problem("structure_problem"))
})

test_that("print.structure_problem renders subclass and slot summary", {
  p <- minimal_problem(class = "skill_problem")
  out <- capture.output(print(p))
  expect_true(any(grepl("skill_problem", out)))
  for (slot in c("propose_initial", "propose_refinement", "score",
                 "log_prior", "validate", "cache_key",
                 "summarize_structure", "summarize_fit",
                 "propose_local_move")) {
    expect_true(any(grepl(slot, out)))
  }
})

test_that("print marks propose_local_move as present when provided", {
  p_with <- minimal_problem(propose_local_move = function(...) NULL)
  p_without <- minimal_problem()
  out_with <- capture.output(print(p_with))
  out_without <- capture.output(print(p_without))
  expect_true(any(grepl("\\+ propose_local_move", out_with)))
  expect_true(any(grepl("- propose_local_move", out_without)))
})
