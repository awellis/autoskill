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
  spec <- model_spec(measurement = "interaction", structural = "dag",
                     population = "grouped", item = "slip_guess", link = "probit")
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
