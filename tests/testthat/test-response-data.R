test_that("response_data accepts a wide matrix", {
  Y <- matrix(sample(0:1, 100 * 8, replace = TRUE), nrow = 100, ncol = 8,
              dimnames = list(paste0("s_", 1:100), paste0("item_", 1:8)))
  rd <- response_data(Y)
  expect_s3_class(rd, "response_data")
  expect_equal(rd$n_students, 100L)
  expect_equal(rd$n_items, 8L)
  expect_equal(dim(rd$Y), c(100, 8))
})

test_that("response_data accepts a long tibble", {
  long <- tidyr::crossing(student_id = paste0("s_", 1:50), item_id = paste0("item_", 1:6)) |>
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
