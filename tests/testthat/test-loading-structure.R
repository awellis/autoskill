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
  expect_true(ls$lambda_mask["item_1", "skill_1"])
  expect_false(ls$lambda_mask["item_1", "skill_2"])
  expect_false(ls$lambda_mask["item_1", "skill_3"])
  expect_true(ls$lambda_mask["item_4", "skill_1"])
  expect_false(ls$lambda_mask["item_4", "skill_2"])
  expect_true(ls$lambda_mask["item_4", "skill_3"])
})

test_that("loading_structure rejects orphan items", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments() |>
    dplyr::filter(item_id != "item_8")
  expect_error(loading_structure(taxonomy, assignments, items), "item_8")
})

test_that("loading_structure rejects skills with fewer than 2 items", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments() |>
    dplyr::filter(!(skill_id == "skill_3" & item_id == "item_7"))
  expect_error(loading_structure(taxonomy, assignments, items), "skill_3")
})

test_that("n_loadings counts nonzero entries", {
  items <- make_test_items()
  taxonomy <- make_test_taxonomy()
  assignments <- make_test_assignments()
  ls <- loading_structure(taxonomy, assignments, items)
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
  ep <- edge_prior(from = c("skill_1", "skill_1"), to = c("skill_2", "skill_3"), prob = c(0.8, 0.6))
  expect_s3_class(ep, "edge_prior")
  expect_equal(nrow(ep$edges), 2)
})

test_that("edge_prior rejects probabilities outside [0, 1]", {
  expect_error(edge_prior(from = "A", to = "B", prob = 1.5), "prob")
})

test_that("edge_prior rejects self-loops", {
  expect_error(edge_prior(from = "A", to = "A", prob = 0.5), "self-loop")
})
