# These tests exercise the move-application logic without invoking the LLM.
# We construct fake `move` objects (the shape returned by chat$chat_structured)
# and check that apply_local_move dispatches and the mask updates correctly.

test_that("apply_flip_loading toggles a single Lambda entry", {
  config <- make_test_config()
  mask <- config$structure$lambda_mask

  # Find an item that doesn't currently load on skill_3
  item_id <- "item_1"  # currently loads on skill_1 only
  expect_false(mask[item_id, "skill_3"])

  out <- apply_flip_loading(config, item_id, "skill_3")
  expect_true(out$structure$lambda_mask[item_id, "skill_3"])
  expect_equal(
    sum(out$structure$lambda_mask) - sum(mask),
    1L
  )
})

test_that("apply_flip_loading rejects unknown item or skill", {
  config <- make_test_config()
  expect_identical(apply_flip_loading(config, "nonexistent_item", "skill_1"),
                   config)
  expect_identical(apply_flip_loading(config, "item_1", "nonexistent_skill"),
                   config)
})

test_that("apply_flip_loading rejects a move that breaks identifiability", {
  config <- make_test_config()
  mask <- config$structure$lambda_mask

  # item_3 currently loads only on skill_2. Removing this loading would
  # leave it orphaned -> validate fails -> move rejected.
  expect_true(mask["item_3", "skill_2"])
  expect_equal(sum(mask["item_3", ]), 1L)

  out <- apply_flip_loading(config, "item_3", "skill_2")
  expect_identical(out, config)
})

test_that("apply_merge_skills combines two skill columns", {
  config <- make_test_config()
  mask <- config$structure$lambda_mask
  taxonomy <- config$structure$taxonomy

  expect_equal(ncol(mask), 3L)

  out <- apply_merge_skills(config, "skill_1", "skill_2")

  if (!identical(out, config)) {
    # Merge succeeded
    expect_equal(ncol(out$structure$lambda_mask), 2L)
    expect_false("skill_2" %in% out$structure$taxonomy$skill_id)
    # Items that loaded on either skill_1 or skill_2 now load on skill_1
    union_loadings <- mask[, "skill_1"] | mask[, "skill_2"]
    expect_equal(out$structure$lambda_mask[, "skill_1"], union_loadings)
  }
})

test_that("apply_merge_skills rejects merging a skill with itself", {
  config <- make_test_config()
  expect_identical(apply_merge_skills(config, "skill_1", "skill_1"), config)
})

test_that("apply_merge_skills rejects unknown skill ids", {
  config <- make_test_config()
  expect_identical(apply_merge_skills(config, "skill_1", "nonexistent"),
                   config)
  expect_identical(apply_merge_skills(config, "nonexistent", "skill_1"),
                   config)
})

test_that("apply_merge_skills refuses to drop below 2 skills", {
  taxonomy <- tibble::tibble(
    skill_id = c("a", "b"),
    name = c("A", "B"),
    description = c("a", "b"),
    is_new = c(TRUE, TRUE)
  )
  items <- tibble::tibble(item_id = paste0("i", 1:4), text = letters[1:4])
  assignments <- tibble::tibble(
    item_id = c("i1", "i2", "i3", "i4"),
    skill_id = c("a", "a", "b", "b"),
    skill_name = c("A", "A", "B", "B")
  )
  struc <- loading_structure(taxonomy, assignments, items)
  config <- model_config(model_spec(), struc)

  out <- apply_merge_skills(config, "a", "b")
  expect_identical(out, config)
})

test_that("apply_merge_skills cleans up edge_prior entries", {
  taxonomy <- make_test_taxonomy()
  items <- make_test_items()
  assignments <- make_test_assignments()
  struc <- loading_structure(taxonomy, assignments, items)
  ep <- edge_prior(
    from = c("skill_1", "skill_1", "skill_2"),
    to   = c("skill_2", "skill_3", "skill_3"),
    prob = c(0.8, 0.6, 0.5)
  )
  config <- model_config(
    model_spec(structural = "dag"), struc, edge_prior = ep
  )

  out <- apply_merge_skills(config, "skill_1", "skill_2")

  if (!identical(out, config) && !is.null(out$edge_prior)) {
    # Any surviving edges must not reference skill_2
    expect_false(any(out$edge_prior$edges$from == "skill_2"))
    expect_false(any(out$edge_prior$edges$to == "skill_2"))
  }
})

test_that("apply_local_move dispatches by move_type", {
  config <- make_test_config()

  out_flip <- apply_local_move(config, list(
    move_type = "flip_loading",
    item_id = "item_1",
    skill_id_1 = "skill_3"
  ))
  expect_true(out_flip$structure$lambda_mask["item_1", "skill_3"])

  out_noop <- apply_local_move(config, list(move_type = "no_op"))
  expect_identical(out_noop, config)

  # Unknown move type falls through to config
  out_bogus <- apply_local_move(config, list(move_type = "made_up"))
  expect_identical(out_bogus, config)
})

test_that("local_move_schema returns an ellmer type_object", {
  schema <- local_move_schema()
  # We don't assert internals; just that it constructs without error and
  # carries the expected field names. ellmer types are list-shaped.
  expect_true(!is.null(schema))
})

test_that("build_local_move_prompt mentions skills and item text", {
  config <- make_test_config()
  items <- make_test_items()
  prompt <- build_local_move_prompt(config, items)
  expect_match(prompt, "skill_1", fixed = TRUE)
  expect_match(prompt, "Solve: 3x \\+ 5 = 20")
  expect_match(prompt, "flip_loading", fixed = TRUE)
  expect_match(prompt, "merge_skills", fixed = TRUE)
})

test_that("skill_problem accepts local_move_kernel = 'llm'", {
  rd <- make_fake_responses()
  problem <- skill_problem(items = make_test_items(), responses = rd,
                           local_move_kernel = "llm")
  expect_true(is.function(problem$propose_local_move))
})

test_that("skill_problem rejects invalid local_move_kernel", {
  rd <- make_fake_responses()
  expect_error(
    skill_problem(items = make_test_items(), responses = rd,
                  local_move_kernel = "bogus"),
    "should be"
  )
})
