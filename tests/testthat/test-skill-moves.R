test_that("propose_skill_local_move returns a model_config", {
  config <- make_test_config()
  withr::with_seed(1, {
    out <- propose_skill_local_move(config)
  })
  expect_s3_class(out, "model_config")
})

test_that("flip_random_lambda either flips one entry or returns unchanged", {
  config <- make_test_config()
  for (seed in 1:30) {
    withr::with_seed(seed, {
      out <- flip_random_lambda(config)
    })
    expect_s3_class(out, "model_config")
    n_diff <- sum(out$structure$lambda_mask != config$structure$lambda_mask)
    expect_true(n_diff %in% c(0L, 1L),
                info = sprintf("seed=%d gave n_diff=%d", seed, n_diff))
  }
})

test_that("flip_random_lambda preserves identifiability when it accepts", {
  config <- make_test_config()
  for (seed in 1:50) {
    withr::with_seed(seed, {
      out <- flip_random_lambda(config)
    })
    if (!identical(out$structure$lambda_mask, config$structure$lambda_mask)) {
      # The accepted move must be identifiable
      expect_true(check_identifiability(out)$passed,
                  info = sprintf("flipped result at seed=%d not identifiable",
                                 seed))
    }
  }
})

test_that("swap_random_block changes exactly one of measurement/item/link", {
  config <- make_test_config()
  for (seed in 1:20) {
    withr::with_seed(seed, {
      out <- swap_random_block(config)
    })
    n_changed <- sum(c(
      out$spec$measurement != config$spec$measurement,
      out$spec$item != config$spec$item,
      out$spec$link != config$spec$link
    ))
    expect_true(n_changed %in% c(0L, 1L))
    # structural and population are restricted
    expect_equal(out$spec$structural, config$spec$structural)
    expect_equal(out$spec$population, config$spec$population)
  }
})

test_that("swap_random_block preserves loading_structure unchanged", {
  config <- make_test_config()
  withr::with_seed(7, {
    out <- swap_random_block(config)
  })
  expect_identical(out$structure, config$structure)
})

test_that("repeated propose_skill_local_move generates diverse structures", {
  config <- make_test_config()
  withr::with_seed(42, {
    masks <- vector("list", 30L)
    current <- config
    for (i in seq_len(30L)) {
      current <- propose_skill_local_move(current)
      masks[[i]] <- current$structure$lambda_mask
    }
  })
  unique_count <- length(unique(masks))
  # 30 random moves should produce at least a handful of distinct masks
  expect_gt(unique_count, 3L)
})

test_that("skill_problem exposes propose_local_move", {
  rd <- make_fake_responses()
  problem <- skill_problem(items = make_test_items(), responses = rd)
  expect_true(is.function(problem$propose_local_move))
})

test_that("skill_problem$propose_local_move returns a model_config", {
  rd <- make_fake_responses()
  problem <- skill_problem(items = make_test_items(), responses = rd)
  config <- make_test_config()
  withr::with_seed(11, {
    out <- problem$propose_local_move(config)
  })
  expect_s3_class(out, "model_config")
})

test_that("mask_to_assignments round-trips through loading_structure", {
  taxonomy <- make_test_taxonomy()
  items <- make_test_items()
  assignments <- make_test_assignments()
  struc <- loading_structure(taxonomy, assignments, items)

  # Convert mask back to assignments and rebuild
  rebuilt_assignments <- mask_to_assignments(struc$lambda_mask, taxonomy)
  rebuilt <- loading_structure(taxonomy, rebuilt_assignments, items)

  expect_equal(rebuilt$lambda_mask, struc$lambda_mask)
})
