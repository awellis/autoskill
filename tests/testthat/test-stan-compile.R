test_that("compile_model compiles Stan code", {
  skip_if_no_cmdstan()
  config <- make_test_config()
  model <- compile_model(config)
  expect_true(inherits(model, "CmdStanModel"))
})

test_that("compile_model caches by content hash", {
  skip_if_no_cmdstan()
  config <- make_test_config()
  t1 <- system.time(compile_model(config, force = TRUE))
  t2 <- system.time(compile_model(config))
  expect_true(t2["elapsed"] < t1["elapsed"])
})

test_that("all 64 block combinations compile", {
  skip_if_no_cmdstan()
  skip_on_cran()

  specs <- all_valid_specs()
  failures <- character(0)

  for (i in seq_len(nrow(specs))) {
    row <- specs[i, ]
    label <- paste(row$measurement, row$structural, row$population, row$item, row$link, sep = "/")

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
