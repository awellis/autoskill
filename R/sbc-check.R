#' Quick identifiability pre-check
#' @param config A `model_config` object.
#' @return A list with `$passed` (logical) and `$problems` (character vector).
#' @export
check_identifiability <- function(config) {
  problems <- character(0)
  mask <- config$structure$lambda_mask

  items_per_skill <- colSums(mask)
  thin <- names(items_per_skill[items_per_skill < 2])
  if (length(thin) > 0)
    problems <- c(problems, paste0("Skills with < 2 items (not identifiable): ", paste(thin, collapse = ", ")))

  skills_per_item <- rowSums(mask)
  orphans <- names(skills_per_item[skills_per_item == 0])
  if (length(orphans) > 0)
    problems <- c(problems, paste0("Items with no skills: ", paste(orphans, collapse = ", ")))

  patterns <- apply(mask, 2, paste, collapse = "")
  dups <- names(which(duplicated(patterns)))
  if (length(dups) > 0)
    problems <- c(problems, paste0("Duplicate loading patterns (not identifiable): ", paste(dups, collapse = ", ")))

  if (config$spec$structural == "dag" && !is.null(config$edge_prior)) {
    ep <- config$edge_prior
    skill_ids <- colnames(mask)
    if (!validate_dag(ep$edges$from, ep$edges$to, skill_ids))
      problems <- c(problems, "DAG contains a cycle.")
  }

  list(passed = length(problems) == 0, problems = problems)
}

#' Run simulation-based calibration
#' @param config A `model_config` object.
#' @param n_sims Number of SBC simulations.
#' @param n_students Number of students per simulation.
#' @param ... Additional arguments passed to sampling.
#' @return An S3 object of class `sbc_result`.
#' @export
run_sbc <- function(config, n_sims = 100, n_students = 200, ...) {
  if (!requireNamespace("SBC", quietly = TRUE))
    cli_abort("Package {.pkg SBC} is required. Install with: {.code install.packages('SBC')}")

  id_check <- check_identifiability(config)
  if (!id_check$passed)
    return(base::structure(list(passed = FALSE, problems = id_check$problems, rank_stats = NULL), class = "sbc_result"))

  gen <- sbc_generator(config, n_students = n_students)
  model <- compile_model(config)

  datasets <- SBC::SBC_generator_function(gen, N = n_sims)
  backend <- SBC::SBC_backend_cmdstan_sample(model, ...)
  results <- SBC::compute_SBC(datasets, backend)

  passed <- !any(results$stats$z_score_warning)

  base::structure(
    list(passed = passed,
         problems = if (!passed) "SBC rank statistics show calibration issues" else character(0),
         rank_stats = results),
    class = "sbc_result"
  )
}

#' @export
print.sbc_result <- function(x, ...) {
  status <- if (x$passed) "PASSED" else "FAILED"
  cat(sprintf("<sbc_result>: %s\n", status))
  if (length(x$problems) > 0) cat(sprintf("  ! %s\n", x$problems))
  invisible(x)
}
