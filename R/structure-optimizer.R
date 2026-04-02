#' Run one iteration of the optimization loop
#' @param responses A `response_data` object.
#' @param config A `model_config` object.
#' @param previous_results List of previous iteration_results.
#' @param skip_sbc If TRUE, skip SBC check.
#' @param ... Additional arguments passed to `fit_model()`.
#' @return An S3 object of class `iteration_result`.
#' @export
run_iteration <- function(responses, config, previous_results = list(),
                          skip_sbc = TRUE, ...) {
  id_check <- check_identifiability(config)
  if (!id_check$passed) {
    return(base::structure(
      list(config = config, fit_result = NULL, identifiable = FALSE,
           sbc_passed = NA, elpd = NA_real_, problems = id_check$problems),
      class = "iteration_result"
    ))
  }

  sbc_passed <- NA
  if (!skip_sbc) {
    sbc <- run_sbc(config, n_sims = 20, n_students = 100)
    sbc_passed <- sbc$passed
    if (!sbc$passed) {
      return(base::structure(
        list(config = config, fit_result = NULL, identifiable = TRUE,
             sbc_passed = FALSE, elpd = NA_real_, problems = sbc$problems),
        class = "iteration_result"
      ))
    }
  }

  fit_result <- fit_model(responses, config, ...)
  elpd <- fit_result$loo$estimates["elpd_loo", "Estimate"]

  base::structure(
    list(config = config, fit_result = fit_result, identifiable = TRUE,
         sbc_passed = sbc_passed, elpd = elpd, problems = character(0)),
    class = "iteration_result"
  )
}

#' @export
print.iteration_result <- function(x, ...) {
  if (!x$identifiable) {
    cat("<iteration_result>: rejected (not identifiable)\n")
  } else if (identical(x$sbc_passed, FALSE)) {
    cat("<iteration_result>: rejected (SBC failed)\n")
  } else {
    cat(sprintf("<iteration_result>: ELPD = %.1f\n", x$elpd))
  }
  invisible(x)
}

#' Run the full optimization loop
#' @param responses A `response_data` object.
#' @param items Tibble with `item_id` and `text`.
#' @param initial_config A `model_config` to start from.
#' @param max_iter Maximum iterations.
#' @param patience Stop after N consecutive non-improving.
#' @param interactive If TRUE, pause for approval after each iteration.
#' @param skip_sbc Skip SBC checks.
#' @param chat An ellmer chat object.
#' @param model LLM model name.
#' @param log_file Optional JSONL log file path.
#' @param ... Additional arguments passed to `fit_model()`.
#' @return An S3 object of class `optimization_result`.
#' @export
optimize_structure <- function(responses, items, initial_config = NULL,
                               max_iter = 10, patience = 3, interactive = FALSE,
                               skip_sbc = TRUE, chat = NULL,
                               model = "claude-sonnet-4-20250514",
                               log_file = NULL, ...) {
  if (is.null(chat)) {
    chat <- ellmer::chat_anthropic(
      system_prompt = build_reflection_system_prompt(),
      model = model
    )
  }

  if (is.null(initial_config)) {
    cli_inform("No initial config provided; running skill proposer...")
    struc <- propose_skills(items, chat = chat)
    initial_config <- model_config(model_spec(), struc)
  }

  history <- list()
  best_elpd <- -Inf
  best_result <- NULL
  non_improving <- 0

  for (iter in seq_len(max_iter)) {
    config <- if (iter == 1) initial_config else history[[iter - 1]]$proposed_config
    config_label <- paste(config$spec$measurement, config$spec$structural,
                          config$spec$population, config$spec$item, sep = "/")
    cli_inform("Iteration {iter}/{max_iter}: {config_label}")

    iter_result <- run_iteration(responses, config, skip_sbc = skip_sbc, ...)

    improved <- FALSE
    if (!is.na(iter_result$elpd) && iter_result$elpd > best_elpd) {
      best_elpd <- iter_result$elpd
      best_result <- iter_result
      improved <- TRUE
      non_improving <- 0
    } else {
      non_improving <- non_improving + 1
    }

    if (!is.null(log_file)) {
      log_iteration(
        iter = iter, elpd = iter_result$elpd, improved = improved,
        diagnostics = if (!is.null(iter_result$fit_result)) iter_result$fit_result$diagnostics else NULL,
        config_label = config_label,
        rationale = if (iter == 1) "initial" else "LLM refinement",
        log_file = log_file
      )
    }

    cli_inform("  ELPD = {round(iter_result$elpd, 1)}, improved = {improved}, patience = {patience - non_improving}")

    if (non_improving >= patience) {
      cli_inform("Stopping: {patience} consecutive non-improving iterations.")
      break
    }

    if (iter < max_iter && !is.null(iter_result$fit_result)) {
      previous_configs <- lapply(history, function(h) h$config)
      proposed <- tryCatch(
        propose_refinement(chat, iter_result$fit_result, items, previous_configs = previous_configs),
        error = function(e) { cli_warn("LLM refinement failed: {conditionMessage(e)}"); NULL }
      )

      if (interactive && !is.null(proposed)) {
        cat(sprintf("\nProposed: %s/%s/%s/%s\n",
            proposed$spec$measurement, proposed$spec$structural,
            proposed$spec$population, proposed$spec$item))
        cat(sprintf("Skills: %s\n", paste(proposed$structure$taxonomy$name, collapse = ", ")))
        response <- readline("Accept (a), modify (m), or reject (r)? ")
        if (response == "r") proposed <- NULL
      }

      iter_result$proposed_config <- proposed %||% config
    } else {
      iter_result$proposed_config <- config
    }

    history[[iter]] <- iter_result
  }

  base::structure(
    list(best = best_result, history = history,
         n_iterations = length(history), best_elpd = best_elpd),
    class = "optimization_result"
  )
}

#' @export
print.optimization_result <- function(x, ...) {
  cat(sprintf("<optimization_result>: %d iterations\n", x$n_iterations))
  cat(sprintf("  Best ELPD: %.1f\n", x$best_elpd))
  if (!is.null(x$best)) {
    cat(sprintf("  Best config: %s/%s/%s/%s\n",
        x$best$config$spec$measurement, x$best$config$spec$structural,
        x$best$config$spec$population, x$best$config$spec$item))
  }
  invisible(x)
}

#' Log one iteration to JSONL
#' @noRd
log_iteration <- function(iter, elpd, improved, diagnostics,
                          config_label, rationale, log_file) {
  entry <- list(iter = iter, elpd = elpd, improved = improved,
                config = config_label, rationale = rationale,
                timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"))
  if (!is.null(diagnostics)) {
    for (i in seq_len(nrow(diagnostics)))
      entry[[diagnostics$metric[i]]] <- diagnostics$value[i]
  }
  line <- jsonlite::toJSON(entry, auto_unbox = TRUE)
  cat(line, "\n", file = log_file, append = TRUE)
}
