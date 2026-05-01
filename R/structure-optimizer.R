#' Run one iteration of the optimization loop
#'
#' Generic over the structure type. Validates via `problem$validate()`,
#' fits via `problem$score()`, and reports back diagnostics via
#' `problem$summarize_fit()` if the loop needs them.
#'
#' @param problem A `structure_problem` object.
#' @param structure The current candidate structure (e.g. a `model_config`
#'   for skill problems, an `igraph` DAG for causal problems).
#' @param cache Optional `cachem` cache (e.g. from [fit_cache()]).
#' @param ... Additional arguments passed to `problem$score()`.
#' @return An S3 object of class `iteration_result`.
#' @export
run_iteration <- function(problem, structure, cache = NULL, ...) {
  if (!is_structure_problem(problem)) {
    cli_abort("{.arg problem} must be a {.cls structure_problem} object.")
  }

  v <- problem$validate(structure)
  if (!isTRUE(v$passed)) {
    return(base::structure(
      list(structure = structure, fit_result = NULL,
           identifiable = FALSE, sbc_passed = NA,
           elpd = NA_real_, problems = v$problems %||% character(0)),
      class = "iteration_result"
    ))
  }

  fit_result <- problem$score(structure, cache = cache, ...)
  elpd <- fit_result$loo$estimates["elpd_loo", "Estimate"]

  base::structure(
    list(structure = structure, fit_result = fit_result,
         identifiable = TRUE, sbc_passed = NA,
         elpd = elpd, problems = character(0)),
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

#' Run the optimization loop
#'
#' Greedy ELPD-argmax with LLM-driven refinement and stacking over the
#' visited structures. Drives a [structure_problem()]: propose initial,
#' score, reflect, propose refinement, repeat.
#'
#' @param problem A `structure_problem` object (e.g. from
#'   [skill_problem()] or any other domain constructor).
#' @param max_iter Maximum iterations.
#' @param patience Stop after `N` consecutive non-improving iterations.
#' @param cache Optional `cachem` cache for memoising fits across
#'   iterations and across runs.
#' @param log_file Optional JSONL log file path.
#' @param ... Additional arguments threaded to `problem$score()` (e.g.
#'   `chains`, `iter_warmup`).
#' @return An S3 object of class `optimization_result`.
#' @seealso [skill_problem()], [optimize_structure_skill()],
#'   [compute_stacked_weights()]
#' @export
optimize_structure <- function(problem,
                               max_iter = 10, patience = 3,
                               cache = NULL, log_file = NULL, ...) {
  if (!is_structure_problem(problem)) {
    cli_abort("{.arg problem} must be a {.cls structure_problem} object.")
  }

  history <- list()
  best_elpd <- -Inf
  best_result <- NULL
  non_improving <- 0
  current <- problem$propose_initial(...)

  for (iter in seq_len(max_iter)) {
    label <- problem$summarize_structure(current)
    cli_inform("Iteration {iter}/{max_iter}: {label}")

    iter_result <- run_iteration(problem, current, cache = cache, ...)

    improved <- !is.na(iter_result$elpd) && iter_result$elpd > best_elpd
    if (improved) {
      best_elpd <- iter_result$elpd
      best_result <- iter_result
      non_improving <- 0
    } else {
      non_improving <- non_improving + 1
    }

    if (!is.null(log_file)) {
      log_iteration(
        iter = iter, elpd = iter_result$elpd, improved = improved,
        diagnostics = iter_result$fit_result$diagnostics,
        label = label,
        rationale = if (iter == 1) "initial" else "LLM refinement",
        log_file = log_file
      )
    }

    cli_inform("  ELPD = {round(iter_result$elpd, 1)}, improved = {improved}, patience = {patience - non_improving}")

    history[[iter]] <- iter_result

    if (non_improving >= patience) {
      cli_inform("Stopping: {patience} consecutive non-improving iterations.")
      break
    }

    if (iter < max_iter && !is.null(iter_result$fit_result)) {
      proposed <- problem$propose_refinement(current,
                                             iter_result$fit_result,
                                             history)
      current <- proposed %||% current
    }
  }

  base::structure(
    list(best = best_result, history = history,
         n_iterations = length(history), best_elpd = best_elpd,
         stacked_weights = collect_stacked_weights(history),
         problem = problem),
    class = "optimization_result"
  )
}

#' Skill-flavoured legacy wrapper around [optimize_structure()]
#'
#' Preserves the pre-refactor calling convention: takes responses, items,
#' optional initial config and chat. Internally constructs a
#' [skill_problem()] and forwards to the generic [optimize_structure()].
#'
#' New code should prefer the generic form
#' (`optimize_structure(skill_problem(...))`); this wrapper exists so
#' existing examples and downstream callers keep working unchanged.
#'
#' @param responses A `response_data` object.
#' @param items Tibble with `item_id` and `text`.
#' @param initial_config Optional `model_config` to start from. If `NULL`,
#'   the LLM proposer runs.
#' @param edge_prior Optional `edge_prior` object.
#' @param chat An `ellmer` chat object.
#' @param model LLM model identifier (used when `chat` is `NULL`).
#' @param interactive If `TRUE`, prompts to accept/reject each refinement.
#' @param max_iter,patience,cache,log_file See [optimize_structure()].
#' @param ... Passed to `problem$score()` (i.e. fit_cached / fit_model).
#' @return An S3 object of class `optimization_result`.
#' @export
optimize_structure_skill <- function(responses, items,
                                     initial_config = NULL,
                                     edge_prior = NULL,
                                     chat = NULL,
                                     model = "claude-sonnet-4-20250514",
                                     interactive = FALSE,
                                     max_iter = 10, patience = 3,
                                     cache = NULL, log_file = NULL, ...) {
  problem <- skill_problem(
    items = items, responses = responses,
    edge_prior = edge_prior, chat = chat, model = model,
    interactive = interactive
  )

  if (!is.null(initial_config)) {
    # Inject the user-supplied starting structure as the initial proposal,
    # bypassing the LLM proposer.
    problem$propose_initial <- function(...) initial_config
  }

  optimize_structure(
    problem,
    max_iter = max_iter, patience = patience,
    cache = cache, log_file = log_file, ...
  )
}

#' @noRd
collect_stacked_weights <- function(history) {
  fit_results <- purrr::map(history, "fit_result")
  has_fit <- !purrr::map_lgl(fit_results, is.null)
  if (!any(has_fit)) return(NULL)

  fits <- fit_results[has_fit]
  names(fits) <- paste0("iter_", which(has_fit))

  tryCatch(
    compute_stacked_weights(fits),
    error = function(e) {
      cli_warn("Stacking failed: {conditionMessage(e)}")
      NULL
    }
  )
}

#' @export
print.optimization_result <- function(x, ..., n_top = 5) {
  cat(sprintf("<optimization_result>: %d iterations\n", x$n_iterations))
  cat(sprintf("  Best ELPD: %.1f\n", x$best_elpd))
  if (!is.null(x$best) && !is.null(x$problem)) {
    label <- x$problem$summarize_structure(x$best$structure)
    cat(sprintf("  Best: %s\n", label))
  }
  if (!is.null(x$stacked_weights) && length(x$stacked_weights) > 0L) {
    sw <- sort(x$stacked_weights, decreasing = TRUE)
    top <- utils::head(sw, n_top)
    cat("  Stacked weights:\n")
    for (i in seq_along(top)) {
      cat(sprintf("    %s: %.3f\n", names(top)[i], top[i]))
    }
    if (length(sw) > n_top) {
      cat(sprintf("    (... %d more)\n", length(sw) - n_top))
    }
  }
  invisible(x)
}

#' Log one iteration to JSONL
#' @noRd
log_iteration <- function(iter, elpd, improved, diagnostics,
                          label, rationale, log_file) {
  entry <- list(iter = iter, elpd = elpd, improved = improved,
                structure = label, rationale = rationale,
                timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"))
  if (!is.null(diagnostics)) {
    for (i in seq_len(nrow(diagnostics))) {
      entry[[diagnostics$metric[i]]] <- diagnostics$value[i]
    }
  }
  line <- jsonlite::toJSON(entry, auto_unbox = TRUE)
  cat(line, "\n", file = log_file, append = TRUE)
}
