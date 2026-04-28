#' Construct a structure_problem
#'
#' A `structure_problem` is the interface between a domain (skills,
#' causal discovery, mediation, etc.) and the generic optimisation
#' machinery (the optimization loop, future SMC). It bundles all the
#' domain-specific concerns into a single object the loop can drive
#' uniformly.
#'
#' Each slot is a closure typically built by a domain-specific
#' constructor such as [skill_problem()]. The optimisation loop only
#' interacts with these slots: it never imports domain-specific code
#' directly.
#'
#' @param data Anything. Typically a list of the data the problem
#'   operates over. Bound for reporting and for closures to reference.
#' @param propose_initial `function(...) -> structure`. Returns the
#'   starting structure for the search.
#' @param propose_refinement `function(current, fit_result, history) ->
#'   structure | NULL`. Given a fit and the run history, returns the
#'   next structure to evaluate. May return `NULL` to signal "no
#'   refinement, keep current".
#' @param score `function(structure, cache, ...) -> fit_result`. Fits
#'   the model implied by the structure and returns a `fit_result`-shaped
#'   object with at minimum a `$loo` field.
#' @param log_prior `function(structure) -> numeric`. Log probability of
#'   the structure under the problem's prior.
#' @param validate `function(structure) -> list(passed, problems)`.
#'   Cheap pre-fit check (identifiability, acyclicity, etc.).
#' @param cache_key `function(structure, fit_args) -> character`. Hash
#'   identifying the (structure, data, fit-settings) triple for caching.
#' @param summarize_structure `function(structure) -> character`. Short
#'   human-readable label, used in logs and progress reports.
#' @param summarize_fit `function(fit_result) -> list`. Domain-relevant
#'   summary of a fit, fed to the LLM for reflection.
#' @param propose_local_move Optional. `function(structure, move_type,
#'   ...) -> structure`. Required only by SMC; greedy search ignores it.
#' @param class Additional S3 classes to prepend to `"structure_problem"`.
#'   Domain constructors set this (e.g. `class = "skill_problem"`).
#' @return An S3 object inheriting from `"structure_problem"`.
#' @seealso [is_structure_problem()]
#' @export
structure_problem <- function(data,
                              propose_initial,
                              propose_refinement,
                              score,
                              log_prior,
                              validate,
                              cache_key,
                              summarize_structure,
                              summarize_fit,
                              propose_local_move = NULL,
                              class = character(0)) {
  obj <- list(
    data = data,
    propose_initial = propose_initial,
    propose_refinement = propose_refinement,
    score = score,
    log_prior = log_prior,
    validate = validate,
    cache_key = cache_key,
    summarize_structure = summarize_structure,
    summarize_fit = summarize_fit,
    propose_local_move = propose_local_move
  )
  class(obj) <- c(class, "structure_problem")
  validate_structure_problem(obj)
  obj
}

#' @noRd
required_problem_slots <- function() {
  c("propose_initial", "propose_refinement", "score", "log_prior",
    "validate", "cache_key", "summarize_structure", "summarize_fit")
}

#' @noRd
validate_structure_problem <- function(x) {
  for (slot in required_problem_slots()) {
    if (!is.function(x[[slot]])) {
      cli_abort("{.field {slot}} must be a function.")
    }
  }
  if (!is.null(x$propose_local_move) && !is.function(x$propose_local_move)) {
    cli_abort("{.field propose_local_move} must be NULL or a function.")
  }
  invisible(x)
}

#' Test whether an object is a `structure_problem`
#' @param x Any R object.
#' @return Logical scalar.
#' @export
is_structure_problem <- function(x) inherits(x, "structure_problem")

#' @export
print.structure_problem <- function(x, ...) {
  classes <- setdiff(class(x), "structure_problem")
  type_label <- if (length(classes) > 0L) classes[1] else "structure_problem"
  cat(sprintf("<%s>\n", type_label))
  cat("  Slots:\n")
  for (slot in required_problem_slots()) {
    cat(sprintf("    + %s\n", slot))
  }
  optional_marker <- if (is.function(x$propose_local_move)) "+" else "-"
  cat(sprintf("    %s propose_local_move (optional)\n", optional_marker))
  invisible(x)
}
