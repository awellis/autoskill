# Valid options for each block
MEASUREMENT_OPTIONS <- c("linear", "interaction")
STRUCTURAL_OPTIONS <- c("independent", "correlated", "dag", "hierarchical")
POPULATION_OPTIONS <- c("single", "grouped")
ITEM_OPTIONS <- c("basic", "slip_guess")
LINK_OPTIONS <- c("logit", "probit")

#' Create a model specification
#' @param measurement One of "linear" or "interaction".
#' @param structural One of "independent", "correlated", "dag", "hierarchical".
#' @param population One of "single" or "grouped".
#' @param item One of "basic" or "slip_guess".
#' @param link One of "logit" or "probit".
#' @return An S3 object of class `model_spec`.
#' @export
model_spec <- function(measurement = "linear", structural = "independent",
                       population = "single", item = "basic", link = "logit") {
  spec <- structure(
    list(measurement = measurement, structural = structural,
         population = population, item = item, link = link),
    class = "model_spec"
  )
  validate_model_spec(spec)
  spec
}

#' @noRd
validate_model_spec <- function(x) {
  if (!x$measurement %in% MEASUREMENT_OPTIONS)
    cli_abort("{.arg measurement} must be one of {.or {.val {MEASUREMENT_OPTIONS}}}, not {.val {x$measurement}}.")
  if (!x$structural %in% STRUCTURAL_OPTIONS)
    cli_abort("{.arg structural} must be one of {.or {.val {STRUCTURAL_OPTIONS}}}, not {.val {x$structural}}.")
  if (!x$population %in% POPULATION_OPTIONS)
    cli_abort("{.arg population} must be one of {.or {.val {POPULATION_OPTIONS}}}, not {.val {x$population}}.")
  if (!x$item %in% ITEM_OPTIONS)
    cli_abort("{.arg item} must be one of {.or {.val {ITEM_OPTIONS}}}, not {.val {x$item}}.")
  if (!x$link %in% LINK_OPTIONS)
    cli_abort("{.arg link} must be one of {.or {.val {LINK_OPTIONS}}}, not {.val {x$link}}.")
  invisible(x)
}

#' @export
print.model_spec <- function(x, ...) {
  cat("<model_spec>\n")
  cat("* measurement:", x$measurement, "\n")
  cat("* structural: ", x$structural, "\n")
  cat("* population: ", x$population, "\n")
  cat("* item:       ", x$item, "\n")
  cat("* link:       ", x$link, "\n")
  invisible(x)
}

#' Test whether a model spec uses SEM structural mode
#' @param spec A `model_spec` object.
#' @return Logical.
#' @export
is_sem_mode <- function(spec) {
  spec$structural %in% c("dag", "hierarchical")
}

#' Test whether a model spec uses factor-analytic structural mode
#' @param spec A `model_spec` object.
#' @return Logical.
#' @export
is_fa_mode <- function(spec) {
  !is_sem_mode(spec)
}

#' Enumerate all valid model specifications
#' @return A tibble with columns: measurement, structural, population, item, link.
#' @export
all_valid_specs <- function() {
  tidyr::crossing(
    measurement = MEASUREMENT_OPTIONS, structural = STRUCTURAL_OPTIONS,
    population = POPULATION_OPTIONS, item = ITEM_OPTIONS, link = LINK_OPTIONS
  )
}
