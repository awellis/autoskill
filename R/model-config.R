#' Create a model configuration
#' @param spec A `model_spec` object.
#' @param struc A `loading_structure` object.
#' @param edge_prior An `edge_prior` object (required for `structural = "dag"`).
#' @return An S3 object of class `model_config`.
#' @export
model_config <- function(spec, struc, edge_prior = NULL) {
  config <- base::structure(
    list(spec = spec, structure = struc, edge_prior = edge_prior),
    class = "model_config"
  )
  validate_model_config(config)
  config
}

#' @noRd
validate_model_config <- function(x) {
  if (!inherits(x$spec, "model_spec"))
    cli_abort("{.arg spec} must be a {.cls model_spec} object.")
  if (!inherits(x$structure, "loading_structure"))
    cli_abort("{.arg structure} must be a {.cls loading_structure} object.")
  if (x$spec$structural == "dag" && is.null(x$edge_prior))
    cli_abort("Structural model {.val dag} requires an {.arg edge_prior}.")
  if (!is.null(x$edge_prior) && !inherits(x$edge_prior, "edge_prior"))
    cli_abort("{.arg edge_prior} must be an {.cls edge_prior} object.")
  invisible(x)
}

#' Compute a content hash for a model configuration
#' @param config A `model_config` object.
#' @return A character string (hex digest).
#' @export
config_hash <- function(config) {
  content <- list(
    spec = unclass(config$spec),
    lambda_mask = config$structure$lambda_mask,
    edge_prior = if (!is.null(config$edge_prior)) config$edge_prior$edges else NULL
  )
  rlang::hash(content)
}

#' @export
print.model_config <- function(x, ...) {
  n_items <- nrow(x$structure$lambda_mask)
  n_skills <- ncol(x$structure$lambda_mask)
  mode <- if (is_sem_mode(x$spec)) "SEM" else "Factor analysis"
  cat(sprintf("<model_config> (%s): %d items, %d skills\n", mode, n_items, n_skills))
  cat(sprintf("  measurement: %s\n", x$spec$measurement))
  cat(sprintf("  structural:  %s\n", x$spec$structural))
  cat(sprintf("  population:  %s\n", x$spec$population))
  cat(sprintf("  item:        %s\n", x$spec$item))
  cat(sprintf("  link:        %s\n", x$spec$link))
  invisible(x)
}
