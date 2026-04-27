#' Fit multiple configs in parallel
#'
#' Runs [fit_cached()] over a list of configs using `furrr::future_map()`.
#' Parallelism is controlled by the user via `future::plan()`; with the
#' default sequential plan, fits run serially. Cache hits are served
#' without invoking Stan, so duplicate configs in the batch are deduped.
#'
#' Errors in a single fit do not abort the batch: failed configs return a
#' `fit_error` object with the underlying error message. Filter the result
#' with `purrr::keep(results, inherits, "fit_result")`.
#'
#' To enable parallel execution:
#' ```r
#' future::plan(future::multisession, workers = 4)
#' results <- fit_many(responses, configs, cache = fit_cache())
#' ```
#'
#' @param responses A `response_data` object.
#' @param configs A list of `model_config` objects.
#' @param cache A `cachem` cache (optional). When sharing across workers,
#'   construct with an absolute path.
#' @param ... Arguments passed to [fit_cached()].
#' @return A list the same length as `configs`. Each element is either a
#'   `fit_result` (success) or a `fit_error` (failure).
#' @seealso [fit_cached()], [fit_cache()]
#' @export
fit_many <- function(responses, configs, cache = NULL, ...) {
  if (length(configs) == 0L) return(list())

  if (!requireNamespace("furrr", quietly = TRUE)) {
    cli_abort(c(
      "{.pkg furrr} is required for parallel fitting.",
      i = "Install via {.run install.packages(c(\"future\", \"furrr\"))}."
    ))
  }

  furrr::future_map(
    configs,
    function(cfg) {
      tryCatch(
        fit_cached(responses, cfg, cache = cache, ...),
        error = function(e) {
          base::structure(
            list(config = cfg, error = conditionMessage(e)),
            class = "fit_error"
          )
        }
      )
    },
    .options = furrr::furrr_options(seed = TRUE)
  )
}

#' @export
print.fit_error <- function(x, ...) {
  cat(sprintf("<fit_error>: %s\n", x$error))
  invisible(x)
}
