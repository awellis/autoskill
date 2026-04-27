#' Construct a disk-backed cache for fit results
#'
#' Wraps [cachem::cache_disk()] with defaults tuned for Stan fit_results.
#' The returned object is suitable for passing to [fit_cached()] or
#' [fit_many()].
#'
#' @param dir Directory for cache files. Use an absolute path when sharing
#'   the cache across parallel workers.
#' @param max_size Maximum cache size in bytes. Default 5 GB.
#' @param max_age Maximum entry age in seconds. Default `Inf`.
#' @return A `cachem` cache object.
#' @seealso [fit_cached()], [fit_many()]
#' @export
fit_cache <- function(dir = "fit_cache",
                      max_size = 5 * 1024^3,
                      max_age = Inf) {
  if (!requireNamespace("cachem", quietly = TRUE)) {
    cli_abort(c(
      "{.pkg cachem} is required for caching.",
      i = "Install via {.run install.packages(\"cachem\")}."
    ))
  }
  cachem::cache_disk(dir = dir, max_size = max_size, max_age = max_age)
}

#' Cached model fit
#'
#' Wraps [fit_model()] with optional disk-backed caching keyed on a hash of
#' the config, the response data, and the fit settings. With `cache = NULL`
#' (the default) behaviour is identical to [fit_model()].
#'
#' Use cases: resumability across sessions, replay during development, and
#' duplicate-config deduplication when batching multiple proposals or
#' running structure-search algorithms over a population of particles.
#'
#' @param responses A `response_data` object.
#' @param config A `model_config` object.
#' @param cache A `cachem` cache (e.g. from [fit_cache()]) or `NULL`.
#' @param ... Arguments passed to [fit_model()] (e.g. `chains`,
#'   `iter_warmup`, `iter_sampling`, `adapt_delta`, `max_treedepth`).
#' @return A `fit_result` object.
#' @seealso [fit_cache()], [fit_many()]
#' @export
fit_cached <- function(responses, config, cache = NULL, ...) {
  if (is.null(cache)) return(fit_model(responses, config, ...))

  key <- fit_cache_key(responses, config, list(...))
  hit <- cache$get(key)
  if (!cachem::is.key_missing(hit)) return(hit)

  result <- fit_model(responses, config, ...)
  cache$set(key, result)
  result
}

#' Compute a cache key for a fit
#'
#' Hashes the config, response matrix, and the subset of fit arguments that
#' affect inference. Exposed for testing and for callers that want to look
#' up cached fits manually.
#'
#' @param responses A `response_data` object.
#' @param config A `model_config` object.
#' @param fit_args A named list of arguments passed to [fit_model()].
#' @return A character string (hex digest).
#' @export
fit_cache_key <- function(responses, config, fit_args = list()) {
  keyed_names <- c("chains", "iter_warmup", "iter_sampling",
                   "adapt_delta", "max_treedepth", "parallel_chains",
                   "seed", "init", "thin")
  keyed <- fit_args[intersect(names(fit_args), keyed_names)]
  if (length(keyed) > 1) keyed <- keyed[order(names(keyed))]

  rlang::hash(list(
    config = config_hash(config),
    responses = rlang::hash(list(
      Y = responses$Y,
      item_ids = responses$item_ids,
      student_ids = responses$student_ids
    )),
    fit_args = keyed
  ))
}
