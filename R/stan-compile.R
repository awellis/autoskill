#' Compile a Stan model from a model configuration
#' @param config A `model_config` object.
#' @param dir Cache directory.
#' @param force If TRUE, recompile even if cached.
#' @return A CmdStanModel object.
#' @export
compile_model <- function(config, dir = NULL, force = FALSE) {
  dir <- dir %||% file.path(tools::R_user_dir("autoskill", "cache"), "stan")
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
  code <- generate_stan_code(config)
  hash <- rlang::hash(code)
  stan_file <- file.path(dir, paste0("autoskill_", hash, ".stan"))
  if (!file.exists(stan_file) || force) {
    writeLines(code, stan_file)
  }
  cmdstanr::cmdstan_model(stan_file, compile = TRUE)
}

#' Clear the autoskill Stan model cache
#' @param dir Cache directory.
#' @export
clear_stan_cache <- function(dir = NULL) {
  dir <- dir %||% file.path(tools::R_user_dir("autoskill", "cache"), "stan")
  if (dir.exists(dir)) {
    files <- list.files(dir, full.names = TRUE)
    file.remove(files)
    cli_inform("Removed {length(files)} cached Stan files.")
  }
  invisible()
}
