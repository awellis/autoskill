#' Generate complete Stan code from a model configuration
#' @param config A `model_config` object.
#' @return A character string containing the complete Stan program.
#' @export
generate_stan_code <- function(config) {
  frag_structural <- stan_structural(config)
  frag_measurement <- stan_measurement(config)
  frag_population <- stan_population(config)
  frag_item <- stan_item(config)

  # For slip-guess: override compute_prob with the slip-guess version
  # and update the call sites to pass guess, slip
  if (config$spec$item == "slip_guess") {
    frag_measurement$functions <- ""
    frag_measurement$model <- gsub(
      "compute_prob\\(ii\\[n\\], theta\\[jj\\[n\\]\\], Lambda, alpha\\)",
      "compute_prob(ii[n], theta[jj[n]], Lambda, alpha, guess, slip)",
      frag_measurement$model
    )
    frag_measurement$generated_quantities <- gsub(
      "compute_prob\\(ii\\[n\\], theta\\[jj\\[n\\]\\], Lambda, alpha\\)",
      "compute_prob(ii[n], theta[jj[n]], Lambda, alpha, guess, slip)",
      frag_measurement$generated_quantities
    )
  }

  merged <- collapse_stan_lists(frag_structural, frag_measurement, frag_population, frag_item)
  assemble_stan_program(merged, config)
}

#' @noRd
assemble_stan_program <- function(fragments, config) {
  # Use paste0 instead of glue to avoid issues with Stan braces
  header <- paste0(
    "// autoskill ", utils::packageVersion("autoskill") %||% "0.1.0", "\n",
    "// ", config$spec$measurement, " / ", config$spec$structural, " / ",
    config$spec$population, " / ", config$spec$item, " / ", config$spec$link, "\n"
  )

  blocks <- c(
    paste0("functions {\n", fragments$functions, "\n}\n"),
    paste0("data {\n", fragments$data, "\n}\n"),
    paste0("transformed data {\n", fragments$transformed_data, "\n}\n"),
    paste0("parameters {\n", fragments$parameters, "\n}\n"),
    paste0("transformed parameters {\n", fragments$transformed_parameters, "\n}\n"),
    paste0("model {\n", fragments$model, "\n}\n"),
    paste0("generated quantities {\n", fragments$generated_quantities, "\n}\n")
  )

  paste0(header, "\n", paste(blocks, collapse = "\n"))
}
