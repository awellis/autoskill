#' Compare multiple fitted models by LOO-CV
#' @param ... Named `fit_result` objects.
#' @return A tibble with LOO comparison statistics.
#' @export
compare_models <- function(...) {
  results <- list(...)
  loo_list <- lapply(results, function(r) r$loo)
  comp <- loo::loo_compare(loo_list)
  tibble::tibble(
    model = rownames(comp),
    elpd_diff = comp[, "elpd_diff"],
    se_diff = comp[, "se_diff"],
    elpd_loo = comp[, "elpd_loo"],
    se_elpd_loo = comp[, "se_elpd_loo"],
    p_loo = comp[, "p_loo"]
  )
}

#' Flag observations with high Pareto k
#' @param fit_result A `fit_result` object.
#' @param k_threshold Pareto k threshold (default 0.7).
#' @return A tibble with flagged observations.
#' @export
flag_problem_items <- function(fit_result, k_threshold = 0.7) {
  k_values <- loo::pareto_k_values(fit_result$loo)
  tibble::tibble(
    obs_id = seq_along(k_values),
    pareto_k = k_values,
    status = dplyr::case_when(
      k_values > 1.0 ~ "very_bad",
      k_values > k_threshold ~ "bad",
      k_values > 0.5 ~ "ok",
      TRUE ~ "good"
    )
  ) |>
    dplyr::filter(pareto_k > k_threshold)
}

#' Format diagnostics as structured text for LLM reflection
#' @param fit_result A `fit_result` object.
#' @param comparison Optional tibble from `compare_models()`.
#' @param previous_configs Optional list of previously tried configs.
#' @param items Tibble with `item_id` and `text`.
#' @return A character string.
#' @export
format_reflection_prompt <- function(fit_result, comparison = NULL,
                                     previous_configs = NULL, items = NULL) {
  config <- fit_result$config
  diag <- fit_result$diagnostics
  elpd <- fit_result$loo$estimates["elpd_loo", "Estimate"]

  parts <- c(
    "## Current Model",
    paste0("- Measurement: ", config$spec$measurement),
    paste0("- Structural: ", config$spec$structural),
    paste0("- Population: ", config$spec$population),
    paste0("- Item: ", config$spec$item),
    paste0("- Link: ", config$spec$link),
    paste0("- Skills: ", paste(config$structure$taxonomy$name, collapse = ", ")),
    "",
    "## Loading Matrix",
    format_mask_for_prompt(config$structure, items),
    "",
    "## Diagnostics",
    paste0("- ELPD (LOO): ", round(elpd, 1)),
    paste(paste0("- ", diag$metric, ": ", round(diag$value, 3), " (", diag$status, ")"), collapse = "\n")
  )

  problems <- flag_problem_items(fit_result)
  if (nrow(problems) > 0) {
    parts <- c(parts, "", paste0("## Problem Observations (", nrow(problems), " with Pareto k > 0.7)"))
  }

  if (!is.null(comparison)) {
    parts <- c(parts, "", "## Model Comparison (LOO)",
               paste(utils::capture.output(print(comparison)), collapse = "\n"))
  }

  if (!is.null(previous_configs) && length(previous_configs) > 0) {
    labels <- vapply(previous_configs, function(pc) {
      paste0("- ", pc$spec$measurement, "/", pc$spec$structural, "/", pc$spec$population, "/", pc$spec$item)
    }, character(1))
    parts <- c(parts, "", "## Previously Tried Configurations", paste(labels, collapse = "\n"))
  }

  paste(parts, collapse = "\n")
}

#' @noRd
format_mask_for_prompt <- function(struc, items = NULL) {
  mask <- struc$lambda_mask
  skill_names <- struc$taxonomy$name

  header <- paste(c("Item", skill_names), collapse = " | ")
  separator <- paste(rep("---", length(skill_names) + 1), collapse = " | ")

  rows <- vapply(seq_len(nrow(mask)), function(i) {
    item_label <- if (!is.null(items)) {
      items$text[items$item_id == rownames(mask)[i]]
    } else {
      rownames(mask)[i]
    }
    marks <- ifelse(mask[i, ], "X", ".")
    paste(c(item_label, marks), collapse = " | ")
  }, character(1))

  paste(c(header, separator, rows), collapse = "\n")
}
