#' @keywords internal
"_PACKAGE"

#' @importFrom rlang %||% .data := abort warn inform
#' @importFrom cli cli_abort cli_warn cli_inform
#' @importFrom glue glue
#' @importFrom stats setNames
NULL

# Global variable declarations for NSE (tidyverse column references)
utils::globalVariables(c(
  "skills", "skill_id", "name", "item_id", "skill_name", "pareto_k"
))
