#' Create a loading structure
#' @param taxonomy A tibble with columns `skill_id`, `name`, `description`.
#' @param assignments A tibble with columns `item_id`, `skill_id`, `skill_name`.
#' @param items A tibble with columns `item_id`, `text`.
#' @return An S3 object of class `loading_structure`.
#' @export
loading_structure <- function(taxonomy, assignments, items) {
  mask <- build_lambda_mask(assignments, items$item_id, taxonomy$skill_id)
  n_loadings <- sum(mask)
  x <- structure(
    list(taxonomy = taxonomy, assignments = assignments, items = items,
         lambda_mask = mask, n_loadings = as.integer(n_loadings)),
    class = "loading_structure"
  )
  validate_loading_structure(x)
  x
}

#' @noRd
build_lambda_mask <- function(assignments, item_ids, skill_ids) {
  mask <- matrix(FALSE, nrow = length(item_ids), ncol = length(skill_ids),
                 dimnames = list(item_ids, skill_ids))
  for (i in seq_len(nrow(assignments))) {
    mask[assignments$item_id[i], assignments$skill_id[i]] <- TRUE
  }
  mask
}

#' @noRd
validate_loading_structure <- function(x) {
  mask <- x$lambda_mask
  orphans <- rownames(mask)[rowSums(mask) == 0L]
  if (length(orphans) > 0)
    cli_abort("Items have no skill assignments: {.val {orphans}}.")
  thin_skills <- colnames(mask)[colSums(mask) < 2L]
  if (length(thin_skills) > 0) {
    names <- x$taxonomy$name[x$taxonomy$skill_id %in% thin_skills]
    cli_abort("Skills have fewer than 2 items (not identifiable): {.val {thin_skills}} ({names}).")
  }
  invisible(x)
}

#' @export
print.loading_structure <- function(x, ...) {
  n_items <- nrow(x$lambda_mask)
  n_skills <- ncol(x$lambda_mask)
  cat(sprintf("<loading_structure>: %d items, %d skills, %d loadings\n",
              n_items, n_skills, x$n_loadings))
  cat(sprintf("  Skills: %s\n", paste(x$taxonomy$name, collapse = ", ")))
  invisible(x)
}

#' Create an edge prior for SEM mode
#' @param from Character vector of parent skill IDs.
#' @param to Character vector of child skill IDs.
#' @param prob Numeric vector of prior probabilities in [0, 1].
#' @return An S3 object of class `edge_prior`.
#' @export
edge_prior <- function(from, to, prob) {
  if (length(from) != length(to) || length(from) != length(prob))
    cli_abort("{.arg from}, {.arg to}, and {.arg prob} must have the same length.")
  if (any(prob < 0 | prob > 1))
    cli_abort("All {.arg prob} values must be in [0, 1].")
  if (any(from == to))
    cli_abort("Edge prior contains a self-loop (from == to).")
  edges <- tibble::tibble(from = from, to = to, prob = prob)
  structure(list(edges = edges), class = "edge_prior")
}

#' @export
print.edge_prior <- function(x, ...) {
  cat(sprintf("<edge_prior>: %d edges\n", nrow(x$edges)))
  print(x$edges)
  invisible(x)
}
