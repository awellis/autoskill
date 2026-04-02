#' Merge named lists of Stan code fragments
#'
#' Concatenates same-named elements across multiple fragment lists.
#' This is the core composition mechanism for the block-based Stan
#' code generator (analogous to brms's internal collapse approach).
#'
#' @param ... Named lists of Stan code fragments.
#' @return A named list with all unique keys, values concatenated.
#' @noRd
collapse_stan_lists <- function(...) {
  ls <- list(...)
  all_keys <- unique(unlist(lapply(ls, names)))
  out <- lapply(all_keys, function(key) {
    parts <- vapply(ls, function(x) x[[key]] %||% "", character(1))
    paste0(parts, collapse = "")
  })
  setNames(out, all_keys)
}


#' Topological sort of a DAG
#'
#' Returns nodes in topological order (parents before children).
#' Uses Kahn's algorithm.
#'
#' @param from Character vector of parent node IDs.
#' @param to Character vector of child node IDs (same length as `from`).
#' @param nodes Character vector of all node IDs.
#' @return Character vector of nodes in topological order.
#' @noRd
topological_sort <- function(from, to, nodes) {
  n <- length(nodes)
  if (length(from) == 0L) return(nodes)

  # Build adjacency list and in-degree count
  children <- setNames(vector("list", n), nodes)
  in_degree <- setNames(integer(n), nodes)

  for (i in seq_along(from)) {
    children[[from[i]]] <- c(children[[from[i]]], to[i])
    in_degree[[to[i]]] <- in_degree[[to[i]]] + 1L
  }

  # Kahn's algorithm
  queue <- nodes[in_degree == 0L]
  result <- character(0)

  while (length(queue) > 0L) {
    node <- queue[1]
    queue <- queue[-1]
    result <- c(result, node)

    for (child in children[[node]]) {
      in_degree[[child]] <- in_degree[[child]] - 1L
      if (in_degree[[child]] == 0L) {
        queue <- c(queue, child)
      }
    }
  }

  if (length(result) != n) {
    cli_abort("DAG contains a cycle: topological sort failed.")
  }

  result
}


#' Validate that edges form a DAG (no cycles)
#'
#' @param from Character vector of parent node IDs.
#' @param to Character vector of child node IDs.
#' @param nodes Character vector of all node IDs.
#' @return `TRUE` if acyclic, `FALSE` if cyclic.
#' @noRd
validate_dag <- function(from, to, nodes) {
  if (length(from) == 0L) return(TRUE)
  tryCatch(
    {
      topological_sort(from, to, nodes)
      TRUE
    },
    error = function(e) FALSE
  )
}


#' Create an empty Stan fragment list
#'
#' Returns a named list with all 7 Stan block keys set to empty strings.
#' Block generators should start from this and fill in their pieces.
#'
#' @return Named list with keys: functions, data, transformed_data,
#'   parameters, transformed_parameters, model, generated_quantities.
#' @noRd
empty_stan_fragment <- function() {
  list(
    functions = "",
    data = "",
    transformed_data = "",
    parameters = "",
    transformed_parameters = "",
    model = "",
    generated_quantities = ""
  )
}
