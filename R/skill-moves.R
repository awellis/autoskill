#' Propose a single local move on a skill model_config
#'
#' Stochastic mutation kernel for SMC over skill structures. Picks one of
#' two move types and applies it: with probability 0.7, flips a single
#' entry of the loading mask; with probability 0.3, swaps one block
#' option (measurement, item, or link). Moves that would create
#' non-identifiable structures (orphan items, skills with fewer than two
#' items) are rejected and the input is returned unchanged.
#'
#' Wired into [skill_problem()] as the `propose_local_move` slot, so
#' [optimize_structure_smc()] can drive skill_problems just as it drives
#' causal_problems.
#'
#' @param config A `model_config`.
#' @param ... Unused; present for compatibility with the structure_problem
#'   slot signature.
#' @return A new `model_config`, or `config` unchanged if the move would
#'   produce an invalid structure.
#' @export
propose_skill_local_move <- function(config, ...) {
  move_type <- sample(c("flip_lambda", "swap_block"), 1L,
                      prob = c(0.7, 0.3))

  if (move_type == "flip_lambda") {
    flip_random_lambda(config)
  } else {
    swap_random_block(config)
  }
}

#' @noRd
flip_random_lambda <- function(config) {
  mask <- config$structure$lambda_mask
  i <- sample.int(nrow(mask), 1L)
  k <- sample.int(ncol(mask), 1L)

  new_mask <- mask
  new_mask[i, k] <- !new_mask[i, k]

  new_assignments <- mask_to_assignments(new_mask, config$structure$taxonomy)

  new_struc <- tryCatch(
    loading_structure(
      taxonomy = config$structure$taxonomy,
      assignments = new_assignments,
      items = config$structure$items
    ),
    error = function(e) NULL
  )

  if (is.null(new_struc)) return(config)

  model_config(config$spec, new_struc, edge_prior = config$edge_prior)
}

#' @noRd
swap_random_block <- function(config) {
  spec <- config$spec

  # Restricted to blocks that are always valid without extra data:
  # - structural=dag/hierarchical needs an edge_prior
  # - population=grouped needs group structure in responses
  swappable <- c("measurement", "item", "link")
  block <- sample(swappable, 1L)

  options_for_block <- switch(block,
    measurement = MEASUREMENT_OPTIONS,
    item        = ITEM_OPTIONS,
    link        = LINK_OPTIONS
  )

  alternatives <- setdiff(options_for_block, spec[[block]])
  if (length(alternatives) == 0L) return(config)

  new_value <- sample(alternatives, 1L)

  new_spec_args <- list(
    measurement = spec$measurement,
    structural  = spec$structural,
    population  = spec$population,
    item        = spec$item,
    link        = spec$link
  )
  new_spec_args[[block]] <- new_value

  new_spec <- do.call(model_spec, new_spec_args)

  model_config(new_spec, config$structure, edge_prior = config$edge_prior)
}

#' Convert a Lambda mask back to an assignments tibble
#' @noRd
mask_to_assignments <- function(mask, taxonomy) {
  idx <- which(mask, arr.ind = TRUE)
  if (nrow(idx) == 0L) {
    return(tibble::tibble(
      item_id = character(0),
      skill_id = character(0),
      skill_name = character(0)
    ))
  }
  item_ids <- rownames(mask)[idx[, 1L]]
  skill_ids <- colnames(mask)[idx[, 2L]]
  skill_names <- taxonomy$name[match(skill_ids, taxonomy$skill_id)]
  tibble::tibble(
    item_id = item_ids,
    skill_id = skill_ids,
    skill_name = skill_names
  )
}
