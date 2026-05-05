#' Propose a skill local move via LLM
#'
#' Asks the LLM to suggest a single small change to the current model
#' configuration: toggle one loading, merge two conceptually overlapping
#' skills, or decline (no-op). The LLM sees the item text, the current
#' skill taxonomy, and which skills each item currently loads on.
#'
#' Move types:
#'
#' - `flip_loading`: Set `Lambda[item, skill]` from 0 to 1 or 1 to 0.
#'   Useful for adding a missing semantic dependency or removing a
#'   spurious one.
#' - `merge_skills`: Combine two skills that conceptually overlap. The
#'   merged skill takes the union of items from both. Drops one skill ID
#'   and any edges in the prior involving it.
#' - `no_op`: Return the structure unchanged (LLM declines).
#'
#' Failed or invalid moves (LLM error, unknown item or skill ID, move
#' that breaks identifiability) return the input config unchanged. Valid
#' to use as a `propose_local_move` slot on a [skill_problem()].
#'
#' @param config A `model_config`.
#' @param chat An `ellmer` chat object.
#' @param items Tibble with `item_id` and `text` columns. Used to give
#'   the LLM the item text for context.
#' @return A new `model_config`, or `config` unchanged on any failure.
#' @seealso [propose_skill_local_move()], [skill_problem()]
#' @export
propose_skill_local_move_llm <- function(config, chat, items) {
  prompt <- build_local_move_prompt(config, items)
  schema <- local_move_schema()

  result <- tryCatch(
    chat$chat_structured(prompt, type = schema),
    error = function(e) {
      cli_warn("LLM local move failed: {conditionMessage(e)}")
      NULL
    }
  )

  if (is.null(result)) return(config)
  apply_local_move(config, result)
}

#' @noRd
build_local_move_prompt <- function(config, items) {
  taxonomy <- config$structure$taxonomy
  mask <- config$structure$lambda_mask

  tax_md <- paste0(
    "### Current skills\n",
    paste(sprintf("- `%s` (%s): %s",
                  taxonomy$skill_id, taxonomy$name, taxonomy$description),
          collapse = "\n")
  )

  loading_lines <- character(nrow(mask))
  for (i in seq_len(nrow(mask))) {
    item_id_i <- rownames(mask)[i]
    item_text <- items$text[items$item_id == item_id_i]
    if (length(item_text) == 0L) item_text <- "(text not provided)"
    loaded_skill_ids <- colnames(mask)[mask[i, ] == TRUE]
    loaded_names <- taxonomy$name[match(loaded_skill_ids, taxonomy$skill_id)]
    loading_lines[i] <- sprintf(
      "- `%s` \"%s\" -> loads on: %s",
      item_id_i, item_text,
      if (length(loaded_names) == 0L) "(none)"
      else paste(loaded_names, collapse = ", ")
    )
  }
  loadings_md <- paste0(
    "### Current loadings\n", paste(loading_lines, collapse = "\n")
  )

  paste0(
    tax_md, "\n\n", loadings_md, "\n\n",
    "## Your task\n",
    "Propose ONE small local change to the model. Choose:\n\n",
    "1. `flip_loading`: Toggle a single `Lambda[item, skill]` entry. ",
    "Set it from 0 to 1 if the item should require that skill (a ",
    "missed dependency); from 1 to 0 if the loading is spurious.\n",
    "2. `merge_skills`: Combine two skills that are conceptually the ",
    "same thing. Use this when two skills cover overlapping content.\n",
    "3. `no_op`: Decline if no useful local change is apparent.\n\n",
    "Return the relevant IDs (use exact `skill_id` and `item_id` ",
    "values from above) plus a brief rationale."
  )
}

#' @noRd
local_move_schema <- function() {
  ellmer::type_object(
    move_type = ellmer::type_enum(
      values = c("flip_loading", "merge_skills", "no_op"),
      description = "Which local move to apply"
    ),
    item_id = ellmer::type_string(
      description = "For flip_loading: the item_id whose loading to toggle. Empty otherwise.",
      required = FALSE
    ),
    skill_id_1 = ellmer::type_string(
      description = "For flip_loading: the skill_id to toggle. For merge_skills: the first skill_id (kept).",
      required = FALSE
    ),
    skill_id_2 = ellmer::type_string(
      description = "For merge_skills: the second skill_id (will be dropped).",
      required = FALSE
    ),
    rationale = ellmer::type_string(
      description = "Brief explanation of why this move improves the model"
    )
  )
}

#' @noRd
apply_local_move <- function(config, move) {
  result <- switch(move$move_type,
    flip_loading = tryCatch(
      apply_flip_loading(config, move$item_id, move$skill_id_1),
      error = function(e) NULL
    ),
    merge_skills = tryCatch(
      apply_merge_skills(config, move$skill_id_1, move$skill_id_2),
      error = function(e) NULL
    ),
    no_op = config,
    NULL
  )

  if (is.null(result)) config else result
}

#' @noRd
apply_flip_loading <- function(config, item_id, skill_id) {
  mask <- config$structure$lambda_mask
  if (!(item_id %in% rownames(mask)) ||
      !(skill_id %in% colnames(mask))) {
    return(config)
  }

  new_mask <- mask
  new_mask[item_id, skill_id] <- !new_mask[item_id, skill_id]
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
apply_merge_skills <- function(config, skill_keep, skill_drop) {
  taxonomy <- config$structure$taxonomy
  mask <- config$structure$lambda_mask

  if (!(skill_keep %in% taxonomy$skill_id) ||
      !(skill_drop %in% taxonomy$skill_id) ||
      skill_keep == skill_drop) {
    return(config)
  }
  if (ncol(mask) <= 2L) {
    # Refuse to drop below two skills; the IRT model needs >= 1.
    return(config)
  }

  new_mask <- mask
  new_mask[, skill_keep] <- mask[, skill_keep] | mask[, skill_drop]
  new_mask <- new_mask[, colnames(new_mask) != skill_drop, drop = FALSE]
  new_taxonomy <- taxonomy[taxonomy$skill_id != skill_drop, ]

  new_edge_prior <- config$edge_prior
  if (!is.null(new_edge_prior)) {
    keep <- !(new_edge_prior$edges$from == skill_drop |
              new_edge_prior$edges$to == skill_drop)
    if (any(keep)) {
      new_edge_prior$edges <- new_edge_prior$edges[keep, , drop = FALSE]
    } else {
      new_edge_prior <- NULL
    }
  }

  new_assignments <- mask_to_assignments(new_mask, new_taxonomy)

  new_struc <- tryCatch(
    loading_structure(
      taxonomy = new_taxonomy,
      assignments = new_assignments,
      items = config$structure$items
    ),
    error = function(e) NULL
  )
  if (is.null(new_struc)) return(config)

  tryCatch(
    model_config(config$spec, new_struc, edge_prior = new_edge_prior),
    error = function(e) config
  )
}
