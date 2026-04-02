#' Propose a refined model configuration via LLM
#' @param chat An ellmer chat object.
#' @param fit_result A `fit_result` object.
#' @param items Tibble with `item_id` and `text`.
#' @param comparison Optional tibble from `compare_models()`.
#' @param previous_configs Optional list of previously tried configs.
#' @return A `model_config` object.
#' @export
propose_refinement <- function(chat, fit_result, items,
                               comparison = NULL,
                               previous_configs = NULL) {
  prompt <- format_reflection_prompt(
    fit_result, comparison = comparison,
    previous_configs = previous_configs, items = items
  )

  prompt <- paste0(
    prompt,
    "\n\n## Your Task\n",
    "Based on the diagnostics above, propose a refined model configuration.\n",
    "You may change:\n",
    "1. The block configuration (measurement, structural, population, item)\n",
    "2. The loading matrix (which items load on which skills)\n",
    "3. The skill taxonomy (add, remove, or rename skills)\n\n",
    "Explain your rationale.\n"
  )

  response_type <- refinement_output_type()

  result <- chat$chat_structured(prompt, type = response_type)

  build_config_from_refinement(result, items)
}

#' Define structured output type for refinement
#' @noRd
refinement_output_type <- function() {
  ellmer::type_object(
    measurement = ellmer::type_enum(values = MEASUREMENT_OPTIONS, description = "Measurement block choice"),
    structural = ellmer::type_enum(values = STRUCTURAL_OPTIONS, description = "Structural block choice"),
    population = ellmer::type_enum(values = POPULATION_OPTIONS, description = "Population block choice"),
    item = ellmer::type_enum(values = ITEM_OPTIONS, description = "Item block choice"),
    skills = ellmer::type_array(
      items = ellmer::type_object(
        skill_id = ellmer::type_string("Skill identifier"),
        name = ellmer::type_string("Skill name"),
        description = ellmer::type_string("Skill description")
      ),
      description = "Updated skill taxonomy"
    ),
    assignments = ellmer::type_array(
      items = ellmer::type_object(
        item_id = ellmer::type_string("Item identifier"),
        skills = ellmer::type_array(
          items = ellmer::type_string("Skill ID"),
          description = "Skills this item requires"
        )
      ),
      description = "Updated skill assignments per item"
    ),
    edge_prior = ellmer::type_array(
      items = ellmer::type_object(
        from = ellmer::type_string("Parent skill ID"),
        to = ellmer::type_string("Child skill ID"),
        prob = ellmer::type_number("Prior probability of this edge")
      ),
      description = "Edge prior for DAG mode (empty array if not DAG)"
    ),
    rationale = ellmer::type_string("Explanation of proposed changes")
  )
}

#' Build model_config from LLM refinement response
#' @noRd
build_config_from_refinement <- function(result, items) {
  taxonomy <- tibble::tibble(
    skill_id = result$skills$skill_id,
    name = result$skills$name,
    description = result$skills$description,
    is_new = TRUE
  )

  assignments_raw <- result$assignments
  assignments <- purrr::map_dfr(seq_len(nrow(assignments_raw)), function(i) {
    row <- assignments_raw[i, ]
    skill_list <- row$skills[[1]]
    tibble::tibble(
      item_id = row$item_id,
      skill_id = skill_list,
      skill_name = taxonomy$name[match(skill_list, taxonomy$skill_id)]
    )
  })

  struc <- loading_structure(taxonomy, assignments, items)

  spec <- model_spec(
    measurement = result$measurement,
    structural = result$structural,
    population = result$population,
    item = result$item
  )

  ep <- NULL
  if (result$structural == "dag" && nrow(result$edge_prior) > 0) {
    ep <- edge_prior(
      from = result$edge_prior$from,
      to = result$edge_prior$to,
      prob = result$edge_prior$prob
    )
  }

  model_config(spec, struc, edge_prior = ep)
}

#' Build the reflection system prompt
#' @noRd
build_reflection_system_prompt <- function() {
  paste(
    "You are an expert psychometrician and Bayesian modeler.",
    "You are helping refine a latent knowledge component model.",
    "You understand IRT, factor analysis, and structural equation modeling.",
    "You use diagnostic information (LOO-CV, Pareto k, R-hat, ESS) to guide model refinement.",
    "When proposing changes, explain your reasoning and be specific.",
    sep = "\n"
  )
}
