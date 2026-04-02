# Type definitions --------------------------------------------------------

skill_type <- ellmer::type_object(
  name = ellmer::type_string("Short descriptive name for the skill"),
  description = ellmer::type_string("What this skill involves, in one sentence"),
  is_new = ellmer::type_boolean("TRUE if this skill was newly proposed, FALSE if from the known set")
)

assignment_type <- ellmer::type_object(
  item_id = ellmer::type_string("The item identifier"),
  skills = ellmer::type_array(
    items = ellmer::type_string("Name of a required skill"),
    description = "Skills this item requires"
  )
)

# Prompt construction -----------------------------------------------------

#' @noRd
build_taxonomy_prompt <- function(items, known_skills = NULL,
                                  context = NULL, n_skills = NULL) {
  item_list <- items |>
    glue::glue_data("- {item_id}: {text}") |>
    stringr::str_c(collapse = "\n")

  has_known <- !is.null(known_skills) && nrow(known_skills) > 0

  known_block <- if (has_known) {
    skill_list <- known_skills |>
      glue::glue_data("- {name}: {description}") |>
      stringr::str_c(collapse = "\n")

    stringr::str_c(
      "The following skills are already known. Use these wherever they apply.",
      "Only propose a new skill if an item genuinely requires something not covered by the known set.",
      "Mark each skill with is_new = FALSE if from this list, is_new = TRUE if newly proposed.",
      "", "Known skills:", skill_list, sep = "\n"
    )
  }

  parts <- c(
    "Analyze these test items and identify the distinct latent skills required to solve them.",
    if (!is.null(context)) paste0("Context: ", context),
    if (!is.null(n_skills) && !has_known) paste0("Propose approximately ", n_skills, " skills."),
    known_block,
    if (!has_known) c(
      "Skills should be at a consistent level of granularity.",
      "Too broad (e.g. 'math') is useless. Too narrow (e.g. 'multiplying 3-digit numbers') overfits."
    ),
    "Each item may require multiple skills.",
    "", "Items:", item_list
  )

  stringr::str_c(parts, collapse = "\n")
}

#' @noRd
build_assignment_prompt <- function(items, taxonomy) {
  item_list <- items |>
    glue::glue_data("- {item_id}: {text}") |>
    stringr::str_c(collapse = "\n")

  skill_list <- taxonomy |>
    glue::glue_data("- {name}: {description}") |>
    stringr::str_c(collapse = "\n")

  paste0(
    "For each item, identify which of the following skills are required to solve it.\n",
    "An item can require multiple skills. Only assign a skill if it is genuinely needed.\n",
    "Use the exact skill names provided.\n\n",
    "Skills:\n", skill_list, "\n\n",
    "Items:\n", item_list
  )
}

# Core functions ----------------------------------------------------------

#' Propose a skill taxonomy from item text
#' @param chat An ellmer chat object.
#' @param items Tibble with `item_id` and `text`.
#' @param known_skills Optional tibble of known skills.
#' @param context Optional domain context string.
#' @param n_skills Optional target number of skills.
#' @return A tibble with `skill_id`, `name`, `description`, `is_new`.
#' @export
propose_taxonomy <- function(chat, items, known_skills = NULL,
                             context = NULL, n_skills = NULL) {
  prompt <- build_taxonomy_prompt(items, known_skills, context, n_skills)
  chat$chat_structured(
    prompt,
    type = ellmer::type_array(items = skill_type, description = "List of identified skills")
  ) |>
    dplyr::mutate(skill_id = stringr::str_c("skill_", dplyr::row_number()), .before = 1)
}

#' Assign skills to items
#' @param chat An ellmer chat object.
#' @param items Tibble with `item_id` and `text`.
#' @param taxonomy Tibble from `propose_taxonomy()`.
#' @return A tibble with `item_id`, `skill_id`, `skill_name`.
#' @export
assign_skills <- function(chat, items, taxonomy) {
  prompt <- build_assignment_prompt(items, taxonomy)
  chat$chat_structured(
    prompt,
    type = ellmer::type_array(items = assignment_type, description = "Skill assignments per item")
  ) |>
    tidyr::unnest(skills) |>
    dplyr::rename(skill_name = skills) |>
    dplyr::left_join(taxonomy |> dplyr::select(skill_id, name), by = c("skill_name" = "name")) |>
    dplyr::select(item_id, skill_id, skill_name)
}

#' Propose skills for a set of items
#' @param items Tibble with `item_id` and `text`.
#' @param known_skills Optional tibble of known skills.
#' @param context Optional domain context string.
#' @param n_skills Optional target number of skills.
#' @param chat Optional pre-configured chat object.
#' @param model LLM model name.
#' @return A `loading_structure` object.
#' @export
propose_skills <- function(items, known_skills = NULL, context = NULL,
                           n_skills = NULL, chat = NULL,
                           model = "claude-sonnet-4-20250514") {
  if (is.null(chat)) {
    chat <- ellmer::chat_anthropic(
      system_prompt = "You are an expert in cognitive task analysis and psychometrics.",
      model = model
    )
  }
  taxonomy <- propose_taxonomy(chat, items, known_skills, context, n_skills)
  assignments <- assign_skills(chat, items, taxonomy)
  loading_structure(taxonomy, assignments, items)
}
