library(ellmer)
library(tidyverse)


# Type definitions --------------------------------------------------------

skill_type <- type_object(
  name = type_string("Short descriptive name for the skill"),
  description = type_string("What this skill involves, in one sentence"),
  is_new = type_boolean("TRUE if this skill was newly proposed, FALSE if from the known set")
)

assignment_type <- type_object(
  item_id = type_string("The item identifier"),
  skills = type_array(
    items = type_string("Name of a required skill"),
    description = "Skills this item requires"
  )
)


# Prompt construction -----------------------------------------------------

build_taxonomy_prompt <- function(items, known_skills = NULL,
                                  context = NULL, n_skills = NULL) {
  item_list <- items |>
    glue::glue_data("- {item_id}: {text}") |>
    str_c(collapse = "\n")

  has_known <- !is.null(known_skills) && nrow(known_skills) > 0

  known_block <- if (has_known) {
    skill_list <- known_skills |>
      glue::glue_data("- {name}: {description}") |>
      str_c(collapse = "\n")

    str_c(
      "The following skills are already known. Use these wherever they apply.",
      "Only propose a new skill if an item genuinely requires something not covered by the known set.",
      "Mark each skill with is_new = FALSE if from this list, is_new = TRUE if newly proposed.",
      "",
      "Known skills:",
      skill_list,
      sep = "\n"
    )
  }

  parts <- c(
    "Analyze these test items and identify the distinct latent skills required to solve them.",
    if (!is.null(context)) glue::glue("Context: {context}"),
    if (!is.null(n_skills) && !has_known) glue::glue("Propose approximately {n_skills} skills."),
    known_block,
    if (!has_known) c(
      "Skills should be at a consistent level of granularity.",
      "Too broad (e.g. 'math') is useless. Too narrow (e.g. 'multiplying 3-digit numbers') overfits."
    ),
    "Each item may require multiple skills.",
    "",
    "Items:",
    item_list
  )

  str_c(parts, collapse = "\n")
}

build_assignment_prompt <- function(items, taxonomy) {
  item_list <- items |>
    glue::glue_data("- {item_id}: {text}") |>
    str_c(collapse = "\n")

  skill_list <- taxonomy |>
    glue::glue_data("- {name}: {description}") |>
    str_c(collapse = "\n")

  glue::glue("
    For each item, identify which of the following skills are required to solve it.
    An item can require multiple skills. Only assign a skill if it is genuinely needed.
    Use the exact skill names provided.

    Skills:
    {skill_list}

    Items:
    {item_list}
  ")
}


# Core functions ----------------------------------------------------------

propose_taxonomy <- function(chat, items, known_skills = NULL,
                             context = NULL, n_skills = NULL) {
  prompt <- build_taxonomy_prompt(items, known_skills, context, n_skills)

  chat$chat_structured(
    prompt,
    type = type_array(items = skill_type, description = "List of identified skills")
  ) |>
    mutate(skill_id = str_c("skill_", row_number()), .before = 1)
}

assign_skills <- function(chat, items, taxonomy) {
  prompt <- build_assignment_prompt(items, taxonomy)

  chat$chat_structured(
    prompt,
    type = type_array(items = assignment_type, description = "Skill assignments per item")
  ) |>
    unnest(skills) |>
    rename(skill_name = skills) |>
    left_join(taxonomy |> select(skill_id, name), by = c("skill_name" = "name")) |>
    select(item_id, skill_id, skill_name)
}

propose_skills <- function(items, known_skills = NULL, context = NULL,
                           n_skills = NULL, model = "claude-sonnet-4-20250514") {
  chat <- chat_anthropic(
    system_prompt = "You are an expert in cognitive task analysis and psychometrics.",
    model = model
  )

  taxonomy <- propose_taxonomy(chat, items, known_skills, context, n_skills)
  assignments <- assign_skills(chat, items, taxonomy)

  list(taxonomy = taxonomy, assignments = assignments)
}


# Utilities ---------------------------------------------------------------

to_loading_matrix <- function(assignments, items, taxonomy) {
  crossing(
    item_id = items$item_id,
    skill_id = taxonomy$skill_id
  ) |>
    left_join(
      assignments |> mutate(loads = TRUE),
      by = c("item_id", "skill_id")
    ) |>
    replace_na(list(loads = FALSE)) |>
    left_join(items |> select(item_id, text), by = "item_id") |>
    left_join(taxonomy |> select(skill_id, name), by = "skill_id")
}

format_loading_matrix <- function(loading_matrix) {
  loading_matrix |>
    mutate(loads = if_else(loads, "\u00d7", "\u00b7")) |>
    select(text, name, loads) |>
    pivot_wider(names_from = name, values_from = loads)
}
