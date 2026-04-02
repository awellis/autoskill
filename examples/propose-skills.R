source("R/skill-proposer.R")


# Define items ------------------------------------------------------------

items <- tibble(
  item_id = str_c("item_", 1:8),
  text = c(
    "Solve: 3x + 5 = 20",
    "Simplify: 2(x + 3) - x",
    "Compute: 1/2 + 1/3",
    "A train goes x km/h for 3 hours, covering 180 km. Find x.",
    "Factor: x^2 - 9",
    "Convert 3/4 to a decimal",
    "Two numbers sum to 20 and differ by 4. Find them.",
    "Compute: 3/8 + 5/8"
  )
)


# Open discovery (no known skills) ----------------------------------------

result_open <- propose_skills(
  items,
  context = "Middle school mathematics assessment",
  n_skills = 3
)

result_open$taxonomy
result_open$assignments

loading_open <- to_loading_matrix(result_open$assignments, items, result_open$taxonomy)
format_loading_matrix(loading_open)


# Constrained discovery (known skills, novel items) -----------------------

known_skills <- tibble(
  name = c("Linear equations", "Fraction arithmetic"),
  description = c(
    "Solving and rearranging equations with one unknown",
    "Adding, converting, and computing with fractions and decimals"
  )
)

result_constrained <- propose_skills(
  items,
  known_skills = known_skills,
  context = "Middle school mathematics assessment"
)

# Taxonomy now shows which skills are known vs. newly proposed
result_constrained$taxonomy
result_constrained$taxonomy |> filter(is_new)

loading_constrained <- to_loading_matrix(
  result_constrained$assignments, items, result_constrained$taxonomy
)
format_loading_matrix(loading_constrained)
