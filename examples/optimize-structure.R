library(autoskill)
library(tibble)

# --- 1. Define items ---
items <- tibble(
  item_id = paste0("item_", 1:8),
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

# --- 2. Set up a known "true" model (correlated 3-factor) ---
taxonomy <- tibble(
  skill_id = c("skill_1", "skill_2", "skill_3"),
  name = c("Linear equations", "Fraction arithmetic", "Equation setup"),
  description = c(
    "Solving and rearranging equations",
    "Operations with fractions and decimals",
    "Translating word problems into equations"
  ),
  is_new = c(TRUE, TRUE, TRUE)
)

assignments <- tibble(
  item_id = c("item_1", "item_2", "item_3", "item_4", "item_4",
              "item_5", "item_6", "item_7", "item_7", "item_8"),
  skill_id = c("skill_1", "skill_1", "skill_2", "skill_1", "skill_3",
               "skill_1", "skill_2", "skill_1", "skill_3", "skill_2"),
  skill_name = c("Linear equations", "Linear equations", "Fraction arithmetic",
                 "Linear equations", "Equation setup",
                 "Linear equations", "Fraction arithmetic",
                 "Linear equations", "Equation setup", "Fraction arithmetic")
)

true_structure <- loading_structure(taxonomy, assignments, items)
true_config <- model_config(
  model_spec(structural = "correlated"),
  true_structure
)

# --- 3. Simulate data from the true model ---
sim <- simulate_responses(true_config, n_students = 500, seed = 42)

# --- 4. Start optimization from a simpler model (independent) ---
initial_config <- model_config(model_spec(structural = "independent"), true_structure)

# --- 5. Run the optimizer (requires ANTHROPIC_API_KEY) ---
# optimize_structure_skill is the legacy wrapper around the generic
# optimize_structure(problem, ...). New code can also call:
#   optimize_structure(skill_problem(items, sim$responses, ...), ...)
result <- optimize_structure_skill(
  sim$responses,
  items,
  initial_config = initial_config,
  max_iter = 5,
  patience = 3,
  log_file = "optimization-log.jsonl"
)

print(result)
