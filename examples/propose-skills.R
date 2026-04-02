library(autoskill)
library(tibble)

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

# Open discovery (requires ANTHROPIC_API_KEY)
structure <- propose_skills(
  items,
  context = "Middle school mathematics assessment",
  n_skills = 3
)

print(structure)
