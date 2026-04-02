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


# Run skill proposer ------------------------------------------------------

result <- propose_skills(
  items,
  context = "Middle school mathematics assessment",
  n_skills = 3
)


# Inspect results ---------------------------------------------------------

result$taxonomy
# Expected output (will vary):
# # A tibble: 3 x 3
#   skill_id name                    description
#   <chr>    <chr>                   <chr>
# 1 skill_1  Linear equations        Solving and manipulating equations with one unknown
# 2 skill_2  Fraction arithmetic     Adding, converting, and computing with fractions
# 3 skill_3  Word problem modeling   Translating text descriptions into mathematical expressions

result$assignments
# Expected output (will vary):
# # A tibble: ~12 x 3
#   item_id skill_id skill_name
#   <chr>   <chr>    <chr>
# 1 item_1  skill_1  Linear equations
# 2 item_3  skill_2  Fraction arithmetic
# 3 item_4  skill_1  Linear equations
# 4 item_4  skill_3  Word problem modeling
# ...


# View as loading matrix --------------------------------------------------

loading <- to_loading_matrix(result$assignments, items, result$taxonomy)
format_loading_matrix(loading)
# Expected output (will vary):
# # A tibble: 8 x 4
#   text                            `Linear equations` `Fraction arithmetic` `Word problem modeling`
#   <chr>                           <chr>              <chr>                 <chr>
# 1 Solve: 3x + 5 = 20             x                  .                     .
# 2 Simplify: 2(x + 3) - x         x                  .                     .
# 3 Compute: 1/2 + 1/3             .                  x                     .
# 4 A train goes x km/h for 3h...  x                  .                     x
# 5 Factor: x^2 - 9                x                  .                     .
# 6 Convert 3/4 to a decimal       .                  x                     .
# 7 Two numbers sum to 20...       x                  .                     x
# 8 Compute: 3/8 + 5/8             .                  x                     .
