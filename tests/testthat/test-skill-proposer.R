test_that("build_taxonomy_prompt includes items", {
  items <- make_test_items()
  prompt <- build_taxonomy_prompt(items)
  expect_match(prompt, "3x \\+ 5 = 20")
})

test_that("build_taxonomy_prompt includes known skills", {
  items <- make_test_items()
  known <- tibble::tibble(name = "Algebra", description = "Basic algebra")
  prompt <- build_taxonomy_prompt(items, known_skills = known)
  expect_match(prompt, "Algebra")
})
