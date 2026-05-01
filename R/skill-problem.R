#' Construct a structure_problem for skill discovery
#'
#' Wraps the existing skill workflow (`propose_skills()`,
#' `propose_refinement()`, `model_config()`, `check_identifiability()`,
#' `fit_cached()`) into a [structure_problem()] that the generic
#' optimisation loop can drive. The constructor binds an `ellmer` chat
#' object and the problem's data into closures over each slot.
#'
#' This is the primary application of the generic engine. Other
#' applications (causal discovery, mediation, path analysis) follow the
#' same pattern but with different proposers and scorers.
#'
#' @param items Tibble with columns `item_id` and `text`.
#' @param responses A `response_data` object.
#' @param edge_prior Optional `edge_prior` object (required for
#'   `structural = "dag"`).
#' @param chat An `ellmer` chat object. If `NULL`, one is created with
#'   the standard reflection system prompt.
#' @param model LLM model identifier (used only when `chat` is `NULL`).
#' @param interactive If `TRUE`, prompts the user to accept/reject each
#'   LLM-proposed refinement.
#' @return A `structure_problem` of subclass `"skill_problem"`.
#' @seealso [structure_problem()], [optimize_structure()]
#' @export
skill_problem <- function(items, responses,
                          edge_prior = NULL,
                          chat = NULL,
                          model = "claude-sonnet-4-20250514",
                          interactive = FALSE) {
  # Lazy chat: only created when an LLM-driven slot is first invoked. This
  # lets test code construct a skill_problem without touching the API.
  get_chat <- local({
    cached <- chat
    function() {
      if (is.null(cached)) {
        cached <<- ellmer::chat_anthropic(
          system_prompt = build_reflection_system_prompt(),
          model = model
        )
      }
      cached
    }
  })

  structure_problem(
    data = list(items = items, responses = responses),

    propose_initial = function(...) {
      cli_inform("Running skill proposer...")
      struc <- propose_skills(items, chat = get_chat())
      model_config(model_spec(), struc, edge_prior = edge_prior)
    },

    propose_refinement = function(current, fit_result, history) {
      previous_configs <- lapply(history, function(h) h$structure)
      proposed <- tryCatch(
        propose_refinement(get_chat(), fit_result, items,
                           previous_configs = previous_configs),
        error = function(e) {
          cli_warn("LLM refinement failed: {conditionMessage(e)}")
          NULL
        }
      )

      if (interactive && !is.null(proposed)) {
        cat(sprintf("\nProposed: %s/%s/%s/%s\n",
            proposed$spec$measurement, proposed$spec$structural,
            proposed$spec$population, proposed$spec$item))
        cat(sprintf("Skills: %s\n",
            paste(proposed$structure$taxonomy$name, collapse = ", ")))
        response <- readline("Accept (a), modify (m), or reject (r)? ")
        if (response == "r") proposed <- NULL
      }

      proposed
    },

    score = function(structure, cache = NULL, ...) {
      fit_cached(responses, structure, cache = cache, ...)
    },

    log_prior = function(structure) {
      # Placeholder: skill IRT priors live inside the Stan model itself.
      # Edge-level structural priors will be folded in here when the SMC
      # acceptance kernel needs them.
      0
    },

    validate = function(structure) check_identifiability(structure),

    cache_key = function(structure, fit_args) {
      fit_cache_key(responses, structure, fit_args)
    },

    summarize_structure = function(structure) {
      paste(structure$spec$measurement, structure$spec$structural,
            structure$spec$population, structure$spec$item, sep = "/")
    },

    summarize_fit = function(fit_result) {
      list(
        diagnostics = fit_result$diagnostics,
        elpd = fit_result$loo$estimates["elpd_loo", "Estimate"]
      )
    },

    class = "skill_problem"
  )
}
