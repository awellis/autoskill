# autoskill R Package Design Spec

## Purpose

Build a state-of-the-art R package for LLM-driven discovery of latent knowledge component (KC) structures in student response data. The package combines cognitive task analysis via LLMs with multilevel IRT evaluation via Stan, using composable model blocks and Bayesian model comparison to search over structural hypotheses.

## Scope

v0.1.0 ships the full system described in the README:

- Skill proposer (refactored from existing code)
- Structure optimizer with LLM-in-the-loop refinement
- All 64 valid combinations (2 measurement x 4 structural x 2 population x 2 item x 2 link)
- Both autonomous and interactive optimizer modes
- PSIS-LOO for model scoring

## User-Facing Workflows

### Workflow 1: Skill proposer (standalone)

```r
items <- tibble(item_id = ..., text = ...)
structure <- propose_skills(items, context = "Middle school math", n_skills = 3)
print(structure)
```

LLM reads item text, proposes a taxonomy of KCs, then assigns items to KCs. Returns a `loading_structure` object. Useful on its own without student data.

### Workflow 2: Manual model fitting and comparison

```r
config_a <- model_config(model_spec(structural = "independent"), structure)
config_b <- model_config(model_spec(structural = "correlated"), structure)

fit_a <- fit_model(responses, config_a)
fit_b <- fit_model(responses, config_b)

compare_models(independent = fit_a, correlated = fit_b)
```

User manually specifies model configurations, fits them, and compares via LOO-CV. Full control over the search.

### Workflow 3: Automated optimization loop

```r
result <- optimize_structure(
  responses, items,
  initial_config = config_a,
  max_iter = 10,
  patience = 3,
  interactive = FALSE,
  log_file = "log.jsonl"
)
```

Iteratively fits models, compares by LOO, and uses an LLM to propose refined configurations. Runs autonomously or with human-in-the-loop checkpoints (`interactive = TRUE`).

All three workflows compose and share the same data structures.

## S3 Class Hierarchy

All plain lists with class attributes. No R6 or S4.

### model_spec

What blocks are selected for each model component.

| Field | Type | Values |
|---|---|---|
| measurement | character(1) | "linear", "interaction" |
| structural | character(1) | "independent", "correlated", "dag", "hierarchical" |
| population | character(1) | "single", "grouped" |
| item | character(1) | "basic", "slip_guess" |
| link | character(1) | "logit", "probit" |

Constructor validates that all fields are from the allowed set. Query helpers: `is_sem_mode()`, `is_fa_mode()`. Enumeration: `all_valid_specs()` returns a 64-row tibble (2x4x2x2x2).

### loading_structure

Which items load on which skills. Produced by `propose_skills()` or constructed manually.

| Field | Type | Description |
|---|---|---|
| taxonomy | tibble | skill_id, name, description |
| assignments | tibble | item_id, skill_id, skill_name |
| items | tibble | item_id, text |
| lambda_mask | logical matrix [I x K] | TRUE where item loads on skill |
| n_loadings | integer | number of nonzero entries in lambda_mask |

Validation: every item has >= 1 skill, every skill has >= 2 items (identifiability), no duplicate loading patterns.

### edge_prior

Prior probabilities for directed edges among skills (SEM mode only).

| Field | Type | Description |
|---|---|---|
| edges | tibble | from, to, prob (each in [0, 1]) |

Validation: no self-loops, from/to are valid skill IDs, probabilities in [0, 1].

### response_data

Student response matrix. Accepts wide (matrix J x I) or long (tibble with student_id, item_id, correct).

| Field | Type | Description |
|---|---|---|
| Y | integer matrix [J x I] | 0/1 responses, NA for missing |
| item_ids | character(I) | column names |
| student_ids | character(J) | row names |
| n_students | integer | J |
| n_items | integer | I |

Validation: all non-NA values are 0 or 1.

### model_config

Complete hypothesis: bundles a model_spec with a loading_structure and optional edge_prior.

| Field | Type | Description |
|---|---|---|
| spec | model_spec | block choices |
| structure | loading_structure | item-skill mapping |
| edge_prior | edge_prior or NULL | required when structural = "dag" |

Validation: cross-checks (DAG requires edge_prior, edge_prior skills match taxonomy). `config_hash()` computes a content hash for caching compiled Stan models.

### fit_result

Output of model fitting. Wraps cmdstanr output with diagnostics and LOO.

| Field | Type | Description |
|---|---|---|
| fit | CmdStanMCMC | raw cmdstanr fit object |
| config | model_config | the configuration that was fitted |
| diagnostics | tibble | metric, value, status (ok/warning/critical) |
| loo | loo object | PSIS-LOO cross-validation result |
| param_summary | tibble | posterior summaries mapped to item/skill names |

## Data Flow

```
items + LLM -> loading_structure
                    +
              model_spec  ->  model_config  ->  generate_stan_code()
                                   +            prepare_stan_data()
              response_data  ------+            compile_model()
                                                      |
                                                  fit_model()
                                                      |
                                                  fit_result
                                                      |
                                            compare_models()
                                            format_reflection_prompt()
                                                      |
                                            propose_refinement() [LLM]
                                                      |
                                              new model_config
                                                   (loop)
```

## Stan Code Generation

### Architecture

Flat fragment composition (brms pattern). Each block generator returns a named list with 7 keys:

```r
list(functions, data, transformed_data, parameters,
     transformed_parameters, model, generated_quantities)
```

`collapse_stan_lists()` merges fragments by concatenating same-named strings. `assemble_stan_program()` wraps the merged result in Stan block syntax.

### Block contracts

Blocks interact through shared variable names:

- **Structural block** defines `theta` (matrix[J, K]): declares it, sets up its parameterization
- **Measurement block** reads `theta`, defines `Lambda` (matrix[I, K]) and `alpha` (vector[I]), generates the likelihood and log_lik
- **Item block** (slip-guess) wraps the measurement block's likelihood via a `compute_prob()` Stan function. The measurement block always generates a `compute_prob()` user-defined function in the `functions` block. When item = "basic", this function is used directly. When item = "slip_guess", the item block replaces the `compute_prob()` definition with a wrapper that adds asymptote parameters: `guess[i] + (1 - guess[i] - slip[i]) * base_compute_prob(...)`.
- **Population block** (grouped) adds group-level means that shift theta. When combined with structural blocks: for independent/correlated, grouped adds `mu_group[group[j]]` to theta after the structural draw. For dag/hierarchical, group means are added to the root nodes only (nodes with no parents in the DAG, or the higher-order factor in hierarchical). The population block reads the structural block's theta and modifies it.
- **Link function** chosen by measurement block: `inv_logit` vs `Phi`

### Block generators (16 functions)

**Measurement:**
- `stan_measurement_linear()`: Standard compensatory IRT. Sparse loading parameterization with `vector<lower=0>[N_loadings] lambda_free` and index arrays. Generates `compute_prob()` function for composability.
- `stan_measurement_interaction()`: Extends linear with pairwise interaction terms (`gamma * theta_k1 * theta_k2`) for cross-loading items.

**Structural:**
- `stan_structural_independent()`: `theta[j,k] ~ std_normal()` independently.
- `stan_structural_correlated()`: Non-centered parameterization. Cholesky factor `L_Omega ~ lkj_corr_cholesky(2)`, `sigma_theta ~ normal(0,1)`, `z_theta ~ std_normal()`. Theta computed as `(diag_pre_multiply(sigma_theta, L_Omega) * z_theta)'`.
- `stan_structural_dag()`: DAG among skills. Structural coefficients `B_free`, topologically sorted fill loop. Data inputs: `edge_from`, `edge_to`, `topo_order`.
- `stan_structural_hierarchical()`: Single higher-order factor `phi` feeding into K domain factors via loadings `beta_hier`.

**Population:**
- `stan_population_single()`: No-op (empty fragment).
- `stan_population_grouped()`: Declares `N_groups`, `group[J]`, `mu_group[N_groups, K]`. Adds group-level prior on theta.

**Item:**
- `stan_item_basic()`: No-op.
- `stan_item_slip_guess()`: Declares `guess[I]` and `slip[I]` with Beta(1,9) priors. Overrides `compute_prob()` to wrap: `guess[i] + (1 - guess[i] - slip[i]) * base_prob`.

**Dispatchers:** `stan_measurement()`, `stan_structural()`, `stan_population()`, `stan_item()` dispatch by `config$spec$*`.

### Key Stan parameterization choices

- **Sparse loadings**: Index arrays `loading_item[n]`, `loading_skill[n]` map into `Lambda[I, K]`. Nonzero entries are `<lower=0>` for sign identification.
- **Non-centered parameterization** for correlated and hierarchical blocks: avoids funnel geometry that causes NUTS divergences.
- **Topological sort** for DAG: parents computed before children, enabling a single forward pass.
- **log_lik** always in generated quantities for PSIS-LOO.

### Compilation invariant

Every valid `model_spec` combination (64 total) produces compilable Stan code. Enforced by integration tests. Content-hash caching avoids recompilation of identical models.

## Optimizer Loop

### Iteration structure

```
for each iteration:
  1. check_identifiability(config)     quick structural pre-check
  2. fit_model(responses, config)      compile, sample, diagnostics, LOO
  3. compare with best so far          ELPD difference
  4. log_iteration() to JSONL          iter, elpd, improved, diagnostics
  5. if interactive: present and wait  user approves/modifies/rejects
  6. if patience exhausted: stop       N consecutive non-improving
  7. propose_refinement(chat, ...)     LLM structured output -> new config
```

### LLM reflection

`format_reflection_prompt()` compiles:
- Current block configuration
- Loading matrix (item x skill table with item text)
- Diagnostics (divergences, R-hat, ESS, ELPD)
- Problem observations (high Pareto k)
- Comparison table vs previous models
- History of configs already tried

The LLM returns structured output (via ellmer `type_object`):
- New block choices (measurement, structural, population, item)
- Updated skill taxonomy and assignments
- Edge prior (if DAG)
- Rationale (free text)

The response is validated and converted to a `model_config`.

### Interactive mode

When `interactive = TRUE`, after each iteration the optimizer:
1. Prints the diagnostic summary
2. Prints the LLM's proposed refinement with rationale
3. Asks the user to accept, modify, or reject
4. If modified, uses the user's version; if rejected, tries a different direction

### Stopping rules

- `patience` consecutive non-improving iterations (default 3)
- `max_iter` hard cap (default 10)
- Critical diagnostic failure (all chains divergent)

### JSONL logging

Each iteration appended as one JSON line:
```json
{"iter": 1, "elpd": -150.3, "improved": true, "config": "linear/correlated/single/basic",
 "n_divergences": 0, "max_rhat": 1.002, "rationale": "initial model", "timestamp": "..."}
```

## Simulation

`simulate_responses(config, n_students, seed)` generates synthetic data by following the generative process: draw theta per structural block, compute linear predictor, apply link, apply slip-guess if applicable, sample Bernoulli responses. Returns `list(responses, params, config)`.

`sbc_generator(config, n_students)` returns a function compatible with `SBC::SBC_generator_function()` for simulation-based calibration.

## File Organization

```
R/
  autoskill-package.R        package docs, global imports
  utils.R                    collapse_stan_lists(), topological_sort(), validate_dag()
  model-spec.R               model_spec S3 class + validation + query helpers
  loading-structure.R         loading_structure + edge_prior S3 classes
  response-data.R             response_data S3 class (wide/long)
  model-config.R              model_config S3 class + config_hash()
  stan-blocks.R               16 block-specific Stan fragment generators
  stan-generator.R            generate_stan_code(), assemble_stan_program()
  stan-data.R                 prepare_stan_data()
  stan-compile.R              compile_model() with content-hash caching
  simulate-data.R             simulate_responses(), sbc_generator()
  sbc-check.R                 check_identifiability(), run_sbc()
  model-fit.R                 fit_model(), extract_diagnostics(), compute_loo()
  model-compare.R             compare_models(), flag_problem_items(), format_reflection_prompt()
  reflection.R                propose_refinement(), structured output types, build_config_from_refinement()
  skill-proposer.R            propose_skills(), propose_taxonomy(), assign_skills() (refactored existing)
  structure-optimizer.R       optimize_structure(), run_iteration(), log_iteration()
```

All filenames use dashes (kebab-case).

## Dependencies

```
Imports: tibble, dplyr, tidyr, purrr, stringr, glue, rlang, cli, ellmer, cmdstanr, posterior, loo, jsonlite
Suggests: testthat (>= 3.0.0), SBC, bayesplot, priorsense, brms, withr, knitr, rmarkdown
```

cmdstanr in Imports (core engine). SBC/bayesplot/priorsense in Suggests (optional).

## Testing Strategy

Four tiers:

1. **Unit tests (fast, no Stan):** ~50 tests. model_spec validation, loading_structure construction, lambda_mask, DAG validation, response_data conversion, stan fragment content checks, collapse_stan_lists, config_hash. Run in seconds.

2. **Compilation tests (skip_on_cran):** Generate and compile all 32 block combinations. The critical correctness gate. ~30s each, cached.

3. **Parameter recovery tests (slow, skip_on_cran):** Simulate from known parameters, fit, verify estimates within 95% CIs. Simulate from correlated, compare independent vs correlated, verify LOO prefers correlated. 2-5 minutes.

4. **Mocked LLM tests:** Mock `chat$chat_structured()` to return fixed responses. Test skill proposer and reflection pipelines without API calls.

SBC is available as `run_sbc()` for manual validation but not in CI (too slow).

## Patterns Borrowed

| Pattern | Source | Adaptation |
|---|---|---|
| Named-list fragment composition | brms `collapse_lists()` | `collapse_stan_lists()` with 7 fixed keys |
| S3 dispatch pipeline | brms formula -> terms -> frame -> stancode | model_spec -> model_config -> validate -> generate |
| Content-hash caching | novel (brms recompiles) | Cache compiled models by Stan code hash |
| JSONL iteration logging | autostan log.jsonl | Same format with ELPD instead of NLPD |
| Patience-based stopping | autostan (3 non-improving) | Configurable patience parameter |
| Diagnostic feedback to LLM | autostan report format | Richer: LOO table, Pareto k, loading matrix |

## Non-Goals for v0.1.0

- pkgdown website (later)
- Vignettes beyond a basic intro (later)
- Temporal/sequential data (future extension per README)
- Free-form Stan mode (phase 5 in README incremental strategy)
- GPU/threading support for Stan
- Multiple LLM provider support (ellmer handles this already)
