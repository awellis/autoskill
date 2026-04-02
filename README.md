# autoskill

LLM-driven discovery of latent skill structure in student response data.

The core model is a **confirmatory factor model with binary outcomes**: student responses (correct/incorrect) follow a Bernoulli distribution with a logit link, where continuous latent skill factors per student are mapped to item response probabilities through a factor loading matrix. The structure of this loading matrix — which entries are nonzero, how many latent dimensions exist — defines the skill structure hypothesis.

An LLM proposes these structural hypotheses (the sparsity pattern of the loading matrix), encodes them as Stan programs, and an automated evaluation pipeline scores them with Bayesian model comparison. The loop iterates: propose, validate, fit, compare, reflect, refine.

## The model

Student *j*'s response to item *i* is modeled as:

```
y_ij ~ Bernoulli(logit⁻¹(alpha_i + Lambda_i * theta_j))
```

where:

- `theta_j` is a vector of continuous latent skill factors for student *j*
- `Lambda` is the factor loading matrix (item discriminations)
- `alpha_i` is the item intercept (easiness)

The **loading matrix structure** is the search target. A fully unconstrained Lambda is unidentified without rotation constraints (the standard factor analysis problem). The LLM proposes a sparsity pattern — which items load on which skills — that constrains Lambda and makes the model identified:

```
Lambda[i,k] = 0        if item i does not require skill k
Lambda[i,k] = free     if item i requires skill k (estimate the loading)
```

This is equivalent to proposing which paths exist in a structural equation model. The number of latent dimensions and the sparsity pattern together define the structural hypothesis.

### Example

Six items on a math test. The LLM hypothesizes two latent skills: **algebra** and **word problem comprehension**.

| Item | Algebra | Word problems |
|---|---|---|
| Solve 3x + 5 = 20 | lambda_11 | 0 |
| Simplify 2(x + 3) - x | lambda_21 | 0 |
| "You have $11, apples cost $2. How many can you buy?" | 0 | lambda_32 |
| "A train goes x km/h for 3h, covering 180 km. Find x." | lambda_41 | lambda_42 |
| Factor x^2 - 9 | lambda_51 | 0 |
| "Two numbers sum to 20, differ by 4. Find them." | lambda_61 | lambda_62 |

The zeros are the hypothesis. Items 1, 2, 5 require only algebra. Item 3 requires only word problem comprehension. Items 4 and 6 require both.

Now consider two students:

- **Student A** (theta = [1.5, -1.0]): strong algebra, weak word problems. Gets items 1, 2, 5 right easily. Struggles with item 3. Items 4 and 6 could go either way — the algebra loading helps, the word problem loading hurts.
- **Student B** (theta = [-0.5, 1.5]): weak algebra, strong word problems. Opposite pattern.

If this loading structure is correct, these two students will produce systematically different response patterns that a single-factor IRT model cannot explain. The LLM's job is to discover that this two-factor structure (and this specific sparsity pattern) fits the data better than alternatives — maybe a one-factor model, maybe three factors, maybe items 4 and 6 load differently.

## Architecture

Two components, usable independently or together.

### Skill proposer

Given item text (and optionally curriculum standards, learning objectives, or domain context), an LLM performs automated cognitive task analysis in two stages:

1. **Taxonomy.** The LLM reads all items and proposes a skill taxonomy: a list of skills with names, descriptions, and a consistent level of granularity appropriate to the context. This is a reviewable artifact — a teacher or domain expert can edit the skill list before proceeding.
2. **Assignment.** The LLM assigns each item to the relevant skills from the taxonomy. This is a simpler judgment than inventing the whole structure at once — for each item-skill pair, the LLM decides whether the item requires that skill. Items can load on multiple skills (cross-loadings).

The two-stage decomposition matters. The taxonomy step is where the hard conceptual work happens (what are the skills? how fine-grained?). The assignment step is mechanical by comparison. Separating them means the taxonomy can be reviewed, edited, or provided externally (e.g., from curriculum standards) before any assignments are made.

This component is useful on its own, without any student response data. A teacher building a new test can get a first-pass skill map before any students have taken the test.

### Structure optimizer

Given a loading structure (from the skill proposer or any other source) and student response data, the optimizer evaluates and refines it through a Bayesian model comparison loop:

1. **SBC** — Simulation-based calibration checks whether the proposed model is identifiable before fitting real data. Non-identified models are rejected early, with diagnostics fed back to the LLM.
2. **Fit** — MCMC via cmdstanr (NUTS). Convergence diagnostics (R-hat, ESS, divergences, E-BFMI) determine whether inference succeeded.
3. **Compare** — LOO-CV via PSIS-LOO scores out-of-sample predictive performance with a natural complexity penalty. Pareto k diagnostics flag unreliable estimates.
4. **Reflect** — Structured feedback (poorly predicted items, pathological posteriors, parameter summaries, LOO differences) goes back to the LLM for the next iteration.

### Composition

The skill proposer provides a warm start; the structure optimizer tests whether it holds up against data. The proposer's output is a semantically informed hypothesis. The optimizer's output is an empirically validated one. Together they close the loop between what the items *should* require (semantic analysis) and what the response patterns *actually* support (statistical evidence).

## Design decisions

**Continuous latent skills.** Skills are continuous latent factors, not binary mastered/not-mastered. This is more cognitively plausible (students develop graded proficiency), connects to the well-established IRT and factor analysis literature, and keeps the model class within Stan's strengths (no discrete latent variables to marginalize).

**Stan as the modeling language.** The LLM writes `.stan` files, not Python or R model code. Stan is language-agnostic, has gold-standard NUTS inference, and LLMs generate it well due to extensive training data. This cleanly separates model specification from the evaluation harness.

**R as the evaluation engine.** The analytical stack is written once in R and reused across all proposed models. R is where the methodological developers work (Vehtari, Gabry, Bürkner, Betancourt), and the tooling reflects that.

**R as the orchestration language.** The outer loop (call LLM API, compile Stan model, run diagnostics, format feedback, repeat) is also R. The bottleneck is Stan inference (minutes per iteration), not orchestration overhead, so language speed is irrelevant. Keeping everything in one language avoids cross-runtime serialization and environment management. LLM structured output is handled via tool use in the Anthropic API (define a tool schema, get validated JSON back), which works from any language — no need for Python-specific frameworks like PydanticAI. Stan's own compiler provides a better validation loop than JSON schema: if the LLM generates invalid Stan code, the compiler error is fed back directly.

## Tech stack

| Package | Role |
|---|---|
| [cmdstanr](https://mc-stan.org/cmdstanr/) | Stan interface — compile and fit models via CmdStan |
| [SBC](https://hyunjimoon.github.io/SBC/) | Simulation-based calibration for identifiability checking |
| [loo](https://mc-stan.org/loo/) | PSIS-LOO-CV and WAIC for model comparison, Pareto k diagnostics |
| [bayesplot](https://mc-stan.org/bayesplot/) | MCMC diagnostics, posterior predictive checks, SBC rank histograms |
| [posterior](https://mc-stan.org/posterior/) | Draws format, convergence summaries, ESS, R-hat |
| [priorsense](https://n-kall.github.io/priorsense/) | Power-scaling sensitivity analysis |
| [brms](https://paul-buerkner.github.io/brms/) | Baseline IRT/multilevel models for benchmarking |
| [ellmer](https://ellmer.tidyverse.org/) | LLM API interface for R (Anthropic, OpenAI, etc.) |

## Incremental strategy

1. **Phase 1 — Skill proposer.** Build the semantic component: LLM reads item text and proposes a loading structure. No student data, no inference. Evaluate by comparing LLM-proposed structures against expert-labeled skill maps where available.
2. **Phase 2 — Loading matrix sparsity search.** Fix the model framework (confirmatory factor model, Bernoulli/logit, continuous latent factors). Use the skill proposer's output as the initial hypothesis. The optimizer searches over sparsity patterns and number of latent dimensions, scored by LOO.
3. **Phase 3 — Full structural search.** Expand the search space to include the structural model among latent factors (correlated, hierarchical, higher-order), nonlinear interaction terms, and item-level structure beyond simple loadings.

## Data

Synthetic data for controlled experiments during development. Real datasets TBD.
