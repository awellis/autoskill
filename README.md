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

## Composable model blocks

Rather than having the LLM write raw Stan code or fill in a single template, the model is decomposed into composable blocks. The LLM selects and configures blocks; the code generator composes them into valid Stan. Every combination is guaranteed to compile, and each block is independently validated for identifiability.

This gives the LLM an expressive search space while keeping inference reliable. Credit assignment is clean: if you change one block and LOO improves, you know exactly what helped.

### Block 1: Measurement model

How latent skills map to item responses. The loading matrix sparsity pattern (from the skill proposer) plus the combination rule.

| Option | Description |
|---|---|
| **linear** | `logit(p_ij) = alpha_i + Lambda_i * theta_j` — standard compensatory model. Skills contribute additively on the logit scale. |
| **interaction** | Adds pairwise interaction terms between skills that co-load on an item. `logit(p_ij) = alpha_i + Lambda_i * theta_j + theta_j' * Gamma_i * theta_j` where Gamma_i is nonzero only for skill pairs that both load on item i. |

### Block 2: Structural model

How latent skills relate to each other.

| Option | Description |
|---|---|
| **independent** | `theta_jk ~ Normal(0, 1)` independently. Simplest, most identified. |
| **correlated** | `theta_j ~ MVN(0, Sigma)` with `Sigma = L * L'`, `L ~ LKJ_cholesky(eta)`. Estimates skill correlations. |
| **hierarchical** | One or more higher-order factors feed into domain-specific skills. E.g., general math ability -> algebra, fractions. `theta_j = B * phi_j + epsilon_j` where `phi_j` are higher-order factors. |

### Block 3: Population model

How students are distributed.

| Option | Description |
|---|---|
| **single** | All students from one population. The default. |
| **grouped** | Students nested in groups (classrooms, schools). Group-level means for theta. `theta_j ~ MVN(mu_g[j], Sigma_within)`, `mu_g ~ MVN(0, Sigma_between)`. |

### Block 4: Item model

Additional item-level structure beyond intercepts and loadings.

| Option | Description |
|---|---|
| **basic** | Item intercept `alpha_i` only. The default. |
| **slip-guess** | Adds lower and upper asymptotes. `p_ij = g_i + (1 - g_i - s_i) * inv_logit(...)`. |

A model specification is a choice from each block:

```
model_spec(
  measurement = "linear",
  structural  = "correlated",
  population  = "single",
  item        = "basic"
)
```

The Stan code generator composes the blocks. The LLM searches over both the loading matrix sparsity pattern and the block configuration.

## Design decisions

**Continuous latent skills.** Skills are continuous latent factors, not binary mastered/not-mastered. This is more cognitively plausible (students develop graded proficiency), connects to the well-established IRT and factor analysis literature, and keeps the model class within Stan's strengths (no discrete latent variables to marginalize).

**Composable blocks, not free-form Stan.** The LLM selects and configures pre-validated model blocks rather than writing raw Stan. Every block combination is guaranteed to compile. This makes the search space expressive (measurement x structural x population x item options) while keeping inference reliable. Free-form Stan generation is deferred to a later phase, once the composable space is exhausted and a strong baseline exists.

**Stan as the inference engine.** Stan code is generated from block configurations, not written by humans or LLMs directly. Stan has gold-standard NUTS inference and diagnostics. The generated code is deterministic given a model specification.

**R as the evaluation and orchestration engine.** The analytical stack (cmdstanr, loo, bayesplot, SBC, posterior, priorsense) and the outer loop are both R. The bottleneck is Stan inference (minutes per iteration), not orchestration overhead. LLM structured output is handled via tool use in the Anthropic API, which works from any language — no need for Python-specific frameworks.

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
2. **Phase 2 — Measurement model search.** Fix the block configuration to `measurement = "linear", structural = "independent", population = "single", item = "basic"`. The LLM searches only over loading matrix sparsity patterns and number of latent dimensions. This is the simplest model family — standard confirmatory factor analysis with binary outcomes.
3. **Phase 3 — Block configuration search.** Open up the composable blocks. The LLM searches over both the loading matrix and the block configuration (correlated vs. hierarchical factors, slip/guess, grouped populations). Each combination is pre-validated, so the search space expands without sacrificing inference reliability.
4. **Phase 4 — Free-form Stan.** For model structures that can't be expressed as block combinations, the LLM writes raw Stan code. This is the fully open-ended search from the original proposal. SBC becomes the primary safety net.

## Data

Synthetic data for controlled experiments during development. Real datasets TBD.
