# autoskill

LLM-driven Bayesian structure search. Skill discovery from student responses is the founding application; the engine is generic.

The package solves problems of the shape: you have data, you have hypotheses about a latent structure (which skills exist, which causes which, which mediators sit on which paths), and you want a posterior over candidate structures rather than a single best guess. An LLM proposes structural hypotheses, a Bayesian model-comparison pipeline scores them, and the engine refines via either greedy search with stacking or sequential Monte Carlo with Barker MH.

Two applications ship out of the box:

- **`skill_problem`**: discover latent knowledge components from student response data. Multilevel IRT measurement model, optional DAG among the latent skills (SEM), LLM-driven skill proposer.
- **`causal_problem`**: discover a DAG over observed continuous variables. Per-node Gaussian regression with BIC scoring, random local-move mutation kernel.

Adding a third application (mediation, path analysis, network meta-analysis) is one constructor that fills the `structure_problem` slots.

See the **Getting Started** vignette (`vignette("getting-started", package = "autoskill")`) for a runnable end-to-end walkthrough using `causal_problem` on synthetic data.

## The skill problem

Given a dataset of student responses to test items, autoskill discovers which latent knowledge components (KCs) best account for the data. Two modes:

- Factor analysis: discover which KCs exist, which items load on which KCs, and how KCs correlate. No directed dependencies.
- SEM: discover both the measurement model (loadings) and a DAG of directed causal dependencies among KCs on the latent scale.

Factor analysis is a special case of SEM (the DAG has no edges). In both modes, an LLM proposes structural hypotheses, an evaluation pipeline scores them with Bayesian model comparison, and the LLM refines based on diagnostics.

## The model

The observation model is multilevel IRT. Student *j*'s response to item *i*:

```
y_ij ~ Bernoulli(link⁻¹(alpha_i + Lambda_i * theta_j))
```

- `theta_j`: continuous latent KC vector for student *j*, drawn from a population distribution (partial pooling)
- `Lambda`: factor loading matrix (which items load on which KCs)
- `alpha_i`: item intercept (easiness)
- `link`: logit or probit (user setting)

### Measurement model (both modes)

The loading matrix sparsity pattern is the first search target:

```
Lambda[i,k] = 0        if item i does not require KC k
Lambda[i,k] = free     if item i requires KC k
```

### Structural model (SEM mode)

The LLM proposes a DAG among KCs with directed paths on the latent scale:

```
theta_jk = sum(B[k,l] * theta_jl for l in parents(k)) + epsilon_jk
```

`B` encodes structural coefficients (nonzero where the DAG has an edge). Factor analysis mode sets `B = 0` and estimates a free correlation matrix instead.

### Prior over DAGs

Observational data alone cannot distinguish between Markov-equivalent DAGs. The LLM's domain knowledge breaks this ambiguity by providing an informative prior over DAG structures.

The prior is specified at the edge level. For each possible directed edge (KC_a -> KC_b), the LLM assigns a probability that the edge exists and a probability for each direction. These compose into a DAG prior by treating edges as approximately independent, with acyclicity enforced computationally.

```r
edge_prior <- tibble(
  from = c("number_sense", "number_sense", "algebra"),
  to   = c("algebra", "fractions", "word_problems"),
  prob = c(0.85, 0.75, 0.40)
)
```

Each judgment is simple, auditable, and revisable. The prior covers the full space of DAGs (any structure has nonzero probability as long as no edge is assigned exactly 0 or 1), so the optimizer is not restricted to a small set of LLM-proposed candidates. The data updates the prior via model comparison: DAGs consistent with both domain knowledge and statistical evidence are preferred.

### Temporal structure

Ignored initially. Responses are treated as a batch, exchangeable given the student's KC vector. Temporal data (attempt sequences) would provide additional evidence for causal direction and is a natural future extension.

### Example

Eight items on a middle school math test. The LLM hypothesizes three KCs: linear equations, fraction operations, and equation setup (translating a problem description into a solvable equation).

| Item | Linear eq. | Fractions | Eq. setup |
|---|---|---|---|
| Solve: 3x + 5 = 20 | lambda_11 | 0 | 0 |
| Simplify: 2(x + 3) - x | lambda_21 | 0 | 0 |
| Compute: 1/2 + 1/3 | 0 | lambda_32 | 0 |
| A train goes x km/h for 3h, covering 180 km. Find x. | lambda_41 | 0 | lambda_43 |
| Factor: x^2 - 9 | lambda_51 | 0 | 0 |
| Convert 3/4 to a decimal | 0 | lambda_62 | 0 |
| Two numbers sum to 20, differ by 4. Find them. | lambda_71 | 0 | lambda_73 |
| Compute: 3/8 + 5/8 | 0 | lambda_82 | 0 |

Items 4 and 7 cross-load on both linear equations and equation setup: a student needs to formulate the equation from the problem description *and* solve it. A student who can solve 3x + 5 = 20 but fails the train problem may have the algebra but lack the ability to set up the equation.

In SEM mode, the LLM might further propose that equation setup depends on linear equations (you need to know what a solvable equation looks like before you can construct one), giving a directed edge in the structural model.

## Architecture

Two components, usable independently or together.

### Skill proposer

Given item text (and optionally curriculum standards or domain context), an LLM performs automated cognitive task analysis in two stages:

1. Taxonomy: read all items, propose a list of KCs with names and descriptions at a consistent level of granularity. This is a reviewable artifact that can be edited before proceeding.
2. Assignment: for each item, decide which KCs from the taxonomy it requires. Items can load on multiple KCs.

Separating these steps means the taxonomy can be reviewed, edited, or provided externally before any assignments are made. The skill proposer is useful on its own, without student response data.

### Structure optimizer

Given a loading structure and student response data, the optimizer evaluates and refines it through a model comparison loop:

1. SBC: simulation-based calibration checks identifiability. Non-identified models are rejected early.
2. Fit: MCMC via cmdstanr (NUTS). Convergence diagnostics (R-hat, ESS, divergences, E-BFMI) determine whether inference succeeded.
3. Compare: LOO-CV via PSIS-LOO scores out-of-sample predictive performance.
4. Reflect: structured feedback (problem items, pathological posteriors, LOO differences) goes back to the LLM.

### Composition

The skill proposer provides a semantically informed warm start. The structure optimizer tests whether it holds up against data. Together they close the loop between what items *should* require (semantic analysis) and what response patterns *actually* support (statistical evidence).

## The generic engine

Skill discovery is one instance of a more general pattern: LLM-driven Bayesian structure search. The engine is exposed as `structure_problem`, an S3 class whose closures define every domain-specific piece (proposers, scorer, validator, prior, summarisers). The optimisation algorithms drive *any* `structure_problem`.

### `structure_problem` interface

| Slot | Type | Purpose |
|---|---|---|
| `propose_initial` | `() -> structure` | Starting structure |
| `propose_refinement` | `(current, fit, history) -> structure \| NULL` | Greedy refinement |
| `propose_local_move` | `(structure, ...) -> structure` | SMC mutation kernel |
| `score` | `(structure, cache, ...) -> fit_result` | Fit + return LOO |
| `log_prior` | `(structure) -> numeric` | Domain prior |
| `validate` | `(structure) -> list(passed, problems)` | Pre-fit identifiability check |
| `cache_key` | `(structure, fit_args) -> chr` | Hash for caching |
| `summarize_structure` | `(structure) -> chr` | Short label for logs |
| `summarize_fit` | `(fit_result) -> list` | Reflection-prompt input |

Domain constructors fill the slots. `skill_problem(items, responses, ...)` and `causal_problem(data, ...)` are the two shipped instances.

### Two algorithms

**Greedy with stacking** (`optimize_structure`) follows ELPD-improving local moves with patience-based stopping. Returns the best structure plus stacking weights over visited iterations. Cheap, useful when one good structure is enough.

**Sequential Monte Carlo** (`optimize_structure_smc`) maintains a particle population approximating the tempered posterior `pi_t(S) ~ p(S) * exp(gamma_t * elpd(S))`. Each step reweights particles to the next temperature (closed form, no refit), resamples on low ESS, and mutates each particle via Barker MH. Returns particles, posterior weights, and `edge_marginals()` / `structure_marginal()` summaries.

### Caching and parallel fits

`fit_cache(dir = "fit_cache")` builds a disk-backed cache that survives between sessions. `optimize_structure(problem, cache = ...)` and `optimize_structure_smc(problem, cache = ...)` both honour it. Many SMC particles converge to identical structures after resampling; the cache makes this free instead of expensive. `fit_many(responses, configs, cache = ...)` runs a batch of fits in parallel via `furrr` for the cases where you need it.

## Composable model blocks

The model is decomposed into composable blocks. The LLM selects and configures blocks; the code generator composes them into valid Stan. Every combination is guaranteed to compile and independently validated for identifiability.

### Block 1: Measurement model

| Option | Description |
|---|---|
| linear | `link(p_ij) = alpha_i + Lambda_i * theta_j`. Standard compensatory model, additive on the link scale. |
| interaction | Adds pairwise interaction terms between KCs that co-load on an item. |

### Block 2: Structural model

Factor analysis mode:

| Option | Description |
|---|---|
| independent | `theta_jk ~ Normal(0, 1)` independently. Simplest. |
| correlated | `theta_j ~ MVN(0, Sigma)` with LKJ prior on the correlation matrix. |

SEM mode:

| Option | Description |
|---|---|
| dag | `theta_jk = sum(B[k,l] * theta_jl for l in parents(k)) + epsilon_jk`. The LLM proposes both the DAG and the loading matrix. |
| hierarchical | Special case: higher-order factors feed into domain-specific KCs. |

### Block 3: Population model

| Option | Description |
|---|---|
| single | All students from one population. |
| grouped | Students nested in groups (classrooms, schools) with group-level means. |

### Block 4: Item model

| Option | Description |
|---|---|
| basic | Item intercept only. |
| slip-guess | Lower and upper asymptotes: `p_ij = g_i + (1 - g_i - s_i) * inv_link(...)`. |

A model specification:

```
model_spec(
  mode        = "sem",
  measurement = "linear",
  structural  = "dag",
  population  = "single",
  item        = "basic",
  link        = "logit"
)
```

## Design decisions

Continuous latent KCs (not binary mastered/not-mastered) connect to IRT and factor analysis, are more cognitively plausible, and keep the model within Stan's strengths.

Composable blocks rather than free-form Stan. Every block combination compiles. Credit assignment is clean: change one block, observe the effect on LOO. Free-form Stan is deferred to a later phase.

Stan for inference. Generated from block configurations, not written by humans or LLMs. Gold-standard NUTS and diagnostics.

R for evaluation and orchestration. The diagnostic stack (cmdstanr, loo, bayesplot, SBC, posterior, priorsense) and the outer loop are both R. The bottleneck is MCMC, not orchestration.

## Tech stack

| Package | Role |
|---|---|
| [cmdstanr](https://mc-stan.org/cmdstanr/) | Stan interface |
| [SBC](https://hyunjimoon.github.io/SBC/) | Simulation-based calibration |
| [loo](https://mc-stan.org/loo/) | PSIS-LOO-CV, Pareto k diagnostics |
| [bayesplot](https://mc-stan.org/bayesplot/) | MCMC diagnostics, posterior predictive checks |
| [posterior](https://mc-stan.org/posterior/) | Draws format, convergence summaries |
| [priorsense](https://n-kall.github.io/priorsense/) | Power-scaling sensitivity analysis |
| [brms](https://paul-buerkner.github.io/brms/) | Baseline IRT models for benchmarking |
| [ellmer](https://ellmer.tidyverse.org/) | LLM API interface for R |

## Incremental strategy

1. Skill proposer. LLM reads item text, proposes a loading structure (and optionally a DAG). No student data, no inference.
2. Factor analysis. Search over loading matrix sparsity patterns, number of latent dimensions, and correlation structure. Multilevel IRT observation model.
3. Structural equation models. Search over both the loading matrix and the DAG among KCs. Causal discovery on the latent scale, scored by LOO.
4. Block configuration search. Open up remaining blocks (interaction terms, slip/guess, grouped populations, link function).
5. Free-form Stan. For structures that can't be expressed as block combinations. SBC becomes the primary safety net.

## Literature

### Model comparison and diagnostics

- Vehtari, Gelman, Gabry (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413-1432.
- Vehtari, Simpson, Gelman, Yao, Gabry (2024). Pareto smoothed importance sampling. *JMLR*, 25(72), 1-58.
- Talts, Betancourt, Simpson, Vehtari, Gelman (2018). Validating Bayesian inference algorithms with simulation-based calibration. arXiv:1804.06788.
- Kallioinen, Paananen, Burkner, Vehtari (2024). Detecting and diagnosing prior and likelihood sensitivity with power-scaling. *Statistics and Computing*, 34, 57.

### Knowledge structure discovery

- Fitzpatrick, Heusser, Manning (2026). Text embedding models yield detailed conceptual knowledge maps derived from short multiple-choice quizzes. [PsyArXiv](https://osf.io/preprints/psyarxiv/dh3q2). ([code](https://github.com/ContextLab/mapper)). Embeds items into a shared vector space, interpolates knowledge with a GP. A different approach: predictive and actionable, but not explanatory in terms of latent structure.

### LLM-driven model discovery

- AutoStan (2026). Autonomous Bayesian model improvement via predictive feedback. [arXiv:2603.27766](https://arxiv.org/abs/2603.27766). A CLI agent autonomously writes and improves Stan code. The closest existing work to autoskill's structure optimizer, but general-purpose and without SBC, composable blocks, or a semantic skill proposer.
- Lu et al. (2024). The AI Scientist: Towards fully automated open-ended scientific discovery. arXiv:2408.06292.

### Latent variable models and psychometrics

- Tatsuoka (1983). Rule space: An approach for dealing with misconceptions based on item response theory. *Journal of Educational Measurement*, 20(4), 345-354.
- Reckase (2009). *Multidimensional Item Response Theory*. Springer.
- de la Torre (2009). DINA model and parameter estimation: A didactic. *JEBS*, 34(1), 115-130.
- Lewandowski, Kurowicka, Joe (2009). Generating random correlation matrices based on vines and extended onion method. *Journal of Multivariate Analysis*, 100(9), 1989-2001.

### Probabilistic programming

- Carpenter et al. (2017). Stan: A probabilistic programming language. *Journal of Statistical Software*, 76(1).
- Betancourt (2017). A conceptual introduction to Hamiltonian Monte Carlo. arXiv:1701.02434.

## Data

Synthetic data for development. Real datasets TBD.
