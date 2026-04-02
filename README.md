# autoskill

LLM-driven discovery of latent knowledge component structure in student response data.

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

`B` encodes structural coefficients (nonzero where the DAG has an edge). The DAG encodes hypotheses like "algebraic manipulation depends on number sense." Factor analysis mode sets `B = 0` and estimates a free correlation matrix instead.

### Temporal structure

Ignored initially. Responses are treated as a batch, exchangeable given the student's KC vector. Learning dynamics are a future extension.

### Example

Six items on a math test. The LLM hypothesizes two KCs: algebra and word problem comprehension.

| Item | Algebra | Word problems |
|---|---|---|
| Solve 3x + 5 = 20 | lambda_11 | 0 |
| Simplify 2(x + 3) - x | lambda_21 | 0 |
| "You have $11, apples cost $2. How many can you buy?" | 0 | lambda_32 |
| "A train goes x km/h for 3h, covering 180 km. Find x." | lambda_41 | lambda_42 |
| Factor x^2 - 9 | lambda_51 | 0 |
| "Two numbers sum to 20, differ by 4. Find them." | lambda_61 | lambda_62 |

The zeros are the hypothesis. Items 1, 2, 5 require only algebra. Item 3 requires only word problem comprehension. Items 4 and 6 require both.

Two students:

- Student A (theta = [1.5, -1.0]): strong algebra, weak word problems. Gets items 1, 2, 5 right easily. Struggles with item 3.
- Student B (theta = [-0.5, 1.5]): weak algebra, strong word problems. Opposite pattern.

If this structure is correct, these students produce systematically different response patterns that a single-factor model cannot explain. The LLM's job is to discover that this two-factor structure fits the data better than alternatives.

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
