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

## Architecture

The system separates creative work (model structure search) from analytical work (inference and evaluation).

**Creative step:** An LLM generates Stan models encoding structural hypotheses: the number of latent skill dimensions, the sparsity pattern of the loading matrix, and the form of the structural model (e.g., correlated vs. independent factors, hierarchical structure among skills).

**Evaluation pipeline (R):**

1. **SBC** — Simulation-based calibration checks whether the proposed model is identifiable before fitting real data. Non-identified models are rejected early, with diagnostics fed back to the LLM.
2. **Fit** — MCMC via cmdstanr (NUTS). Convergence diagnostics (R-hat, ESS, divergences, E-BFMI) determine whether inference succeeded.
3. **Compare** — LOO-CV via PSIS-LOO scores out-of-sample predictive performance with a natural complexity penalty. Pareto k diagnostics flag unreliable estimates.
4. **Reflect** — Structured feedback (poorly predicted items, pathological posteriors, parameter summaries, LOO differences) goes back to the LLM for the next iteration.

## Design decisions

**Continuous latent skills.** Skills are continuous latent factors, not binary mastered/not-mastered. This is more cognitively plausible (students develop graded proficiency), connects to the well-established IRT and factor analysis literature, and keeps the model class within Stan's strengths (no discrete latent variables to marginalize).

**Stan as the modeling language.** The LLM writes `.stan` files, not Python or R model code. Stan is language-agnostic, has gold-standard NUTS inference, and LLMs generate it well due to extensive training data. This cleanly separates model specification from the evaluation harness.

**R as the evaluation engine.** The analytical stack is written once in R and reused across all proposed models. R is where the methodological developers work (Vehtari, Gabry, Bürkner, Betancourt), and the tooling reflects that.

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

## Incremental strategy

1. **Phase 1 — Loading matrix sparsity search.** Fix the model framework (confirmatory factor model, Bernoulli/logit, continuous latent factors). The LLM's only degree of freedom is the sparsity pattern of the loading matrix and the number of latent dimensions. Inference is standard and well-understood.
2. **Phase 2 — Semantic initialization.** Use the LLM's understanding of item text and curriculum standards to generate informed initial loading structures, then optimize with the loop.
3. **Phase 3 — Full structural search.** Expand the search space to include the structural model among latent factors (correlated, hierarchical, higher-order), nonlinear interaction terms, and item-level structure beyond simple loadings.

## Data

Synthetic data for controlled experiments during development. Real datasets TBD.
