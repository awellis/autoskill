# Related work

Living document. Papers and packages directly relevant to autoskill's design decisions, with notes on what to borrow and what makes autoskill distinct.

## Bayesian causal discovery

### Castelletti et al. (2024) — Bayesian Causal Discovery for Policy Decision-Making

Cambridge *Data and Policy*. https://www.cambridge.org/core/journals/data-and-policy/article/A870EA8ED170647643FB590463E154DD

**Setup.** Bayesian DAG structure learning applied to education policy: Australian school-attendance data (12 binary variables, ~21k observations, 2004 to 2016), with synthetic five-variable validation.

**Method.**

- Inference: Partition MCMC (Kuipers and Moffa, 2017) over DAGs. ~300k iterations on synthetic, ~100k on real data. Implemented via the `BiDAG` R package.
- Scoring: BGe for Gaussian synthetic data, BDe for discrete real data. Both decomposable: `score(G) = sum_v score(v | parents(v))`. Decomposability is what makes 100k iterations tractable: a single-edge proposal only changes one node's local score.
- Output: a posterior over DAG structures, not a MAP. Convergence diagnosed via Jensen-Shannon divergence between true and estimated posteriors on synthetic data.

**Key empirical claim.** The DAG posterior is genuinely multimodal. For five nodes, ~20 structures have comparable posterior probability. For their 12-variable application, the top five structures contain 80% of the posterior mass. This is the same phenomenon that motivates autoskill's stacking output.

**Domain knowledge.** Acknowledged in principle ("Bayesian structure learning allows for prior knowledge from experts or communities") but not actually operationalized. Their experts informed *variable construction*, not the structure prior `P(G)`. This is the gap autoskill's `edge_prior` API addresses directly.

**What to borrow.**

1. Partition MCMC as a third loop algorithm alongside greedy and SMC. PMCMC samples over node orderings rather than edges, which mixes much better across Markov-equivalent structures. For the small node counts typical of skill DAGs, this matters. Slot it into the `structure_problem` interface as `optimize_structure_pmcmc(problem, ...)`.
2. Decomposable scoring with caching. The autoskill IRT score (LOO over the joint observation model) is not decomposable per-node, but the structural part of an SEM is. Cache the per-node structural log-likelihood; recompute only affected pieces on local moves.
3. ESS over a structure summary statistic for convergence diagnostics. Track marginal edge-inclusion probabilities across the chain; declare convergence when minimum ESS crosses a threshold. Adapts the spirit of their JSD diagnostic to the no-ground-truth case.

**What autoskill does that they do not.**

- LLM-driven proposals. Their MCMC explores the DAG space uniformly modulo decomposable scoring. Autoskill uses the LLM as both prior (`edge_prior`) and proposal kernel.
- Latent-variable extension. They handle fully-observed Gaussian or discrete data. Skill discovery is fundamentally about *latent* KC structure, requiring the IRT layer. The Bayesian causal discovery literature has very little on latent-variable DAGs; this is a real gap.
- Joint measurement and structural search. They search the structural DAG only. Autoskill searches over both `Lambda` (loading matrix sparsity) and the DAG among latent factors. The joint search is what makes the skill problem distinctive and harder.
- Stacking for prediction. They report posterior over structures but do not ensemble for prediction. Autoskill's `compute_stacked_weights()` does. For decision-theoretic use this matters: their "five top structures" should be averaged for any actual policy decision, not just listed.

**Validation opportunity.** Their school-attendance analysis is an ideal second-application case for autoskill's `causal_problem` constructor. Replicating it with LLM-elicited edge priors would be a natural validation paper, demonstrating that the LLM prior recovers (or improves on) the experts' implicit knowledge.

### Kuipers and Moffa (2017) — Partition MCMC for structure learning

JMLR 2017. The technical foundation Castelletti et al. build on. Implementation in the `BiDAG` R package (Kuipers, Moffa, Suter). Worth reading before any PMCMC implementation work in autoskill.

Key idea: instead of MCMC over DAGs (high autocorrelation, bad mixing across equivalence classes), MCMC over *partitions* of the node set, where each partition class is a "level" in a topological ordering. Each partition corresponds to a subset of DAGs, and the score factors cleanly.

## Latent variable models and psychometrics

(See main `README.md` references section for foundational psychometric literature: Tatsuoka 1983, Reckase 2009, de la Torre 2009, LKJ 2009.)

## LLM-driven model discovery

### AutoStan (2026)

arXiv:2603.27766. CLI agent that autonomously writes and improves Stan code. Closest existing work to autoskill's structure optimizer. Differences:

- General-purpose; no domain abstraction (no `structure_problem` interface).
- No SBC validation step.
- No composable-blocks layer; the LLM writes Stan source directly, which is more flexible but loses the "every combination compiles" guarantee.
- No semantic skill proposer.
- No structural prior (no analog to `edge_prior`).
- Greedy ELPD argmax; no stacking, no SMC.

Autoskill is essentially AutoStan + structure prior + composable blocks + stacking + (planned) SMC, specialized to latent-variable models initially but generalizable via the `structure_problem` abstraction.

## Reading queue

- Kuipers and Moffa (2017), JMLR. Partition MCMC paper.
- BiDAG R package source. Implementation reference for PMCMC + BGe/BDe scoring with caching.
- Eaton and Murphy (2007), "Exact Bayesian structure learning from order-independent score functions." Background on order-MCMC for DAG search.
- Friedman and Koller (2003), "Being Bayesian about network structure." Earlier order-based sampling, relevant background for SMC over DAGs.
