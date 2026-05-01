#' Construct a causal DAG over named variables
#'
#' A `causal_dag` is a directed graph represented as a binary K x K
#' adjacency matrix with `adj[i, j] = 1` meaning "i is a parent of j."
#' Diagonal entries are forced to zero on construction.
#'
#' @param adj A K by K binary matrix.
#' @param variables Length-K character vector of variable names. Defaults
#'   to `colnames(adj)` if available.
#' @return An S3 object of class `causal_dag`.
#' @seealso [empty_dag()], [is_acyclic()]
#' @export
causal_dag <- function(adj, variables = colnames(adj)) {
  if (!is.matrix(adj)) cli_abort("{.arg adj} must be a matrix.")
  K <- nrow(adj)
  if (ncol(adj) != K) cli_abort("{.arg adj} must be square.")
  if (length(variables) != K) {
    cli_abort("{.arg variables} must have length {K} (matrix dimension).")
  }
  if (!all(adj %in% c(0, 1))) {
    cli_abort("{.arg adj} entries must be 0 or 1.")
  }
  storage.mode(adj) <- "integer"
  diag(adj) <- 0L
  rownames(adj) <- colnames(adj) <- variables

  base::structure(
    list(adj = adj, variables = variables),
    class = "causal_dag"
  )
}

#' Construct an empty DAG over named variables
#' @param variables Character vector of variable names.
#' @return A `causal_dag` with no edges.
#' @export
empty_dag <- function(variables) {
  K <- length(variables)
  causal_dag(matrix(0L, K, K), variables)
}

#' @export
print.causal_dag <- function(x, ...) {
  edges <- which(x$adj == 1L, arr.ind = TRUE)
  cat(sprintf("<causal_dag>: %d nodes, %d edges\n",
              length(x$variables), nrow(edges)))
  if (nrow(edges) > 0L && nrow(edges) <= 20L) {
    for (i in seq_len(nrow(edges))) {
      cat(sprintf("  %s -> %s\n",
                  x$variables[edges[i, 1L]], x$variables[edges[i, 2L]]))
    }
  } else if (nrow(edges) > 20L) {
    cat(sprintf("  (... %d edges, suppressed)\n", nrow(edges)))
  }
  invisible(x)
}

#' Test whether a `causal_dag` is acyclic
#'
#' Implements Kahn's algorithm: repeatedly removes source nodes
#' (in-degree zero); the graph is acyclic iff every node is removed.
#'
#' @param dag A `causal_dag` (or just its adjacency matrix).
#' @return Logical scalar.
#' @export
is_acyclic <- function(dag) {
  adj <- if (inherits(dag, "causal_dag")) dag$adj else dag
  K <- nrow(adj)
  if (K == 0L) return(TRUE)

  in_degree <- colSums(adj)
  removed <- logical(K)

  while (TRUE) {
    sources <- which(!removed & in_degree == 0L)
    if (length(sources) == 0L) break
    src <- sources[1L]
    removed[src] <- TRUE
    in_degree <- in_degree - adj[src, ]
  }

  all(removed)
}

#' Score a DAG against data via per-node Gaussian regression with BIC
#'
#' Decomposable scoring: each node is regressed on its parents using
#' maximum-likelihood Gaussian linear regression; per-node BIC sums to
#' the total. Returns a `fit_result`-shaped object so the standard
#' optimisation loop, caching, and stacking machinery work unchanged.
#'
#' Not a true Bayesian marginal likelihood (BGe) — that's a future
#' upgrade. BIC is principled, decomposable, and small enough to keep
#' the second-application demo tight.
#'
#' @param data A data frame with one column per variable in `dag`.
#' @param dag A `causal_dag`.
#' @return A `fit_result` with `$loo$estimates["elpd_loo", "Estimate"]`
#'   set to the BIC-adjusted log-likelihood.
#' @export
score_causal_dag <- function(data, dag) {
  vars <- dag$variables
  adj <- dag$adj
  n <- nrow(data)

  if (!all(vars %in% names(data))) {
    missing <- setdiff(vars, names(data))
    cli_abort("Variables missing from {.arg data}: {.val {missing}}.")
  }

  log_liks <- vector("list", length(vars))
  total_params <- 0L

  for (j in seq_along(vars)) {
    parents <- vars[adj[, j] == 1L]
    y <- data[[vars[j]]]

    if (length(parents) == 0L) {
      mu_hat <- rep(mean(y), n)
      n_params_node <- 2L  # mean, residual variance
    } else {
      X <- cbind(1, as.matrix(data[, parents, drop = FALSE]))
      coefs <- qr.coef(qr(X), y)
      mu_hat <- as.numeric(X %*% coefs)
      n_params_node <- length(parents) + 2L  # intercept, slopes, variance
    }

    sigma2 <- sum((y - mu_hat)^2) / n
    sigma_hat <- max(sqrt(sigma2), 1e-10)
    log_liks[[j]] <- stats::dnorm(y, mu_hat, sigma_hat, log = TRUE)
    total_params <- total_params + n_params_node
  }

  pointwise <- Reduce(`+`, log_liks)
  total_loglik <- sum(pointwise)
  bic_penalty <- 0.5 * total_params * log(n)
  elpd <- total_loglik - bic_penalty

  loo_obj <- base::structure(
    list(
      estimates = matrix(
        c(elpd, NA_real_), nrow = 1L,
        dimnames = list("elpd_loo", c("Estimate", "SE"))
      ),
      pointwise = matrix(
        pointwise - bic_penalty / n, ncol = 1L,
        dimnames = list(NULL, "elpd_loo")
      )
    ),
    class = "loo"
  )

  base::structure(
    list(
      fit = NULL, config = dag, loo = loo_obj,
      diagnostics = tibble::tibble(
        metric = c("log_lik", "n_params", "bic"),
        value  = c(total_loglik, total_params,
                   -2 * total_loglik + total_params * log(n)),
        status = "ok"
      ),
      param_summary = tibble::tibble()
    ),
    class = "fit_result"
  )
}

#' Propose a single random local move on a DAG
#'
#' Picks one of `add_edge`, `remove_edge`, `reverse_edge` at random and
#' applies it, retrying up to `max_attempts` times if the result would
#' create a cycle. Used as the placeholder mutation kernel for
#' `causal_problem()` until an LLM-driven proposer lands.
#'
#' @param dag A `causal_dag`.
#' @param max_attempts Maximum proposals to try before giving up.
#' @return A new `causal_dag`. Returns the input unchanged if no valid
#'   move was found.
#' @export
random_local_move <- function(dag, max_attempts = 50L) {
  vars <- dag$variables
  K <- length(vars)

  for (attempt in seq_len(max_attempts)) {
    move_type <- sample(c("add", "remove", "reverse"), 1L)
    new_adj <- dag$adj

    edges <- which(new_adj == 1L, arr.ind = TRUE)
    non_edges <- which(new_adj == 0L & row(new_adj) != col(new_adj),
                       arr.ind = TRUE)

    if (move_type == "add") {
      if (nrow(non_edges) == 0L) next
      pick <- non_edges[sample(nrow(non_edges), 1L), ]
      new_adj[pick[1L], pick[2L]] <- 1L
    } else if (move_type == "remove") {
      if (nrow(edges) == 0L) next
      pick <- edges[sample(nrow(edges), 1L), ]
      new_adj[pick[1L], pick[2L]] <- 0L
    } else {
      if (nrow(edges) == 0L) next
      pick <- edges[sample(nrow(edges), 1L), ]
      new_adj[pick[1L], pick[2L]] <- 0L
      new_adj[pick[2L], pick[1L]] <- 1L
    }

    if (is_acyclic(new_adj)) return(causal_dag(new_adj, vars))
  }

  dag
}

#' Construct a structure_problem for causal DAG discovery
#'
#' A second application of the [structure_problem()] interface, built to
#' validate the abstraction. Uses adjacency-matrix DAGs over named
#' variables, scores via per-node Gaussian regression with BIC, and
#' proposes refinements via random local moves.
#'
#' Intentionally minimal: no LLM proposer yet (the random kernel is
#' enough to validate that the loop, caching, and stacking machinery work
#' for non-skill applications). An LLM-driven proposer is a natural
#' follow-up that slots into the same `propose_refinement` slot.
#'
#' @param data A data frame of continuous observations. Column names are
#'   the variable names.
#' @param variables Optional subset of `names(data)` to include. Defaults
#'   to all numeric columns.
#' @return A `structure_problem` of subclass `"causal_problem"`.
#' @seealso [structure_problem()], [optimize_structure()],
#'   [score_causal_dag()]
#' @export
causal_problem <- function(data, variables = NULL) {
  if (is.null(variables)) {
    variables <- names(data)[vapply(data, is.numeric, logical(1L))]
  }
  if (length(variables) < 2L) {
    cli_abort("Need at least 2 numeric variables; got {length(variables)}.")
  }
  if (!all(variables %in% names(data))) {
    missing <- setdiff(variables, names(data))
    cli_abort("Variables missing from {.arg data}: {.val {missing}}.")
  }

  data_hash <- rlang::hash(data[, variables, drop = FALSE])

  structure_problem(
    data = list(observations = data, variables = variables),

    propose_initial = function(...) empty_dag(variables),

    propose_refinement = function(current, fit_result, history,
                                   n_candidates = 10L) {
      # Mini hill-climbing: sample several local moves and return the
      # best improvement, or `current` if none improve. This is the
      # placeholder mutation kernel; an LLM-driven proposer slots into
      # the same slot in a follow-up.
      current_elpd <- fit_result$loo$estimates["elpd_loo", "Estimate"]
      best <- current
      best_elpd <- current_elpd

      for (i in seq_len(n_candidates)) {
        candidate <- random_local_move(current)
        if (identical(candidate$adj, current$adj)) next
        cand_elpd <- score_causal_dag(data, candidate)$loo$estimates[
          "elpd_loo", "Estimate"
        ]
        if (cand_elpd > best_elpd) {
          best <- candidate
          best_elpd <- cand_elpd
        }
      }

      best
    },

    score = function(structure, cache = NULL, ...) {
      if (is.null(cache)) {
        return(score_causal_dag(data, structure))
      }
      key <- causal_cache_key(structure, list(...), data_hash)
      hit <- cache$get(key)
      if (!cachem::is.key_missing(hit)) return(hit)
      result <- score_causal_dag(data, structure)
      cache$set(key, result)
      result
    },

    log_prior = function(structure) 0,

    validate = function(structure) {
      if (!is_acyclic(structure)) {
        return(list(passed = FALSE,
                    problems = "DAG contains a cycle"))
      }
      list(passed = TRUE, problems = character(0))
    },

    cache_key = function(structure, fit_args) {
      causal_cache_key(structure, fit_args, data_hash)
    },

    summarize_structure = function(structure) {
      sprintf("DAG[%d nodes, %d edges]",
              length(structure$variables), sum(structure$adj))
    },

    summarize_fit = function(fit_result) {
      list(
        elpd = fit_result$loo$estimates["elpd_loo", "Estimate"],
        diagnostics = fit_result$diagnostics
      )
    },

    class = "causal_problem"
  )
}

#' @noRd
causal_cache_key <- function(dag, fit_args, data_hash) {
  rlang::hash(list(
    adj = dag$adj,
    variables = dag$variables,
    data = data_hash,
    fit_args = fit_args
  ))
}
