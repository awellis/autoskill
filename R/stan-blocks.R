# -- Dispatchers ---------------------------------------------------------------

#' Generate measurement Stan block
#' @param config A `model_config` object.
#' @return A named list (Stan fragment).
#' @noRd
stan_measurement <- function(config) {
  switch(config$spec$measurement,
    linear      = stan_measurement_linear(config),
    interaction = stan_measurement_interaction(config)
  )
}

#' Generate structural Stan block
#' @param config A `model_config` object.
#' @return A named list (Stan fragment).
#' @noRd
stan_structural <- function(config) {
  switch(config$spec$structural,
    independent  = stan_structural_independent(config),
    correlated   = stan_structural_correlated(config),
    dag          = stan_structural_dag(config),
    hierarchical = stan_structural_hierarchical(config)
  )
}

#' Generate population Stan block
#' @param config A `model_config` object.
#' @return A named list (Stan fragment).
#' @noRd
stan_population <- function(config) {
  switch(config$spec$population,
    single  = stan_population_single(config),
    grouped = stan_population_grouped(config)
  )
}

#' Generate item Stan block
#' @param config A `model_config` object.
#' @return A named list (Stan fragment).
#' @noRd
stan_item <- function(config) {
  switch(config$spec$item,
    basic      = stan_item_basic(config),
    slip_guess = stan_item_slip_guess(config)
  )
}


# -- Measurement blocks --------------------------------------------------------

#' @noRd
stan_measurement_linear <- function(config) {
  link_inv <- if (config$spec$link == "logit") "inv_logit" else "Phi"
  frag <- empty_stan_fragment()

  frag$functions <- glue("
  real compute_prob(int i, row_vector theta_j, matrix Lambda, vector alpha) {{
    return {link_inv}(alpha[i] + Lambda[i] * theta_j');
  }}
")

  frag$data <- glue("
  int<lower=1> N_obs;
  int<lower=1> I;
  int<lower=1> J;
  int<lower=1> K;
  int<lower=1> N_loadings;
  array[N_loadings] int<lower=1,upper=I> loading_item;
  array[N_loadings] int<lower=1,upper=K> loading_skill;
  array[N_obs] int<lower=1,upper=I> ii;
  array[N_obs] int<lower=1,upper=J> jj;
  array[N_obs] int<lower=0,upper=1> y;
")

  frag$parameters <- glue("
  vector<lower=0>[N_loadings] lambda_free;
  vector[I] alpha;
")

  frag$transformed_parameters <- glue("
  matrix[I, K] Lambda = rep_matrix(0, I, K);
  for (n in 1:N_loadings)
    Lambda[loading_item[n], loading_skill[n]] = lambda_free[n];
")

  frag$model <- glue("
  // Measurement priors
  lambda_free ~ normal(0, 1);
  alpha ~ normal(0, 2);

  // Likelihood
  for (n in 1:N_obs)
    y[n] ~ bernoulli(compute_prob(ii[n], theta[jj[n]], Lambda, alpha));
")

  frag$generated_quantities <- glue("
  // Link: {link_inv}
  vector[N_obs] log_lik;
  for (n in 1:N_obs)
    log_lik[n] = bernoulli_lpmf(y[n] | compute_prob(ii[n], theta[jj[n]], Lambda, alpha));
")

  frag
}

#' @noRd
stan_measurement_interaction <- function(config) {
  frag <- stan_measurement_linear(config)

  frag$data <- paste0(frag$data, "\n", glue("
  int<lower=0> N_interactions;
  array[N_interactions] int<lower=1,upper=I> interact_item;
  array[N_interactions] int<lower=1,upper=K> interact_skill1;
  array[N_interactions] int<lower=1,upper=K> interact_skill2;
"))

  frag$parameters <- paste0(frag$parameters, "\n", glue("
  vector[N_interactions] gamma;
"))

  frag$model <- paste0(glue("
  // Interaction priors
  gamma ~ normal(0, 1);
"), "\n", frag$model)

  frag
}


# -- Structural blocks ---------------------------------------------------------

#' @noRd
stan_structural_independent <- function(config) {
  frag <- empty_stan_fragment()

  frag$parameters <- glue("
  matrix[J, K] theta;
")

  frag$model <- glue("
  // Structural: independent skills
  to_vector(theta) ~ std_normal();
")

  frag
}

#' @noRd
stan_structural_correlated <- function(config) {
  frag <- empty_stan_fragment()

  frag$parameters <- glue("
  cholesky_factor_corr[K] L_Omega;
  vector<lower=0>[K] sigma_theta;
  matrix[K, J] z_theta;
")

  frag$transformed_parameters <- glue("
  matrix[J, K] theta = (diag_pre_multiply(sigma_theta, L_Omega) * z_theta)';
")

  frag$model <- glue("
  // Structural: correlated skills (NCP)
  L_Omega ~ lkj_corr_cholesky(2.0);
  sigma_theta ~ normal(0, 1);
  to_vector(z_theta) ~ std_normal();
")

  frag
}

#' @noRd
stan_structural_dag <- function(config) {
  frag <- empty_stan_fragment()

  frag$data <- glue("
  int<lower=0> N_edges;
  array[N_edges] int<lower=1,upper=K> edge_from;
  array[N_edges] int<lower=1,upper=K> edge_to;
  array[K] int<lower=1,upper=K> topo_order;
")

  frag$parameters <- glue("
  vector[N_edges] B_free;
  matrix[J, K] eta;
")

  frag$transformed_parameters <- glue("
  matrix[K, K] B = rep_matrix(0, K, K);
  for (e in 1:N_edges)
    B[edge_from[e], edge_to[e]] = B_free[e];

  matrix[J, K] theta;
  for (j in 1:J) {{
    for (k_idx in 1:K) {{
      int k = topo_order[k_idx];
      theta[j, k] = eta[j, k] + B[, k]' * theta[j]';
    }}
  }}
")

  frag$model <- glue("
  // Structural: DAG
  B_free ~ normal(0, 1);
  to_vector(eta) ~ std_normal();
")

  frag
}

#' @noRd
stan_structural_hierarchical <- function(config) {
  frag <- empty_stan_fragment()

  frag$parameters <- glue("
  vector[J] phi;
  vector[K] beta_hier;
  matrix[J, K] eta;
")

  frag$transformed_parameters <- glue("
  matrix[J, K] theta;
  for (j in 1:J)
    for (k in 1:K)
      theta[j, k] = beta_hier[k] * phi[j] + eta[j, k];
")

  frag$model <- glue("
  // Structural: hierarchical
  phi ~ std_normal();
  beta_hier ~ normal(0, 1);
  to_vector(eta) ~ std_normal();
")

  frag
}


# -- Population blocks ---------------------------------------------------------

#' @noRd
stan_population_single <- function(config) {
  empty_stan_fragment()
}

#' @noRd
stan_population_grouped <- function(config) {
  frag <- empty_stan_fragment()

  frag$data <- glue("
  int<lower=1> N_groups;
  array[J] int<lower=1,upper=N_groups> group;
")

  frag$parameters <- glue("
  matrix[N_groups, K] mu_group;
  vector<lower=0>[K] sigma_group;
")

  frag$model <- glue("
  // Population: grouped
  to_vector(mu_group) ~ normal(0, 2);
  sigma_group ~ normal(0, 1);
")

  frag
}


# -- Item blocks ---------------------------------------------------------------

#' @noRd
stan_item_basic <- function(config) {
  empty_stan_fragment()
}

#' @noRd
stan_item_slip_guess <- function(config) {
  link_inv <- if (config$spec$link == "logit") "inv_logit" else "Phi"
  frag <- empty_stan_fragment()

  frag$functions <- glue("
  real compute_prob(int i, row_vector theta_j, matrix Lambda, vector alpha,
                    vector guess, vector slip) {{
    real base_prob = {link_inv}(alpha[i] + Lambda[i] * theta_j');
    return guess[i] + (1 - guess[i] - slip[i]) * base_prob;
  }}
")

  frag$parameters <- glue("
  vector<lower=0,upper=0.5>[I] guess;
  vector<lower=0,upper=0.5>[I] slip;
")

  frag$model <- glue("
  // Item: slip-guess priors
  guess ~ beta(1, 9);
  slip ~ beta(1, 9);
")

  frag
}
