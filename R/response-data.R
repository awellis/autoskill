#' Create a response data object
#' @param responses A matrix or tibble of student responses.
#' @return An S3 object of class `response_data`.
#' @export
response_data <- function(responses) {
  if (is.data.frame(responses)) {
    rd <- response_data_from_long(responses)
  } else if (is.matrix(responses)) {
    rd <- response_data_from_wide(responses)
  } else {
    cli_abort("{.arg responses} must be a matrix or data frame.")
  }
  validate_response_data(rd)
  rd
}

#' @noRd
response_data_from_wide <- function(Y) {
  if (is.null(rownames(Y))) rownames(Y) <- paste0("student_", seq_len(nrow(Y)))
  if (is.null(colnames(Y))) colnames(Y) <- paste0("item_", seq_len(ncol(Y)))
  storage.mode(Y) <- "integer"
  structure(list(Y = Y, item_ids = colnames(Y), student_ids = rownames(Y),
                 n_students = nrow(Y), n_items = ncol(Y)), class = "response_data")
}

#' @noRd
response_data_from_long <- function(df) {
  required <- c("student_id", "item_id", "correct")
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) cli_abort("Long format requires columns: {.val {missing}}.")
  wide <- df |>
    tidyr::pivot_wider(id_cols = "student_id", names_from = "item_id", values_from = "correct")
  student_ids <- wide$student_id
  Y <- as.matrix(wide[, -1])
  rownames(Y) <- student_ids
  storage.mode(Y) <- "integer"
  structure(list(Y = Y, item_ids = colnames(Y), student_ids = student_ids,
                 n_students = nrow(Y), n_items = ncol(Y)), class = "response_data")
}

#' @noRd
validate_response_data <- function(x) {
  vals <- x$Y[!is.na(x$Y)]
  if (!all(vals %in% c(0L, 1L))) cli_abort("Responses must be binary (0 or 1).")
  invisible(x)
}

#' @export
print.response_data <- function(x, ...) {
  n_missing <- sum(is.na(x$Y))
  cat(sprintf("<response_data>: %d students, %d items\n", x$n_students, x$n_items))
  if (n_missing > 0) cat(sprintf("  ! %d missing values\n", n_missing))
  invisible(x)
}
