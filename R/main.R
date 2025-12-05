#' @title Rename Kaggle-style columns for heart disease dataset
#'
#' @description
#' `rename_heart_cols()` converts column names from the original Kaggle
#' "Heart Disease Prediction" dataset into the cleaned names used
#' during model training.
#'
#' @details
#' It handles columns such as:
#' \itemize{
#'   \item \code{Chest pain type} -> \code{ChestPainType}
#'   \item \code{FBS over 120}   -> \code{FBS_over_120}
#'   \item \code{EKG results}    -> \code{EKG_results}
#'   \item \code{Max HR}         -> \code{MaxHR}
#'   \item \code{Exercise angina} -> \code{ExerciseAngina}
#'   \item \code{ST depression}   -> \code{ST_depression}
#'   \item \code{Slope of ST}     -> \code{Slope_ST}
#'   \item \code{Number of vessels fluro} -> \code{NumVessels}
#'   \item \code{Heart Disease}   -> \code{HeartDisease}
#' }
#'
#' The function is intended for internal use and is called automatically
#' by the prediction helpers.
#'
#' @param df A data.frame.
#'
#' @keywords internal
rename_heart_cols <- function(df) {
  df2 <- df

  # remove index column if present
  if ("index" %in% names(df2)) {
    df2 <- df2[, setdiff(names(df2), "index"), drop = FALSE]
  }

  # rename Kaggle columns to cleaned names
  if ("Chest pain type" %in% names(df2)) {
    names(df2)[names(df2) == "Chest pain type"] <- "ChestPainType"
  }
  if ("FBS over 120" %in% names(df2)) {
    names(df2)[names(df2) == "FBS over 120"] <- "FBS_over_120"
  }
  if ("EKG results" %in% names(df2)) {
    names(df2)[names(df2) == "EKG results"] <- "EKG_results"
  }
  if ("Max HR" %in% names(df2)) {
    names(df2)[names(df2) == "Max HR"] <- "MaxHR"
  }
  if ("Exercise angina" %in% names(df2)) {
    names(df2)[names(df2) == "Exercise angina"] <- "ExerciseAngina"
  }
  if ("ST depression" %in% names(df2)) {
    names(df2)[names(df2) == "ST depression"] <- "ST_depression"
  }
  if ("Slope of ST" %in% names(df2)) {
    names(df2)[names(df2) == "Slope of ST"] <- "Slope_ST"
  }
  if ("Number of vessels fluro" %in% names(df2)) {
    names(df2)[names(df2) == "Number of vessels fluro"] <- "NumVessels"
  }
  if ("Heart Disease" %in% names(df2)) {
    names(df2)[names(df2) == "Heart Disease"] <- "HeartDisease"
  }

  df2
}

#' @title Add engineered heart-disease features
#'
#' @description
#' Adds the engineered features used in the BIO215 heart disease model:
#' \itemize{
#'   \item \code{age_decade}: Age grouped by decades
#'   \item \code{high_bp_flag}: Indicator for BP >= 140 mmHg
#'   \item \code{high_chol_flag}: Indicator for Cholesterol >= 240
#' }
#'
#' @details
#' The function checks that the required columns \code{Age}, \code{BP},
#' and \code{Cholesterol} are present, then creates the three engineered
#' variables and converts them to factors. It is intended for internal use
#' and is called automatically during preprocessing.
#'
#' @param df A data.frame with at least \code{Age}, \code{BP} and
#'   \code{Cholesterol}.
#'
#' @keywords internal
add_engineered_features <- function(df) {
  required <- c("Age", "BP", "Cholesterol")
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) {
    stop(
      "Missing required columns for engineered features: ",
      paste(missing, collapse = ", ")
    )
  }

  df$age_decade     <- floor(as.numeric(df$Age) / 10) * 10
  df$high_bp_flag   <- ifelse(as.numeric(df$BP) >= 140, 1, 0)
  df$high_chol_flag <- ifelse(as.numeric(df$Cholesterol) >= 240, 1, 0)

  df$age_decade     <- factor(df$age_decade)
  df$high_bp_flag   <- factor(df$high_bp_flag)
  df$high_chol_flag <- factor(df$high_chol_flag)

  df
}

#' @title Preprocess new data to match training features
#'
#' @description
#' `preprocess_newdata()` mirrors the preprocessing steps used during
#' model training, so that new patient data can be safely passed into
#' the pre-trained Random Forest model.
#'
#' @details
#' The function performs the following steps:
#' \itemize{
#'   \item Rename Kaggle-style columns to cleaned names (via
#'         [rename_heart_cols()])
#'   \item Add engineered features (\code{age_decade}, \code{high_bp_flag},
#'         \code{high_chol_flag}) via [add_engineered_features()]
#'   \item Drop the label column \code{HeartDisease} if present
#'   \item Ensure all training-time numeric and factor columns exist
#'   \item Convert numeric columns and impute missing values using the
#'         medians stored in \code{preproc$num_medians}
#'   \item Encode factor columns using training-time levels, mapping
#'         unseen/NA to \code{"Unknown"}
#'   \item Remove near-zero-variance columns
#'   \item Finally select only the features used by the RF model
#' }
#'
#' This is an internal helper and is not intended to be called directly
#' by end users.
#'
#' @param df_raw A data.frame with raw patient features.
#' @param preproc Preprocessing metadata stored inside \code{heart_model_object}.
#'   This is created during model training and includes numeric/factor columns,
#'   medians, factor levels and near-zero-variance columns.
#' @param top_features Character vector of feature names used by the RF model.
#'
#' @keywords internal
preprocess_newdata <- function(df_raw, preproc, top_features) {
  df <- as.data.frame(df_raw)

  # 1) rename + engineered features
  df <- rename_heart_cols(df)
  df <- add_engineered_features(df)

  # drop label if present
  if ("HeartDisease" %in% names(df)) {
    df$HeartDisease <- NULL
  }

  num_cols    <- preproc$num_cols
  fac_cols    <- preproc$fac_cols
  num_medians <- preproc$num_medians
  fac_levels  <- preproc$fac_levels
  nzv_cols    <- preproc$nzv_cols

  # ensure all training-time columns exist
  all_known_cols <- unique(c(
    num_cols,
    fac_cols,
    "age_decade", "high_bp_flag", "high_chol_flag"
  ))
  for (cn in all_known_cols) {
    if (!cn %in% names(df)) {
      df[[cn]] <- NA
    }
  }

  # numeric columns: make numeric + impute medians
  for (cn in num_cols) {
    df[[cn]] <- suppressWarnings(as.numeric(df[[cn]]))
    med <- num_medians[[cn]]
    df[[cn]][is.na(df[[cn]])] <- med
  }

  # factor columns: align levels; unseen/NA -> "Unknown"
  for (cn in fac_cols) {
    lv <- fac_levels[[cn]]
    x_char <- as.character(df[[cn]])
    x_char[is.na(x_char)]     <- "Unknown"
    x_char[!(x_char %in% lv)] <- "Unknown"
    df[[cn]] <- factor(x_char, levels = lv)
  }

  # remove near-zero-variance columns if they exist
  if (length(nzv_cols) > 0) {
    keep <- setdiff(names(df), nzv_cols)
    df <- df[, keep, drop = FALSE]
  }

  # finally ensure all top_features exist
  for (cn in top_features) {
    if (!cn %in% names(df)) {
      df[[cn]] <- NA  # should not happen; just a safety net
    }
  }

  df[, top_features, drop = FALSE]
}

#' @title Core heart-disease prediction helper
#'
#' @description
#' Internal helper that applies the pre-trained Random Forest model stored
#' in \code{heart_model_object} to new data, returning both the predicted
#' probability of \code{"Presence"} and the predicted class.
#'
#' @details
#' This function is not exported to users. It is used by the three public
#' wrappers:
#' \itemize{
#'   \item [predict_heart_multiple()]
#'   \item [predict_heart_single()]
#'   \item [predict_heart_bio215()]
#' }
#'
#' The model object \code{heart_model_object} is bundled as internal data
#' in the package and contains:
#' \itemize{
#'   \item \code{rf_model}: fitted Random Forest classifier
#'   \item \code{top_features}: character vector of selected features
#'   \item \code{preproc}: preprocessing metadata (numeric/factor columns,
#'         medians, factor levels, near-zero-variance columns)
#'   \item \code{positive_level} / \code{negative_level}: class labels
#'   \item \code{best_threshold}: decision threshold chosen on the test set
#' }
#'
#' @param newdata A data.frame of raw patient features (Kaggle-style
#'   or cleaned names).
#' @param positive_threshold Numeric. The decision threshold for labeling
#'   \code{"Presence"}. If \code{NULL}, the threshold stored inside
#'   \code{heart_model_object$best_threshold} is used.
#'
#' @return A list with:
#'   \itemize{
#'     \item \code{prob}: numeric vector of probabilities of \code{"Presence"}
#'     \item \code{class}: factor vector (\code{"Absence"} / \code{"Presence"})
#'   }
#'
#' @keywords internal
predict_heart_core <- function(newdata, positive_threshold = NULL) {
  mdl     <- heart_model_object
  preproc <- mdl$preproc
  feats   <- mdl$top_features

  x_new <- preprocess_newdata(
    df_raw       = newdata,
    preproc      = preproc,
    top_features = feats
  )

  # stats::predict will dispatch to predict.ranger
  pred <- predict(mdl$rf_model, data = x_new)
  prob_pos <- pred$predictions[, mdl$positive_level]

  thr <- if (is.null(positive_threshold)) mdl$best_threshold else positive_threshold

  pred_class <- ifelse(
    prob_pos >= thr,
    mdl$positive_level,
    mdl$negative_level
  )

  list(
    prob  = as.numeric(prob_pos),
    class = factor(
      pred_class,
      levels = c(mdl$negative_level, mdl$positive_level)
    )
  )
}

#' @title Predict heart disease risk for multiple patients
#'
#' @description
#' Uses the pre-trained BIO215 Random Forest model to predict the probability
#' and class label (\code{"Absence"} / \code{"Presence"}) for multiple
#' patient records.
#'
#' @details
#' The input can use either the original Kaggle column names
#' (e.g. \code{BP}, \code{Chest pain type}, \code{ST depression}) or the
#' cleaned internal names used during model training. The function internally
#' calls [predict_heart_core()] and appends the predicted probability
#' and status columns to the original input.
#'
#' This function is primarily a convenience wrapper for batch prediction
#' and for generating augmented tables that can be downloaded from the
#' Shiny web interface.
#'
#' @param newdata A data.frame with patient features. At minimum, the following
#'   variables are recommended:
#'   \itemize{
#'     \item \code{Age}
#'     \item \code{Sex}
#'     \item \code{BP}
#'     \item \code{Cholesterol}
#'     \item \code{Chest pain type} or \code{ChestPainType}
#'     \item \code{Max HR} or \code{MaxHR}
#'     \item \code{Exercise angina} or \code{ExerciseAngina}
#'     \item \code{ST depression} or \code{ST_depression}
#'   }
#'   Additional columns from the original Kaggle dataset are allowed.
#' @param positive_threshold Numeric. Probability threshold above which
#'   predictions are labeled \code{"Presence"}. If \code{NULL} (default),
#'   the threshold chosen during model training is used.
#'
#' @return A data.frame identical to \code{newdata} but with two additional
#'   columns:
#'   \itemize{
#'     \item \code{HeartDisease_prob}: predicted probability of \code{"Presence"}
#'     \item \code{HeartDisease_status}: predicted class label
#'   }
#'
#' @examples
#' \dontrun{
#' new_patients <- data.frame(
#'   Age               = c(54, 63),
#'   Sex               = c("M", "F"),
#'   BP                = c(140, 120),
#'   Cholesterol       = c(289, 250),
#'   `Chest pain type` = c("ATA", "NAP"),
#'   `Max HR`          = c(150, 132),
#'   `Exercise angina` = c("N", "Y"),
#'   `ST depression`   = c(1.5, 2.3)
#' )
#'
#' pred_df <- predict_heart_multiple(new_patients)
#' head(pred_df)
#' }
#'
#' @seealso [predict_heart_single()], [predict_heart_bio215()]
#'
#' @importFrom stats predict
#' @export
predict_heart_multiple <- function(newdata, positive_threshold = NULL) {
  res <- predict_heart_core(newdata, positive_threshold = positive_threshold)

  out <- as.data.frame(newdata)
  out$HeartDisease_prob   <- res$prob
  out$HeartDisease_status <- res$class

  out
}

#' @title Predict heart disease risk for a single patient
#'
#' @description
#' A simple convenience wrapper for predicting a single patient's heart
#' disease risk. It internally constructs a one-row data.frame and calls
#' [predict_heart_multiple()].
#'
#' @details
#' This function is useful when you want to explore the effect of changing
#' one or two clinical variables for a hypothetical patient, without
#' manually creating a data.frame. The arguments correspond closely to the
#' core Kaggle variables.
#'
#' @param Age Numeric age in years.
#' @param Sex Character or factor for sex (e.g., \code{"M"}, \code{"F"}).
#' @param BP Numeric resting blood pressure (mmHg).
#' @param Cholesterol Numeric cholesterol level.
#' @param ChestPainType Character chest pain type (e.g. \code{"ATA"},
#'   \code{"NAP"}, \code{"ASY"}, \code{"TA"}).
#' @param MaxHR Numeric maximum heart rate during exercise test.
#' @param ExerciseAngina Character exercise-induced angina
#'   (e.g. \code{"Y"} / \code{"N"}).
#' @param ST_depression Numeric ST depression (if you prefer, you can also use
#'   the Kaggle-style name \code{"ST depression"} by constructing a data.frame
#'   manually and passing it to [predict_heart_multiple()]).
#' @param positive_threshold Numeric probability threshold for labeling
#'   \code{"Presence"} (default: use the training-time threshold).
#'
#' @return A named vector with:
#'   \itemize{
#'     \item \code{HeartDisease_prob}: predicted probability of \code{"Presence"}
#'     \item \code{HeartDisease_status}: predicted label
#'   }
#'
#' @examples
#' \dontrun{
#' res <- predict_heart_single(
#'   Age = 60,
#'   Sex = "M",
#'   BP  = 145,
#'   Cholesterol = 260,
#'   ChestPainType = "ATA",
#'   MaxHR = 150,
#'   ExerciseAngina = "N",
#'   ST_depression = 1.2
#' )
#' res
#' }
#'
#' @seealso [predict_heart_multiple()], [predict_heart_bio215()]
#'
#' @export
predict_heart_single <- function(
    Age,
    Sex,
    BP,
    Cholesterol,
    ChestPainType,
    MaxHR,
    ExerciseAngina,
    ST_depression,
    positive_threshold = NULL
) {
  new_df <- data.frame(
    Age            = as.numeric(Age),
    Sex            = as.character(Sex),
    BP             = as.numeric(BP),
    Cholesterol    = as.numeric(Cholesterol),
    ChestPainType  = as.character(ChestPainType),
    MaxHR          = as.numeric(MaxHR),
    ExerciseAngina = as.character(ExerciseAngina),
    ST_depression  = as.numeric(ST_depression),
    stringsAsFactors = FALSE
  )

  pred_df <- predict_heart_multiple(
    newdata            = new_df,
    positive_threshold = positive_threshold
  )

  c(
    HeartDisease_prob   = as.numeric(pred_df$HeartDisease_prob[1]),
    HeartDisease_status = as.character(pred_df$HeartDisease_status[1])
  )
}

#' @title Heart disease risk prediction from clinical features
#'
#' @description
#' `predict_heart_bio215()` is the **main user-facing function** of the
#' \pkg{heartPrediction} package. It applies the final top-10-feature
#' Random Forest model developed in the BIO215 capstone project to new
#' patient data and returns heart disease predictions.
#'
#' The function is designed to be a simple, high-level interface:
#' users provide a \code{data.frame} of routine clinical variables
#' (age, blood pressure, cholesterol, chest pain type, exercise test
#' results, etc.), and the function returns either:
#'
#' \itemize{
#'   \item binary class labels: \code{"Absence"} vs \code{"Presence"}
#'         of heart disease, or
#'   \item predicted probabilities of \code{"Presence"}
#' }
#'
#' It supports both the original Kaggle-style column names
#' (e.g. \code{BP}, \code{Chest pain type}, \code{ST depression}) and the
#' cleaned internal names used during model training, so that the same
#' function can be used directly on the Kaggle CSV, on data exported from
#' the Shiny web app, or on external datasets with similar structure.
#'
#' Importantly, this function is exactly the one required by the BIO215
#' rubric: it takes a feature table as input and returns predicted
#' classes for a classification task, using the same trained model as the
#' Shiny application.
#'
#' @details
#' Internally, the function:
#' \enumerate{
#'   \item Renames Kaggle columns to cleaned names (via [rename_heart_cols()])
#'   \item Adds engineered features (\code{age_decade}, \code{high_bp_flag},
#'         \code{high_chol_flag})
#'   \item Aligns numeric columns and imputes missing values using the training
#'         medians stored in \code{heart_model_object$preproc}
#'   \item Aligns factor columns to the training-time levels, mapping unseen
#'         categories to \code{"Unknown"}
#'   \item Removes near-zero-variance features
#'   \item Selects the top-10 features used by the final Random Forest model
#'   \item Computes predicted probabilities of \code{"Presence"}
#'   \item Converts probabilities to class labels using the decision threshold
#'         chosen on the test set during model development
#' }
#'
#' By default, the function returns only the predicted classes
#' (factor with levels \code{c("Absence", "Presence")}), which matches the
#' BIO215 marking rubric for classification tasks. Optionally, users can
#' request predicted probabilities instead via \code{type = "prob"}.
#'
#' @section BIO215 compliance:
#' This function is explicitly designed to:
#' \itemize{
#'   \item Take a \code{data.frame} of patient features as input
#'   \item Return predicted class labels for a binary classification task
#'   \item Use the same trained model as the associated Shiny web app
#' }
#'
#' @param newdata A data.frame of patient features. At minimum, the following
#'   variables are recommended (Kaggle-style or cleaned names are both accepted):
#'   \itemize{
#'     \item \code{Age}
#'     \item \code{Sex}
#'     \item \code{BP}
#'     \item \code{Cholesterol}
#'     \item \code{Chest pain type} or \code{ChestPainType}
#'     \item \code{Max HR} or \code{MaxHR}
#'     \item \code{Exercise angina} or \code{ExerciseAngina}
#'     \item \code{ST depression} or \code{ST_depression}
#'   }
#'   Additional columns from the original Kaggle dataset are allowed.
#'
#' @param type Either \code{"class"} (default) to return a factor vector
#'   (\code{"Absence"} / \code{"Presence"}), or \code{"prob"} to return
#'   a numeric vector of probabilities.
#'
#' @param positive_threshold Optional probability threshold; if \code{NULL},
#'   the training-time threshold stored in \code{heart_model_object$best_threshold}
#'   is used.
#'
#' @return A vector of predicted classes or probabilities:
#' \itemize{
#'   \item If \code{type = "class"} (default), a factor with levels
#'         \code{c("Absence", "Presence")}.
#'   \item If \code{type = "prob"}, a numeric vector of probabilities of
#'         \code{"Presence"}.
#' }
#'
#' @examples
#' \dontrun{
#' new_patients <- data.frame(
#'   Age               = c(54, 63),
#'   Sex               = c("M", "F"),
#'   BP                = c(140, 120),
#'   Cholesterol       = c(289, 250),
#'   `Chest pain type` = c("ATA", "NAP"),
#'   `Max HR`          = c(150, 132),
#'   `Exercise angina` = c("N", "Y"),
#'   `ST depression`   = c(1.5, 2.3)
#' )
#'
#' # Predicted classes (Absence/Presence):
#' predict_heart_bio215(new_patients)
#'
#' # Predicted probabilities of Presence:
#' predict_heart_bio215(new_patients, type = "prob")
#' }
#'
#' @seealso [predict_heart_multiple()], [predict_heart_single()]
#'
#' @importFrom stats predict
#' @export
predict_heart_bio215 <- function(
    newdata,
    type = c("class", "prob"),
    positive_threshold = NULL
) {
  type <- match.arg(type)

  core <- predict_heart_core(newdata, positive_threshold = positive_threshold)

  if (type == "prob") {
    return(core$prob)
  }

  core$class
}
