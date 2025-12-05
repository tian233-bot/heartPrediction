# heartPrediction: Heart Disease Risk Prediction from Clinical Features

`heartPrediction` is an R package developed as part of the BIO215 capstone project on heart disease prediction.

The package bundles the final **top-10-feature Random Forest model** trained on the Kaggle *“Heart Disease Prediction”* dataset and provides a simple, high-level interface for making predictions from routine clinical variables:

- Age  
- Sex  
- Resting blood pressure (BP)  
- Cholesterol  
- Chest pain type  
- Exercise-test variables (Max HR, ST depression, etc.)

The same trained model is also used in the associated Shiny web application.

---

## Package goals

The main goals of this package are:

- To provide a **reproducible implementation** of the BIO215 machine learning model in R.
- To offer a simple function that:
  - takes a `data.frame` of patient features as input; and  
  - returns **predicted heart disease classes** (Absence / Presence) or **probabilities**.
- To expose a clean programmatic interface that is consistent with the Shiny web app used for database and model deployment.

This design directly matches the R package requirements in the BIO215 assignment rubric.

---

## Installation

You can install the package directly from GitHub using either **`remotes`** or **`devtools`**.

> Replace `your_github_username` with your actual GitHub username once the repository is online.

### 1. Install prerequisites

```r
install.packages("remotes")   # or: install.packages("devtools")
install.packages("ranger")    # backend for the Random Forest model
```
### 2. Install heartPrediction from GitHub

Using remotes:
```r
remotes::install_github("your_github_username/heartPrediction")
```
### or, using devtools:
```r 
devtools::install_github("your_github_username/heartPrediction")
```
### Then load the package:
```r
library(heartPrediction)
```
---
### Quick start: minimal working examples

Example 1 – BIO215 interface (predict_heart_bio215())

The following example creates a small table of two hypothetical patients and
obtains both predicted classes and probabilities.
```r
library(heartPrediction)

new_patients <- data.frame(
  Age               = c(54, 63),
  Sex               = c("M", "F"),
  BP                = c(140, 120),
  Cholesterol       = c(289, 250),
  `Chest pain type` = c("ATA", "NAP"),
  `Max HR`          = c(150, 132),
  `Exercise angina` = c("N", "Y"),
  `ST depression`   = c(1.5, 2.3)
)

# 1) Predicted classes (Absence / Presence)
pred_class <- predict_heart_bio215(new_patients)
pred_class

# 2) Predicted probabilities for Presence
pred_prob <- predict_heart_bio215(new_patients, type = "prob")
pred_prob
```
Example 2 – Batch prediction with augmented table

predict_heart_multiple() returns the original columns plus two additional
columns: HeartDisease_prob and HeartDisease_status.
```r
library(heartPrediction)

pred_df <- predict_heart_multiple(new_patients)
pred_df
```
Example 3 – Single patient prediction

predict_heart_single() is a convenience wrapper for a single patient. It
accepts individual clinical values instead of a data frame.
```r
library(heartPrediction)

res_single <- predict_heart_single(
  Age            = 60,
  Sex            = "M",
  BP             = 145,
  Cholesterol    = 260,
  ChestPainType  = "ATA",
  MaxHR          = 150,
  ExerciseAngina = "N",
  ST_depression  = 1.2
)

res_single
```
---
## Model performance: ROC and Precision–Recall curves

To demonstrate the performance of the model, you can evaluate it on the
Kaggle “Heart Disease Prediction” dataset and generate ROC and
Precision–Recall (PR) curves.

The code below is suitable for an R script or vignette and can be used to
produce figures for a project report.

### 1. Load data and required packages
```r
library(heartPrediction)
library(readr)
library(pROC)
library(PRROC)
library(ggplot2)
library(dplyr)

# Adjust the path to the location of the Kaggle CSV
heart <- read_csv("Heart_Disease_Prediction.csv", show_col_types = FALSE)
```
Assume the label column is Heart Disease with 0/1 coding, as in the original
Kaggle file. Convert it to a numeric 0/1 vector:
```r
# 1 = Presence, 0 = Absence
if (is.numeric(heart[["Heart Disease"]])) {
  y_true <- as.integer(heart[["Heart Disease"]] == 1)
} else {
  lab_char <- as.character(heart[["Heart Disease"]])
  y_true <- ifelse(lab_char %in% c("1", "Presence", "Yes"), 1, 0)
}
```
Obtain model predictions for all rows:
```r
# Predicted probability of Presence for each row
prob_pos <- predict_heart_bio215(heart, type = "prob")
```
### 2. ROC curve and AUROC
```r
roc_obj <- pROC::roc(response = y_true, predictor = prob_pos)
auroc   <- as.numeric(pROC::auc(roc_obj))
auroc
```
Base R plot:
```r
plot(roc_obj, col = "#1f77b4", lwd = 2,
     main = sprintf("ROC curve (AUROC = %.3f)", auroc))
abline(a = 0, b = 1, lty = 2, col = "grey70")
```
Alternatively, a ggplot-style ROC curve (using ggroc from pROC):
```r
ggroc(roc_obj, size = 1.1) +
  geom_abline(slope = 1, intercept = 0,
              linetype = 2, colour = "grey70") +
  coord_equal() +
  labs(
    title = sprintf("ROC curve – Top-10 feature Random Forest (AUROC = %.3f)", auroc),
    x = "False positive rate (1 – specificity)",
    y = "True positive rate (sensitivity)"
  ) +
  theme_minimal(base_size = 14)

ggsave("fig_roc_top10_rf.png", width = 6, height = 5, dpi = 300)
```
### 3. Precision–Recall curve and AUPRC
```r
pr_obj <- PRROC::pr.curve(
  scores.class0 = prob_pos[y_true == 1],  # scores for positive class
  scores.class1 = prob_pos[y_true == 0],  # scores for negative class
  curve = TRUE
)
auprc <- pr_obj$auc.integral
auprc
```
Base R plot:
```r
plot(pr_obj,
     main = sprintf("Precision–Recall curve (AUPRC = %.3f)", auprc))
```
Or using ggplot:
```r
pr_df <- as.data.frame(pr_obj$curve)
colnames(pr_df) <- c("Recall", "Precision", "Threshold")

ggplot(pr_df, aes(x = Recall, y = Precision)) +
  geom_line(size = 1.1) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = sprintf("Precision–Recall – Top-10 feature Random Forest (AUPRC = %.3f)", auprc),
    x = "Recall (sensitivity)",
    y = "Precision (positive predictive value)"
  ) +
  theme_minimal(base_size = 14)

ggsave("fig_pr_top10_rf.png", width = 6, height = 5, dpi = 300)
```
These saved figures (fig_roc_top10_rf.png and fig_pr_top10_rf.png) can be
included in reports or presentations to visually summarise model performance.

---
## Summary metrics (Accuracy and F1 score)
To report simple summary metrics at a chosen threshold (for example, 0.5):
```r
threshold <- 0.5

pred_label <- ifelse(prob_pos >= threshold, 1, 0)

tp <- sum(pred_label == 1 & y_true == 1)
tn <- sum(pred_label == 0 & y_true == 0)
fp <- sum(pred_label == 1 & y_true == 0)
fn <- sum(pred_label == 0 & y_true == 1)

accuracy <- (tp + tn) / length(y_true)
precision <- tp / (tp + fp)
recall    <- tp / (tp + fn)
F1        <- 2 * precision * recall / (precision + recall)

c(
  Accuracy = accuracy,
  Precision = precision,
  Recall = recall,
  F1 = F1,
  AUROC = auroc,
  AUPRC = auprc
)
```
These values can be reported alongside the ROC and PR curves to provide a
quantitative summary of the model’s performance.

## Model Performance Visualization

To showcase the model’s predictive power, I have included the ROC and PRC curve images.

![](Figures/ROC_curve.png)
![](Figures/PRC_curve.png)
