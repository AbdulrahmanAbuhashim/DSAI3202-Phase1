# Phase 2 - Hypothesis

## Project
Predicting Road Accident Severity in the UK

## Hypothesis
Using road, weather, time, and location information,
a Gradient-Boosted Trees model will achieve a macro
F1-score at least 0.05 higher than Logistic Regression.

## Models
- Baseline: Logistic Regression
- Comparison: Gradient-Boosted Trees

## Evaluation
We will evaluate both models using the same test set.

We will compare their macro F1-scores and examine
the results for each severity class.

The hypothesis is supported if:

Gradient-Boosted Trees macro F1 - Logistic Regression macro F1 >= 0.05

Otherwise, the hypothesis is not supported.
