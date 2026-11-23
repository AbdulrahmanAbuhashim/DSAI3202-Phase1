# Final Model Validation Comparison

Training data: 420,386 records (1979-2018)
Validation data: 20,713 records (2019-2022)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.6292 | 0.3055 |
| Gradient-Boosted Trees | 0.6204 | 0.3071 |

## Model Selection

Gradient-Boosted Trees achieved a slightly higher macro F1
than Logistic Regression.

The improvement was 0.0016, which is below the required
0.05 improvement.

The hypothesis was not supported by the validation results.

Gradient-Boosted Trees was selected because it had the
higher validation macro F1.

Both models had difficulty predicting Fatal and Serious
accidents.

The final test data (2023-2025) has not been used yet.
