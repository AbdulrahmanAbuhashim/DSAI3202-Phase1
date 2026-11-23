# Preliminary Model Comparison

Dataset: 50,000 road accident records
Training: 40,000 records
Testing: 10,000 records
Random state: 42

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.52 | 0.3160 |
| Gradient-Boosted Trees | 0.75 | 0.2862 |
| Balanced Gradient-Boosted Trees | 0.48 | 0.3108 |

## Findings

Gradient-Boosted Trees achieved the highest accuracy but
mostly predicted Slight accidents.

Using class weights improved its macro F1-score from
0.2862 to 0.3108.

Logistic Regression still achieved the highest macro F1.

The hypothesis requires an improvement of at least 0.05.
It was not supported in this preliminary experiment.

These results are from a 50,000-record sample, not the
final Phase 2 evaluation.
