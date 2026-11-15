# Exploratory Data Analysis — Phase 1

## Dataset
UK road collisions, 1979–2025.

Population:
- Input records: 9,116,625
- Accepted records: 9,116,617
- Rejected records: 8

All four EDA charts were generated from the
9,116,617 accepted records, not a sample.

## 1. Severity distribution
- Fatal: 146,175
- Serious: 1,676,556
- Slight: 7,293,886

Finding: Slight collisions dominate the dataset.
Modeling implication: Class imbalance must be
considered when evaluating severity predictions.

## 2. Yearly collision trends
The yearly trend chart covers 1979–2025.

Modeling implication: Historical data may differ
from recent data. A time-based validation strategy
should be considered.

## 3. Speed limits
The chart compares collision severity across
20, 30, 40, 50, 60 and 70 mph roads.

Modeling implication: Speed limit is a potential
predictive feature, but association does not
establish causation.

## 4. Weather conditions
The chart examines collision severity across
recorded weather condition codes.

Data risk: Code -1 appears in the dataset and
requires interpretation as a missing or unknown
category before modeling.

## Data quality and limitations
- Missing geographical coordinates occur in
  historical records.
- Severity classes are imbalanced.
- Historical reporting practices may vary.
- Missing and unknown category codes require
  appropriate treatment.
- Further correlation and outlier examination
  is needed before claiming those analyses complete.

## EDA evidence
- severity_distribution.png
- yearly_collision_trend.png
- severity_by_speed_limit.png
- severity_by_weather.png

## Correlation and outlier checks

Sample: 10,000 records, using the first 1,000 rows
from each of the first 10 processed Parquet files.
This is a deterministic sample, not a random sample,
and may not represent the entire dataset.

Results:
- Speed limit vs collision severity: -0.119.
- No missing values in the four examined columns.
- No speed limits outside 0–70 mph.
- No nonpositive vehicle or casualty counts.

Interpretation:
- Speed limit has a weak negative Pearson correlation
  with the numerically coded severity target.
- Severity is ordinal (1=fatal, 3=slight), so Pearson
  correlation is only a descriptive screening measure.
- Casualty count is examined for data quality only;
  it is excluded from predictive features due to leakage risk.
- These checks do not establish causation or guarantee
  data quality across the entire population.

Detailed results: numeric_checks.txt.
