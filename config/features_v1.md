# Initial Feature Specification — v1

Prediction target: collision_severity (1=fatal, 2=serious, 3=slight).

| Feature | Type | Source | Definition | Available at prediction time |
|---|---|---|---|---|
| hour | Numeric | time | Hour of collision (0–23) | Only if collision time is known |
| month | Numeric | date | Calendar month (1–12) | Only if collision date is known |
| is_weekend | Binary | date | 1 for Saturday/Sunday, otherwise 0 | Only if collision date is known |
| is_night | Binary | time | 1 when hour < 6 or >= 20 | Only if collision time is known |
| high_speed_road | Binary | speed_limit | 1 when speed limit >= 50 mph | Yes, if road is known |

## Assumptions and risks

- These features assume severity is predicted after collision date,
  time and road information are available.
- Missing values require explicit handling before modeling.
- Missing hour values currently produce is_night=0; this should
  not be interpreted as a confirmed daytime collision.
- Missing speed limits currently produce high_speed_road=0;
  this should not be interpreted as a confirmed low-speed road.
- Do not use number_of_casualties, collision_injury_based,
  enhanced_severity_collision, or adjusted severity fields as
  predictors because they may reveal the outcome.
- Feature selection has not been performed.
- Feature engineering was verified on a reproducible
  1,000-record sample from the first processed Parquet file.
- The complete dataset has not been feature-engineered.
