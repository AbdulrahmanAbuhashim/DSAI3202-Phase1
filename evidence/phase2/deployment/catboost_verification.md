# CatBoost Azure Deployment Verification

## Model
- Registered model: road-accident-severity:3
- Algorithm: CatBoost Gradient-Boosted Trees
- Trees: 464
- Features: 21
- Validation macro F1: 0.3842
- Validation accuracy: 0.7195

## Azure Deployment
- Workspace: mlwdsai320260301190
- Endpoint: accident-severity-60301190
- Deployment: catboost
- Instance: Standard_DS3_v2
- Provisioning state: Succeeded

## Functional Testing
- Valid request: Passed
- Invalid request: Missing features detected
- Offline vs deployed predictions: 10/10 matched
- Prediction labels: 1=Fatal, 2=Serious, 3=Slight

## Performance
- Target client-observed latency: 5 seconds
- Request 1: 3.65 seconds
- Request 2: 1.93 seconds
- Request 3: 2.06 seconds
- Request 4: 2.45 seconds
- Request 5: 2.00 seconds
- Average: 2.42 seconds
- Minimum: 1.93 seconds
- Maximum: 3.65 seconds
- All five measured requests were below 5 seconds.

The measurement includes Azure CLI execution, network communication,
and inference. It is not pure server-side model latency.

## Limitations
- Fatal and Serious predictions remain weak.
- Validation data was used for model selection.
- An independent final holdout evaluation remains unresolved.
- Invalid-input errors are returned as JSON; an appropriate HTTP
  error status has not been verified.

## Resource Use
One Standard_DS3_v2 managed online instance was used.
The endpoint can be deleted to avoid ongoing compute charges
and redeployed for assessment.
