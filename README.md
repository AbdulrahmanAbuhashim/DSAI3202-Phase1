# Predicting Road Accident Severity in the UK
## DSAI3202 — Phase 1

### Project Objective
Prepare and explore UK road collision data to support
future prediction of accident severity.

### Dataset
Source: UK Department for Transport
Period: 1979–2025
Original records: 9,116,625
Source columns: 44

### Azure Architecture
1. Azure Blob Storage — Raw data
2. Azure Linux Virtual Machine — Python ETL
3. Azure Blob Storage — Processed Parquet data
4. Metadata JSON — Dataset documentation and lineage

### Data Processing
- Batch size: 50,000 records
- Total processed: 9,116,625
- Accepted: 9,116,617
- Rejected: 8
- Duplicate records: 0
- Rejection threshold: 1%
- Quality gate: PASSED
- Parquet files: 183

### Exploratory Data Analysis
Four visualizations:
1. Collision severity distribution
2. Yearly collision trends
3. Severity by speed limit
4. Severity by weather conditions

Charts are stored in evidence/eda/.

### Feature Engineering
Five engineered features:
- hour
- month
- is_weekend
- is_night
- high_speed_road

Feature engineering was tested successfully on
1,000 records. The full dataset has not yet been
feature-engineered.

### Data Quality
- Required columns validated
- Collision identifiers checked for duplicates
- Invalid records rejected
- Missing geographical coordinates preserved
- Output record counts verified
- Processed data published to Azure Storage

### Project Status
ETL, metadata documentation, EDA chart generation,
and sample feature engineering completed.

### Next Phase
Model development, training, and evaluation.

### Reproducibility

Environment: Ubuntu 24.04, Python 3.12.

Install dependencies:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

The ETL pipeline is implemented in:
scripts/etl_pipeline.py

Configuration:
config/etl_config.json

Processed dataset:
Azure Blob Storage / processed /
projects/road-accident-severity/processed_v1/

Analysis evidence:
evidence/eda/

Feature definitions:
config/features_v1.md

Schema:
config/schema_v1.json

### Repository and Data Availability

Source code and documentation are maintained in Git.
Large source CSV and processed Parquet files are stored
in Azure Blob Storage rather than Git.

### AI Use Disclosure

ChatGPT was used to assist with Python code development,
ETL troubleshooting, feature-engineering scripts,
EDA visualization code, and documentation drafting.

Generated code was executed and tested in the Azure VM.
ETL counts and data quality outputs were verified
against the resulting files.

### Limitations

Feature engineering was tested on 1,000 records.
No predictive model has been trained in Phase 1.
