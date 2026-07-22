# AWS MLA-C01 Predictive Maintenance Portfolio Project

End-to-end machine learning pipeline built on AWS to prepare for the AWS Certified Machine Learning Engineer - Associate (MLA-C01) exam, using the AI4I 2020 Predictive Maintenance Dataset.

## Project Goal

Build a production-style predictive maintenance solution covering all four MLA-C01 exam domains:
1. Data Preparation for Machine Learning
2. ML Model Development
3. Deployment and Orchestration of ML Workflows
4. ML Solution Monitoring, Maintenance, and Security

## Architecture / Tools Used

- Data source: AI4I 2020 Predictive Maintenance Dataset (UCI)
- Storage: Amazon S3 (encrypted, versioned)
- Data Preparation: Amazon SageMaker Data Wrangler (data quality report, target leakage detection, feature drop)
- Model Development: Amazon SageMaker Canvas / Autopilot (baseline), XGBoost (manual, in progress)
- Deployment: Amazon SageMaker Endpoints (planned)
- Monitoring: Amazon SageMaker Model Monitor and Clarify (planned)
- Orchestration: Amazon SageMaker Pipelines (planned)

## Progress Log

### Phase 0 - Data Selection and S3 Setup (Done)
- Selected AI4I 2020 dataset for a fast, tabular classification baseline.
- Created IAM user with scoped access keys, configured AWS CLI.
- Created an encrypted, versioned S3 bucket in eu-central-1 (Frankfurt) for GDPR-friendly data residency.
- Uploaded raw dataset to s3://.../raw/ai4i2020/.

### Phase 1 - Data Preparation (Done)
- Explored data with SageMaker Data Wrangler's Data Quality and Insights Report.
- Key finding: columns TWF, HDF, PWF, OSF, RNF are sub-failure-mode indicators that directly compose the Machine failure target column, a classic target leakage case. An initial quick model showed unrealistically perfect scores (F1 = 0.97) because of this.
- Dropped leakage columns (TWF, HDF, PWF, OSF, RNF) plus non-predictive identifiers (UDI, Product ID).
- Re-ran the quality report on the cleaned data: realistic baseline performance (F1 = 0.729, Recall = 0.632), confirming the earlier leakage.
- Exported the cleaned dataset to s3://.../processed/ai4i2020/.

### Phase 2 - Model Development (In Progress)
- Training a SageMaker Canvas Autopilot (Standard build) baseline model on the cleaned dataset.
- Next: train an XGBoost model manually, tuning scale_pos_weight to address class imbalance (about 3.4 percent failure rate), and compare against the Autopilot baseline.

### Phase 3 - Deployment (Planned)
### Phase 4 - Monitoring and Security (Planned)
### Phase 5 - SageMaker Pipelines automation (Planned)

## Dataset

Source: AI4I 2020 Predictive Maintenance Dataset - UCI
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset