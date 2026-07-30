### Phase 3 — Deployment (Done)

The winning model (XGBoost + physics-based features) was deployed to a SageMaker real-time inference endpoint (`ml.t2.medium`) and validated with live inference requests:

| Input scenario | Predicted failure probability | Decision |
|---|---|---|
| Normal operation (low torque, no tool wear) | 0.0051 | OK |
| Critical state (high torque, 220 min tool wear) | 0.9791 | FAILURE |

The endpoint was deleted after validation to avoid ongoing hosting costs. The full deployment workflow — cleanup, deploy, inference test, teardown — is reproducible via `src/deploy.py`.

**Implementation note:** SageMaker's managed XGBoost container requires the model artifact in native XGBoost format (`booster.save_model()`). An initial deployment attempt using a `joblib`-serialized scikit-learn wrapper failed the container health check — a subtle but important distinction when serving models through AWS-managed inference containers.

# Predictive Maintenance on AWS — MLA-C01 Portfolio Project

An end-to-end machine learning pipeline for predictive maintenance, built on AWS as a hands-on portfolio project while preparing for the **AWS Certified Machine Learning Engineer – Associate (MLA-C01)** certification.

## Overview

Unplanned equipment failure is one of the costliest problems in manufacturing. This project builds a binary classification system that predicts machine failure from real-time sensor readings (temperature, rotational speed, torque, tool wear), using the **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository, 10,000 records).

The project deliberately covers all four MLA-C01 exam domains end to end:

| Domain | Coverage |
|---|---|
| Data Preparation for ML | S3 ingestion, SageMaker Data Wrangler, data quality analysis, target leakage detection |
| ML Model Development | AutoML baseline (SageMaker Autopilot), manual XGBoost, hyperparameter tuning, feature engineering |
| Deployment & Orchestration | SageMaker real-time inference endpoint, reproducible deployment script |
| Monitoring, Maintenance & Security | Encrypted/versioned S3, scoped IAM roles, SageMaker Model Monitor (planned) |

## Architecture

## Tech Stack

- **Storage:** Amazon S3
- **Data Preparation:** Amazon SageMaker Data Wrangler
- **AutoML:** Amazon SageMaker Canvas (Autopilot, Standard Build)
- **Model Development:** Amazon SageMaker Studio (JupyterLab), XGBoost, scikit-learn
- **Hyperparameter Optimization:** `RandomizedSearchCV` (local), SageMaker Automatic Model Tuning (planned for production pipeline)
- **Deployment:** Amazon SageMaker Real-Time Inference Endpoint
- **Infrastructure:** AWS IAM, AWS CLI
- **Version Control:** Git / GitHub

## Repository Structure

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
10,000 rows, 14 columns, binary target (`Machine failure`), class imbalance ≈ 3.4% positive rate.

## Methodology & Results

### Phase 0 — Infrastructure Setup

- Provisioned an IAM user with scoped programmatic access and configured the AWS CLI.
- Created an encrypted (SSE-S3), versioned S3 bucket in `eu-central-1` (Frankfurt), aligned with GDPR data residency expectations for the German market.
- Uploaded the raw dataset to `s3://.../raw/ai4i2020/`.

### Phase 1 — Data Preparation

Data was profiled using SageMaker Data Wrangler's **Data Quality and Insights Report**.

**Key finding — Target Leakage:** the columns `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are sub-failure-mode indicators that directly compose the `Machine failure` target (i.e., `Machine failure = 1` whenever any of them is `1`). An initial quick model trained on the raw data scored an unrealistic F1 of 0.97 — a textbook example of target leakage.

These columns, along with non-predictive identifiers (`UDI`, `Product ID`), were dropped. Re-running the quality report on the cleaned data produced a realistic baseline (F1 = 0.729, Recall = 0.632), confirming the leakage had been eliminated. The cleaned dataset was exported to `s3://.../processed/ai4i2020/`.

### Phase 2 — Model Development

Three modeling approaches were benchmarked on an identical held-out test set (15% of the data, stratified split):

| Approach | F1 | Precision | Recall |
|---|---|---|---|
| SageMaker Autopilot (Standard Build) | **0.820** | 0.926 | 0.735 |
| Manual XGBoost, raw features, `scale_pos_weight` tuned only | 0.716 | 0.672 | 0.765 |
| Manual XGBoost, raw features, 100-trial hyperparameter search | 0.718 | 0.712 | 0.725 |
| **Manual XGBoost + physics-based feature engineering** | **0.860** | 0.878 | **0.843** |

**Class imbalance handling:** the dataset has a ~28.5:1 negative-to-positive ratio. `scale_pos_weight` was swept across [1, 3, 5, 10, 15, 20, 28.5] to characterize the precision–recall trade-off explicitly, rather than applying the theoretical ratio blindly — the mathematically "correct" weight (28.5) in fact produced the worst F1 by over-correcting toward recall at the expense of precision.

**Feature engineering:** three domain-informed features were engineered from the known physical failure mechanisms documented for this dataset:

- `Power_W` = Torque × Rotational speed (angular power, relevant to Power Failure)
- `Temp_diff_K` = Process temperature − Air temperature (relevant to Heat Dissipation Failure)
- `Wear_x_Torque` = Tool wear × Torque (relevant to Overstrain Failure)

Adding these features and re-running the hyperparameter search produced the best-performing model overall (F1 = 0.860), surpassing the AutoML baseline.

**Key takeaway:** AutoML (Autopilot) provided a strong, fast baseline by searching across algorithms and hyperparameters automatically. However, domain-informed feature engineering — encoding known physical failure mechanisms directly as features — outperformed both the AutoML baseline and extensive hyperparameter tuning on raw features. This demonstrates that automated tooling and subject-matter expertise are complementary, not substitutes for one another.

### Phase 3 — Deployment (Done)

The winning model (XGBoost + physics-based features) was deployed to a SageMaker real-time inference endpoint (`ml.t2.medium`) and validated with live inference requests:

| Input scenario | Predicted failure probability | Decision |
|---|---|---|
| Normal operation (low torque, no tool wear) | 0.0051 | OK |
| Critical state (high torque, 220 min tool wear) | 0.9791 | FAILURE |

The endpoint was deleted after validation to avoid ongoing hosting costs. The full deployment workflow — cleanup, deploy, inference test, teardown — is reproducible via `src/deploy.py`.

**Implementation note:** SageMaker's managed XGBoost container requires the model artifact in native XGBoost format (`booster.save_model()`). An initial deployment attempt using a `joblib`-serialized scikit-learn wrapper failed the container health check — a subtle but important distinction when serving models through AWS-managed inference containers.


### Phase 4 — Monitoring & Security (Attempted, blocked by AWS platform change)

A full Model Monitor workflow was implemented: data capture was enabled on the endpoint, five test inferences were sent to generate captured request/response data, and a baseline job (`suggest_baseline`) was successfully run against the training data to produce `statistics.json` and `constraints.json`.

**Finding:** As of July 30, 2026, AWS moved 10 SageMaker AI features — including Model Monitor, Clarify, Ground Truth, and Debugger — into maintenance mode for new AWS accounts. `CreateDataQualityJobDefinition` (the API underlying `create_monitoring_schedule`) now returns a `ValidationException` for new customers: *"This operation is in maintenance mode and is not available to new customers."* This is a live platform change, not a bug in this project's code — existing AWS accounts are unaffected.

**Practical impact:** the monitoring schedule itself could not be created on this (new) account. The baseline artifacts and data capture logs remain in S3 as evidence of a correctly implemented workflow up to the platform-imposed boundary. For a production environment on an established account, the same code would create a working hourly monitoring schedule.

**Notes for future reference:** instance-type quota constraints were also encountered during this phase — `ml.t3.medium` (4GB RAM) was insufficient for the Spark-based baseline analysis (job killed by OOM), while `ml.t3.xlarge` (16GB RAM) succeeded. `ml.m5.large` and `ml.c5.xlarge` had zero account-level quota for processing jobs.

- SageMaker Clarify for bias and explainability monitoring — also affected by the same maintenance-mode change
- CloudWatch alarms on endpoint latency and error rate — remains available and would be the primary supported monitoring path going forward

### Phase 5 — Pipeline Automation (Planned)

The manual workflow above will be re-implemented as a SageMaker Pipeline (processing → training → evaluation → conditional registration → deployment) to demonstrate a reproducible, CI/CD-ready MLOps workflow.

## Key Learnings

1. **Always audit for target leakage before trusting a model's metrics** — a near-perfect score is a red flag, not a success.
2. **The theoretically correct class weight is not always the practically optimal one** — empirical sweeps matter.
3. **Domain knowledge, encoded as features, can outperform both AutoML and hyperparameter tuning** — the biggest single performance gain in this project (F1 +0.14) came from three physics-based features, not from model tuning.
4. **Model serialization format matters for deployment** — a scikit-learn-pickled model can fail silently against a managed inference container that expects the framework's native format.

## Author

Built by Hassco as part of AWS Certified Machine Learning Engineer – Associate (MLA-C01) exam preparation and portfolio development for the German job market.