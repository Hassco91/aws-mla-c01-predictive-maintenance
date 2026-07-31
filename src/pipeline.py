"""
SageMaker Pipeline fuer das Predictive-Maintenance-Projekt.

Automatisiert den kompletten Workflow:
Preprocessing -> Training -> Evaluation -> Bedingung (F1 >= 0.75) -> Modellregistrierung.

Hinweis: Die Trainings-Stufe laeuft als ProcessingStep (nicht als klassischer
TrainingStep), da das AWS-Konto keine Service-Quota fuer SageMaker Training
Jobs besitzt (0 Instances fuer alle getesteten Instance-Typen). Processing
Jobs haben dagegen ausreichend Quota. Diese Umgehung wird im README als
reales Praxis-Learning dokumentiert.
"""

import boto3
import sagemaker
from sagemaker.image_uris import retrieve
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep


def build_pipeline(role: str, bucket: str, session: sagemaker.Session) -> Pipeline:
    """Baut die komplette SageMaker Pipeline und gibt sie zurueck."""

    region = session.boto_region_name
    xgboost_image_uri = retrieve(framework="xgboost", region=region, version="1.7-1")

    # Parameter: koennen bei jedem pipeline.start() ueberschrieben werden
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.t3.xlarge")
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.t3.xlarge")
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    input_data = ParameterString(name="InputData", default_value=f"s3://{bucket}/processed/ai4i2020/")

    # --- Schritt 1: Preprocessing (Aufteilen in Train/Validation/Test + Feature Engineering) ---
    from sagemaker.sklearn.processing import SKLearnProcessor

    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        sagemaker_session=session,
    )

    step_process = ProcessingStep(
        name="PreprocessPredictiveMaintenanceData",
        processor=sklearn_processor,
        code="preprocessing.py",
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
    )

    # --- Schritt 2: Training als ProcessingStep (Umgehung der Training-Job-Quota=0) ---
    train_processor = ScriptProcessor(
        image_uri=xgboost_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        role=role,
        base_job_name="train-processing",
    )

    step_train = ProcessingStep(
        name="TrainXGBoostModel",
        processor=train_processor,
        code="train_processing.py",
        inputs=[
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                destination="/opt/ml/processing/train",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                destination="/opt/ml/processing/validation",
            ),
        ],
        outputs=[ProcessingOutput(output_name="model", source="/opt/ml/processing/model")],
    )

    # --- Schritt 3: Evaluation (F1/Precision/Recall auf Testdaten) ---
    eval_processor = ScriptProcessor(
        image_uri=xgboost_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=session,
    )

    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")

    step_eval = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        code="evaluation.py",
        inputs=[
            ProcessingInput(
                source=step_train.properties.ProcessingOutputConfig.Outputs["model"].S3Output.S3Uri,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        property_files=[evaluation_report],
    )

    # --- Schritt 4: Modellregistrierung (nur bei F1 >= 0.75) ---
    # PipelineSession ist noetig, damit model.register() nur "step_args" zurueckgibt
    # (verzoegerte Ausfuehrung), statt sofort einen echten API-Call zu machen.
    pipeline_session = PipelineSession()

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )

    # model_data muss auf die exakte tar.gz-Datei zeigen, nicht nur auf den Ordner
    model_data_url = Join(
        on="/",
        values=[
            step_train.properties.ProcessingOutputConfig.Outputs["model"].S3Output.S3Uri,
            "model.tar.gz",
        ],
    )

    model = Model(
        image_uri=xgboost_image_uri,
        model_data=model_data_url,
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_register = ModelStep(
        name="RegisterPredictiveMaintenanceModel",
        step_args=model.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.t2.medium"],
            transform_instances=["ml.m5.large"],
            model_package_group_name="predictive-maintenance-models",
            approval_status=model_approval_status,
            model_metrics=model_metrics,
        ),
    )

    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.f1.value",
        ),
        right=0.75,
    )

    step_cond = ConditionStep(
        name="CheckF1ScoreThreshold",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )

    pipeline = Pipeline(
        name="predictive-maintenance-pipeline",
        parameters=[processing_instance_type, training_instance_type, model_approval_status, input_data],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=session,
    )

    return pipeline


def main():
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = "hassco-predictive-maintenance-ai-model2"

    pipeline = build_pipeline(role=role, bucket=bucket, session=session)
    pipeline.upsert(role_arn=role)

    execution = pipeline.start()
    print("Pipeline-Ausfuehrung gestartet. Execution ARN:", execution.arn)


if __name__ == "__main__":
    main()
