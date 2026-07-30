"""
Deployment-Skript für das Predictive-Maintenance-Modell.

Stellt das trainierte XGBoost-Modell als SageMaker Real-Time Endpoint bereit,
testet es mit Beispieldaten und löscht es anschließend wieder, um Kosten zu vermeiden.

Wichtig: Das Modell muss im nativen XGBoost-Format gespeichert sein
(booster.save_model), nicht mit joblib - sonst schlägt der Health Check
des SageMaker-Containers fehl.
"""

import boto3
import sagemaker
from botocore.exceptions import ClientError
from sagemaker import image_uris
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer

BUCKET = "hassco-predictive-maintenance-ai-model2"
ENDPOINT_NAME = "predictive-maintenance-xgb"
INSTANCE_TYPE = "ml.t2.medium"
XGBOOST_VERSION = "1.7-1"


def cleanup_existing(sm_client, endpoint_name: str):
    """Entfernt vorhandene Endpoints/Konfigurationen mit gleichem Namen."""
    for delete_fn, kwargs, label in [
        (sm_client.delete_endpoint, {"EndpointName": endpoint_name}, "Endpoint"),
        (sm_client.delete_endpoint_config, {"EndpointConfigName": endpoint_name}, "Konfiguration"),
    ]:
        try:
            delete_fn(**kwargs)
            print(f"Vorhandene {label} gelöscht.")
        except ClientError:
            pass


def deploy_model(session, role, region, model_s3_path: str) -> Predictor:
    """Stellt das Modell als Real-Time Endpoint bereit."""
    container = image_uris.retrieve("xgboost", region, version=XGBOOST_VERSION)

    model = Model(
        image_uri=container,
        model_data=model_s3_path,
        role=role,
        sagemaker_session=session,
    )

    return model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )


def test_endpoint(session):
    """Testet den Endpoint mit einem normalen und einem kritischen Betriebszustand."""
    predictor = Predictor(
        endpoint_name=ENDPOINT_NAME,
        sagemaker_session=session,
        serializer=CSVSerializer(),
    )

    # Reihenfolge: Air_temp, Process_temp, Rot_speed, Torque, Tool_wear, Type_L, Type_M,
    #              Power_W, Temp_diff_K, Wear_x_Torque, Torque_per_rpm, Wear_squared
    testfaelle = {
        "Normaler Betrieb": [298.1, 308.6, 1551, 42.8, 0, 1, 0, 6950.9, 10.5, 0, 0.0276, 0],
        "Kritischer Zustand": [302.5, 312.0, 1300, 65.0, 220, 1, 0, 8848.0, 9.5, 14300, 0.05, 48400],
    }

    for name, werte in testfaelle.items():
        antwort = predictor.predict(werte)
        score = float(antwort.decode("utf-8").strip())
        status = "AUSFALL" if score > 0.5 else "OK"
        print(f"{name}: Ausfallwahrscheinlichkeit = {score:.4f} -> {status}")

    return predictor


def main():
    session = sagemaker.Session()
    region = session.boto_region_name
    role = sagemaker.get_execution_role()
    sm_client = boto3.client("sagemaker", region_name=region)

    model_s3_path = f"s3://{BUCKET}/xgboost/model/model.tar.gz"

    print("Räume vorhandene Ressourcen auf ...")
    cleanup_existing(sm_client, ENDPOINT_NAME)

    print("Stelle Modell bereit (dauert ca. 5-8 Minuten) ...")
    deploy_model(session, role, region, model_s3_path)
    print(f"Endpoint bereit: {ENDPOINT_NAME}")

    print("\nTeste Endpoint ...")
    predictor = test_endpoint(session)

    print("\nLösche Endpoint, um laufende Kosten zu vermeiden ...")
    predictor.delete_endpoint()
    print("Fertig.")


if __name__ == "__main__":
    main()