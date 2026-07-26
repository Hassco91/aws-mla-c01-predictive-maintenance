"""
Trainingsskript für das Predictive-Maintenance-Modell (AI4I 2020 Datensatz).

Pipeline:
  1. Laden der bereinigten Daten aus S3 (nach Data-Wrangler-Verarbeitung)
  2. Aufteilen in Train/Validation/Test (stratifiziert)
  3. Physikbasierte Feature-Engineering-Schritte
  4. Hyperparameter-Suche mit Kreuzvalidierung (optimiert auf F1)
  5. Evaluierung auf dem Testset
  6. Speichern des finalen Modells und Hochladen nach S3
"""

import re
import joblib
import tarfile
import boto3
import pandas as pd
import sagemaker
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    make_scorer,
)

# ----------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------
BUCKET = "hassco-predictive-maintenance-ai-model2"
PROCESSED_PREFIX = "processed/ai4i2020"
TARGET_COLUMN = "Machine_failure"
RANDOM_STATE = 42


def load_processed_data(bucket: str, prefix: str) -> pd.DataFrame:
    """Lädt die von SageMaker Data Wrangler bereinigten Daten direkt aus S3."""
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    csv_key = next(
        obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".csv")
    )
    return pd.read_csv(f"s3://{bucket}/{csv_key}")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Entfernt Sonderzeichen aus Spaltennamen (XGBoost erlaubt keine [, ] oder <)."""
    df = df.copy()
    df.columns = [
        re.sub(r"[\[\]<>]", "", str(c)).strip().replace(" ", "_") for c in df.columns
    ]
    return df


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt physikbasierte Features hinzu, die auf den bekannten Ausfallmechanismen
    des AI4I-2020-Datensatzes beruhen:
      - Power_W:        Leistung = Drehmoment x Winkelgeschwindigkeit (PWF)
      - Temp_diff_K:     Temperaturdifferenz Prozess/Luft (HDF)
      - Wear_x_Torque:   Werkzeugverschleiß x Drehmoment (OSF)
    """
    d = df.copy()
    d["Power_W"] = d["Torque_Nm"] * d["Rotational_speed_rpm"] * 2 * 3.14159 / 60
    d["Temp_diff_K"] = d["Process_temperature_K"] - d["Air_temperature_K"]
    d["Wear_x_Torque"] = d["Tool_wear_min"] * d["Torque_Nm"]
    d["Torque_per_rpm"] = d["Torque_Nm"] / d["Rotational_speed_rpm"]
    d["Wear_squared"] = d["Tool_wear_min"] ** 2
    return d


def prepare_datasets(df: pd.DataFrame):
    """Kodiert kategorische Spalten und teilt die Daten stratifiziert auf."""
    df_encoded = pd.get_dummies(df, columns=["Type"], drop_first=True)
    cols = [TARGET_COLUMN] + [c for c in df_encoded.columns if c != TARGET_COLUMN]
    df_encoded = df_encoded[cols]

    bool_cols = [c for c in df_encoded.columns if df_encoded[c].dtype == bool]
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    train, temp = train_test_split(
        df_encoded, test_size=0.3, stratify=df_encoded[TARGET_COLUMN], random_state=RANDOM_STATE
    )
    validation, test = train_test_split(
        temp, test_size=0.5, stratify=temp[TARGET_COLUMN], random_state=RANDOM_STATE
    )
    return train, validation, test


def run_hyperparameter_search(X_train, y_train) -> RandomizedSearchCV:
    """Zufällige Hyperparameter-Suche mit 3-facher Kreuzvalidierung, optimiert auf F1."""
    param_dist = {
        "scale_pos_weight": [3, 5, 10, 15, 20, 25],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "n_estimators": [100, 200, 300, 500],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.5],
    }

    base_model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr", random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=100,
        scoring=make_scorer(f1_score),
        cv=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def evaluate(model, X_test, y_test, label: str):
    """Gibt F1, Precision, Recall und die Confusion Matrix für ein Modell aus."""
    y_pred = model.predict(X_test)
    print(f"\n=== {label} ===")
    print(f"F1:        {f1_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n", classification_report(y_test, y_pred, target_names=["Kein Ausfall", "Ausfall"]))
    return y_pred


def save_and_upload_model(model, session, bucket: str) -> str:
    """Speichert das Modell lokal als tar.gz und lädt es nach S3 hoch."""
    joblib.dump(model, "xgboost-model")
    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add("xgboost-model")
    return session.upload_data("model.tar.gz", bucket=bucket, key_prefix="xgboost/model")


def main():
    session = sagemaker.Session()

    print("Lade bereinigte Daten aus S3 ...")
    df = load_processed_data(BUCKET, PROCESSED_PREFIX)
    df = clean_column_names(df)

    print("Teile Daten in Train/Validation/Test auf ...")
    train, validation, test = prepare_datasets(df)

    print("Füge physikbasierte Features hinzu ...")
    train_f = add_physics_features(train)
    val_f = add_physics_features(validation)
    test_f = add_physics_features(test)

    X_train, y_train = train_f.drop(columns=[TARGET_COLUMN]), train_f[TARGET_COLUMN]
    X_test, y_test = test_f.drop(columns=[TARGET_COLUMN]), test_f[TARGET_COLUMN]

    print("Starte Hyperparameter-Suche (100 Kombinationen, 3-fache CV) ...")
    search = run_hyperparameter_search(X_train, y_train)
    print(f"\nBeste Parameter: {search.best_params_}")
    print(f"Bester F1 (Kreuzvalidierung): {search.best_score_:.3f}")

    best_model = search.best_estimator_
    evaluate(best_model, X_test, y_test, label="XGBoost + Physik-Features (final)")

    print("\nSpeichere finales Modell und lade es nach S3 hoch ...")
    model_s3_path = save_and_upload_model(best_model, session, BUCKET)
    print(f"Modell gespeichert unter: {model_s3_path}")


if __name__ == "__main__":
    main()