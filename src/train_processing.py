import glob
import os
import tarfile

import pandas as pd
import xgboost as xgb

if __name__ == "__main__":
    # Trainings- und Validierungsdaten aus den ProcessingStep-Ausgaben laden
    train_path = glob.glob("/opt/ml/processing/train/*.csv")[0]
    val_path = glob.glob("/opt/ml/processing/validation/*.csv")[0]

    train_df = pd.read_csv(train_path, header=None)
    val_df = pd.read_csv(val_path, header=None)

    # Erste Spalte ist das Ziel (Machine_failure), Rest sind Features
    y_train = train_df.iloc[:, 0]
    X_train = train_df.iloc[:, 1:]
    y_val = val_df.iloc[:, 0]
    X_val = val_df.iloc[:, 1:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Gleiche Hyperparameter wie im ursprünglichen manuellen Training
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": 5,
        "max_depth": 4,
        "eta": 0.01,
        "subsample": 1.0,
        "min_child_weight": 1,
        "gamma": 0.1,
        "colsample_bytree": 0.6,
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "validation")],
        verbose_eval=50,
    )

    # Modell im nativen XGBoost-Format speichern
    # (wichtig: kompatibel mit dem SageMaker XGBoost Inference Container,
    # im Gegensatz zu joblib/pickle)
    output_dir = "/opt/ml/processing/model"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "xgboost-model")
    booster.save_model(model_path)

    # Manuell zu model.tar.gz packen
    # (Training Jobs machen das automatisch, ProcessingStep nicht)
    tar_path = os.path.join(output_dir, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_path, arcname="xgboost-model")

    print("Training abgeschlossen. Modell gespeichert unter:", tar_path)