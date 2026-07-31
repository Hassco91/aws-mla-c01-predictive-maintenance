import json
import tarfile
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score

if __name__ == "__main__":
    # Modell aus dem Trainingsjob entpacken
    with tarfile.open("/opt/ml/processing/model/model.tar.gz") as tar:
        tar.extractall(path=".")

    model = xgb.Booster()
    model.load_model("xgboost-model")

    # Testdaten laden (erste Spalte = Zielspalte, wie im Preprocessing-Schritt)
    test_df = pd.read_csv("/opt/ml/processing/test/test.csv", header=None)
    y_test = test_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]

    dtest = xgb.DMatrix(X_test)
    predictions_proba = model.predict(dtest)
    predictions = (predictions_proba > 0.5).astype(int)

    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    print(f"F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")

    # Standardformat, das SageMaker Pipelines' ConditionStep lesen kann
    report_dict = {
        "binary_classification_metrics": {
            "f1": {"value": f1, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
        }
    }

    with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
        f.write(json.dumps(report_dict))