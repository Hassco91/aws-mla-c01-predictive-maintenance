import argparse
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_column_names(df):
    df = df.copy()
    df.columns = [re.sub(r"[\[\]<>]", "", str(c)).strip().replace(" ", "_") for c in df.columns]
    return df

def add_physics_features(df):
    d = df.copy()
    d["Power_W"] = d["Torque_Nm"] * d["Rotational_speed_rpm"] * 2 * 3.14159 / 60
    d["Temp_diff_K"] = d["Process_temperature_K"] - d["Air_temperature_K"]
    d["Wear_x_Torque"] = d["Tool_wear_min"] * d["Torque_Nm"]
    d["Torque_per_rpm"] = d["Torque_Nm"] / d["Rotational_speed_rpm"]
    d["Wear_squared"] = d["Tool_wear_min"] ** 2
    return d

if __name__ == "__main__":
    import glob
    csv_files = glob.glob("/opt/ml/processing/input/**/*.csv", recursive=True)
    input_path = csv_files[0]
    print(f"Verwende Datei: {input_path}")
    df = pd.read_csv(input_path)
    df = clean_column_names(df)

    df_encoded = pd.get_dummies(df, columns=["Type"], drop_first=True)
    bool_cols = [c for c in df_encoded.columns if df_encoded[c].dtype == bool]
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    target = "Machine_failure"
    cols = [target] + [c for c in df_encoded.columns if c != target]
    df_encoded = df_encoded[cols]

    train, temp = train_test_split(df_encoded, test_size=0.3, stratify=df_encoded[target], random_state=42)
    validation, test = train_test_split(temp, test_size=0.5, stratify=temp[target], random_state=42)

    train = add_physics_features(train)
    validation = add_physics_features(validation)
    test = add_physics_features(test)

    # SageMaker Processing schreibt Ausgaben nach /opt/ml/processing/{train,validation,test}
    train.to_csv("/opt/ml/processing/train/train.csv", header=False, index=False)
    validation.to_csv("/opt/ml/processing/validation/validation.csv", header=False, index=False)
    test.to_csv("/opt/ml/processing/test/test.csv", header=False, index=False)

    print("Preprocessing abgeschlossen.")