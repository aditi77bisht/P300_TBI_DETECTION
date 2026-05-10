import numpy as np
import pandas as pd
import joblib
import os

# load once
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


def predict_visual(file_path):

    df = pd.read_csv(file_path)

    X = df.values

    # scaling
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    amplitude = df.mean(axis=1)
    latency = df.mean(axis=1) * 0.5

    final_result = []

    for p in preds:
        if p == 0:
            final_result.append("Non-TBI ✅")
        else:
            final_result.append("TBI Detected ❌")

    return final_result[0], "Consult neurologist"

predictions = best_model.predict(X_input)

output_folder = "prediction_outputs"
os.makedirs(output_folder, exist_ok=True)