#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# ================= LOAD DATA =================
file_path = r"E:\aditi_eeg\visual_eeg\final_dataset\final_dataset.csv"
df = pd.read_csv(file_path)

X = df.drop("Label", axis=1)
y = df["Label"]


# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ================= SCALING =================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# ================= MODELS =================
models = {
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier()
}

results = {}
trained_models = {}


# ================= TRAIN + CONFUSION MATRIX =================
for name, model in models.items():

    if name in ["SVM", "Logistic Regression", "KNN"]:
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    trained_models[name] = model

    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # ===== CONFUSION MATRIX =====
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(6,5))

    ax = sns.heatmap(cm,
                     cmap='Blues',
                     xticklabels=["Non-TBI", "TBI"],
                     yticklabels=["Non-TBI", "TBI"],
                     linewidths=1,
                     linecolor='black',
                     cbar=True)

    # ===== FORCE SHOW ALL VALUES =====
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5,
                    str(cm[i, j]),
                    ha='center', va='center',
                    color='black',
                    fontsize=14,
                    fontweight='bold')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} - Confusion Matrix")

    plt.tight_layout()
    plt.show()


# ================= BEST MODEL =================
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)


# ================= FEATURE EXTRACTION =================
X_all = df.drop("Label", axis=1)

amp_cols = [col for col in X_all.columns if "amp" in col.lower()]
lat_cols = [col for col in X_all.columns if "lat" in col.lower()]

amplitude = X_all[amp_cols].mean(axis=1)
latency = X_all[lat_cols].mean(axis=1)


# ================= ADD TO DATAFRAME =================
df["Amplitude"] = amplitude
df["Latency"] = latency


# ================= TBI vs NON-TBI COMPARISON =================
tbi_data = df[df["Label"] == 1]
non_tbi_data = df[df["Label"] == 0]

tbi_amp_mean = tbi_data["Amplitude"].mean()
tbi_lat_mean = tbi_data["Latency"].mean()

non_tbi_amp_mean = non_tbi_data["Amplitude"].mean()
non_tbi_lat_mean = non_tbi_data["Latency"].mean()

print("\n===== TBI vs Non-TBI Comparison =====")
print(f"TBI     -> Amplitude: {tbi_amp_mean:.6f}, Latency: {tbi_lat_mean:.3f}")
print(f"Non-TBI -> Amplitude: {non_tbi_amp_mean:.6f}, Latency: {non_tbi_lat_mean:.3f}")


# ================= COMPARISON GRAPH =================
labels = ["Amplitude", "Latency"]

tbi_values = [tbi_amp_mean, tbi_lat_mean]
non_tbi_values = [non_tbi_amp_mean, non_tbi_lat_mean]

x = np.arange(len(labels))

plt.figure(figsize=(6,4))
plt.bar(x - 0.2, tbi_values, width=0.4, label="TBI")
plt.bar(x + 0.2, non_tbi_values, width=0.4, label="Non-TBI")

plt.xticks(x, labels)
plt.ylabel("Value")
plt.title("TBI vs Non-TBI Feature Comparison")
plt.legend()
plt.grid(axis='y')

plt.tight_layout()
plt.show()


# ================= SAVE RESULTS =================
output_folder = "prediction_outputs"
os.makedirs(output_folder, exist_ok=True)

comparison_df = pd.DataFrame({
    "Class": ["TBI", "Non-TBI"],
    "Amplitude": [tbi_amp_mean, non_tbi_amp_mean],
    "Latency": [tbi_lat_mean, non_tbi_lat_mean]
})

comparison_df.to_csv(os.path.join(output_folder, "comparison.csv"), index=False)

print("Comparison saved successfully.")