import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import joblib

# ================= LOAD DATA =================
df = pd.read_csv(r"E:\aditi_eeg\visual_eeg\final_dataset\final_dataset.csv")

X = df.drop("Label", axis=1).values
y = df["Label"].values

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= SCALER =================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)

# ================= MODEL (BEST ONE ONLY) =================
model = SVC()   # you can replace with best model

model.fit(X_train_s, y_train)

# ================= SAVE =================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")