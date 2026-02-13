# -*- coding: utf-8 -*-
"""
UAS Data Science - Campus Recruitment
Studi kasus: Faktor Akademik yang Mempengaruhi Penempatan Kerja

Cara menjalankan (lokal):
1) pip install -r requirements.txt
2) python analysis_campus_recruitment.py

Catatan:
- File dataset dipakai: Campus Recruitment.csv
- Target: status kelulusan (Bekerja/Belum)
- Kolom Gaji tidak dipakai untuk prediksi (potensi data leakage)
"""

import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = r"/mnt/data/Campus Recruitment.csv"
OUT_DIR = r"/mnt/data/UAS_CampusRecruitment_GitHub"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

TARGET = "status kelulusan (Bekerja/Belum)"
SALARY = "Gaji"

# EDA: Distribusi target
plt.figure()
df[TARGET].value_counts().plot(kind="bar")
plt.title("Distribusi Status Placement")
plt.xlabel("Status")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_distribusi_status.png"), dpi=200)
plt.close()

# Korelasi numerik
num_cols_orig = ["Nilai rata-rata SMP","Nilai rata-rata SMA","IPK","Nilai tes kemampuan kerja","Nilai rata-rata pascasarjana"]
corr = df[num_cols_orig].corr(numeric_only=True)
plt.figure(figsize=(6,5))
plt.imshow(corr.values)
plt.title("Korelasi Variabel Numerik")
plt.xticks(range(len(num_cols_orig)), num_cols_orig, rotation=90)
plt.yticks(range(len(num_cols_orig)), num_cols_orig)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_korelasi_numerik.png"), dpi=200)
plt.close()

# Persiapan data model
X = df.drop(columns=[TARGET, SALARY], errors="ignore").copy()
y = df[TARGET].astype(str)

# Drop ID jika ada
if "ID" in X.columns:
    X = X.drop(columns=["ID"])

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

log_reg = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
])

rf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(n_estimators=600, random_state=42, class_weight="balanced", n_jobs=-1))
])

log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)

def eval_and_save(pipe, model_name):
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=["Not Placed","Placed"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Placed","Placed"])
    fig, ax = plt.subplots(figsize=(5,4))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"cm_{model_name.lower().replace(' ','_')}.png"), dpi=200)
    plt.close()

    print("\n=== {model_name} ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    proba = pipe.predict_proba(X_test)[:, list(pipe.named_steps["model"].classes_).index("Placed")]
    fpr, tpr, _ = roc_curve((y_test=="Placed").astype(int), proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

lr_fpr, lr_tpr, lr_auc = eval_and_save(log_reg, "Logistic Regression")
rf_fpr, rf_tpr, rf_auc = eval_and_save(rf, "Random Forest")

# ROC Curve comparison
plt.figure(figsize=(6,4))
plt.plot(lr_fpr, lr_tpr, label=f"LogReg (AUC={lr_auc:.3f})")
plt.plot(rf_fpr, rf_tpr, label=f"RF (AUC={rf_auc:.3f})")
plt.plot([0,1],[0,1], linestyle="--")
plt.title("ROC Curve - Perbandingan Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"), dpi=200)
plt.close()

# Feature importance (RF)
ohe = rf.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)
all_feature_names = np.concatenate([np.array(num_cols), cat_feature_names])

importances = rf.named_steps["model"].feature_importances_
imp_df = (pd.DataFrame({"feature": all_feature_names, "importance": importances})
          .sort_values("importance", ascending=False))

top10 = imp_df.head(10).iloc[::-1]
plt.figure(figsize=(7,4))
plt.barh(top10["feature"], top10["importance"])
plt.title("Feature Importance (Top 10) - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "feature_importance_top10.png"), dpi=200)
plt.close()

imp_df.to_csv(os.path.join(OUT_DIR, "feature_importance_all.csv"), index=False)
print("\nSelesai. Output tersimpan di:", OUT_DIR)
