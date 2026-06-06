"""
modelling.py for MLProject (Workflow-CI)
========================================
Script training yang menerima parameter CLI untuk dijalankan via MLflow Project.
Melacak eksperimen ke DagsHub.
"""

import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import contextlib

@contextlib.contextmanager
def get_run_context():
    if mlflow.active_run() is not None:
        print("[INFO] Logging to existing active run.")
        yield mlflow.active_run()
    else:
        print("[INFO] Starting a new MLflow run.")
        with mlflow.start_run() as run:
            yield run

# Argumen CLI
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=10)
args = parser.parse_args()

# Setup DagsHub
DAGSHUB_USERNAME = "Andreal14"
DAGSHUB_REPO     = "msml-project"

print(f"[INFO] Menginisialisasi DagsHub untuk {DAGSHUB_USERNAME}/{DAGSHUB_REPO}...")
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)

# Muat data
DATA_DIR = "winequality_preprocessing"
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

mlflow.set_experiment("Workflow-CI")

with get_run_context():
    # Log parameters only if not running within an MLflow Project (which automatically logs them)
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average='weighted')
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score_weighted", f1)

    # Simpan model
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="wine-quality-rf-model"
    )

    # Artefak: Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=["Bad Wine", "Good Wine"],
        yticklabels=["Bad Wine", "Good Wine"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - CI Run")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    print(f"[RESULT] Accuracy: {acc:.4f} | F1: {f1:.4f}")
