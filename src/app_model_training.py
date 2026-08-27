"""
app_model_training.py

Retrains a PCOS model using ONLY the exact 22 features the Flutter app's
"PMOS Detection" form will collect (matching the screenshot: hormonal/lab
markers, blood pressure, ultrasound findings, symptoms/lifestyle, plus
Age, BMI, and Cycle length/regularity). This ensures the deployed model
and the app's form fields match exactly -- no placeholder/fake values
needed for missing features.

Trains all 7 models from models.py on this feature subset, picks the
best by test F1-score (better than raw accuracy for medical screening,
since it balances precision and recall), calibrates its probability
outputs so predictions are graduated (e.g. 62%, 78%) instead of
overconfident 0%/100% extremes, and saves:
  - the fitted scaler
  - the fitted, calibrated best model
  - a metadata JSON describing feature order and encodings, so the
    FastAPI backend and Flutter app both know exactly what to send/expect.

Run from the project root:
    python src/app_model_training.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models import get_models

RANDOM_STATE = 42
TARGET_COL = "PCOS (Y/N)"

# Exact feature list the Flutter form will collect, in a fixed order.
# This order matters -- the same order is used by the FastAPI backend
# and must be used by the Flutter app when sending data.
APP_FEATURES = [
    "Age (yrs)",
    "BMI",
    "Cycle(R/I)",
    "Cycle length(days)",
    "PRL(ng/mL)",
    "Vit D3 (ng/mL)",
    "PRG(ng/mL)",
    "RBS(mg/dl)",
    "BP _Systolic (mmHg)",
    "BP _Diastolic (mmHg)",
    "Follicle No. (L)",
    "Follicle No. (R)",
    "Avg. F size (L) (mm)",
    "Avg. F size (R) (mm)",
    "Endometrium (mm)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)",
]


def load_and_prepare_data(cleaned_csv_path):
    """
    Loads the cleaned without-infertility dataset and selects only the
    app's exact feature set. Fixes the one known data-entry typo in
    Cycle(R/I) (a stray value of 5, which isn't a documented category).

    Args:
        cleaned_csv_path (str): path to pcos_without_infertility_clean.csv.

    Returns:
        tuple: (X pandas.DataFrame, y pandas.Series)
    """
    df = pd.read_csv(cleaned_csv_path)

    stray_mask = df["Cycle(R/I)"] == 5
    n_stray = stray_mask.sum()
    if n_stray > 0:
        print(f"[DECISION] Found {n_stray} row(s) with Cycle(R/I)=5, which isn't a "
              f"documented category (only 2=Regular, 4=Irregular exist). "
              f"Treating as a data-entry typo and correcting to 4 (Irregular).")
        df.loc[stray_mask, "Cycle(R/I)"] = 4

    missing = [c for c in APP_FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] These expected columns are missing from the "
                          f"cleaned dataset: {missing}")

    X = df[APP_FEATURES].copy()
    y = df[TARGET_COL].copy()

    print(f"[INFO] Using {len(APP_FEATURES)} app-matched features.")
    print(f"[INFO] Feature matrix shape: {X.shape}")
    print(f"[INFO] Class balance:\n{y.value_counts()}")

    return X, y


def train_and_select_best(X, y):
    """
    Splits the data (80/20, stratified), scales features, trains all 7
    models, selects the best one by test F1-score, then wraps it in
    probability calibration so predict_proba returns realistic,
    graduated values instead of overconfident 0%/100% extremes.

    Args:
        X (pandas.DataFrame): feature matrix (app-matched features only).
        y (pandas.Series): target vector.

    Returns:
        tuple: (best_model_name, calibrated_model, scaler, metrics_dict)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    metrics = {}

    print("\n--- Training all 7 models on app-matched feature set ---")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metrics[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        print(f"{name}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

    # NOTE: Naive Bayes previously won this comparison on raw F1-score, but it
    # was dropped as a deployment candidate. GaussianNB multiplies per-feature
    # Gaussian likelihoods, and with 22 features this makes its confidence
    # scores saturate almost immediately for any input that isn't razor-close
    # to the decision boundary -- even a single feature landing in a
    # low-density region under one class blows up the likelihood ratio.
    # After sigmoid calibration this collapsed into two near-constant output
    # values (~8% / ~85%) instead of a smooth, input-sensitive percentage,
    # which defeats the purpose of a probability screen. Random Forest
    # doesn't have this failure mode -- its probability is the fraction of
    # trees voting PCOS, which varies smoothly as inputs change -- and it
    # also scored competitively (often best) on F1/accuracy in testing, so
    # it's used as the deployment model instead of a pure F1-max pick.
    PREFERRED_DEPLOYMENT_MODEL = "RandomForest"
    if PREFERRED_DEPLOYMENT_MODEL in metrics:
        best_name = PREFERRED_DEPLOYMENT_MODEL
    else:
        best_name = max(metrics, key=lambda k: metrics[k]["f1"])
    raw_best_model = models[best_name]

    print(f"\n[INFO] Deploying: {best_name} (F1={metrics[best_name]['f1']:.4f}) "
          f"-- chosen for smooth, input-sensitive probability output.")
    print(f"[INFO] Calibrating {best_name} so probabilities are more graduated "
          f"(e.g. 62%, 78%) instead of extreme 0%/100% values...")

    # Wrap the best model in calibration (5-fold CV) so predict_proba
    # returns realistic, well-spread probabilities instead of overconfident
    # near-0/near-1 values -- common with NaiveBayes especially.
    calibrated_model = CalibratedClassifierCV(raw_best_model, method="sigmoid", cv=5)
    calibrated_model.fit(X_train_scaled, y_train)

    return best_name, calibrated_model, scaler, metrics


def save_artifacts(best_name, best_model, scaler, metrics, out_dir):
    """
    Saves the trained model, scaler, and metadata (feature order,
    categorical encodings) needed by the FastAPI backend.

    Args:
        best_name (str): name of the best model.
        best_model: fitted (calibrated) best model.
        scaler: fitted StandardScaler.
        metrics (dict): metrics for all 7 models (for reference/audit).
        out_dir (str): directory to save artifacts into.

    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(out_dir, "pcos_app_model.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "pcos_app_scaler.joblib"))

    metadata = {
        "model_name": best_name,
        "calibrated": True,
        "feature_order": APP_FEATURES,
        "categorical_encodings": {
            "Cycle(R/I)": {"Regular": 2, "Irregular": 4},
            "binary_yes_no_fields": [
                "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",
                "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)"
            ],
            "binary_encoding": {"Yes": 1, "No": 0},
        },
        "bmi_formula": "BMI = weight_kg / ((height_cm / 100) ** 2)",
        "all_model_metrics": metrics,
    }
    with open(os.path.join(out_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[SAVED] Model, scaler, and metadata saved to: {out_dir}")


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING APP-MATCHED PCOS MODEL")
    print("=" * 60)

    X, y = load_and_prepare_data(
        os.path.join("data", "cleaned", "pcos_without_infertility_clean.csv")
    )
    best_name, best_model, scaler, metrics = train_and_select_best(X, y)
    save_artifacts(
        best_name, best_model, scaler, metrics,
        out_dir=os.path.join("app_deployment"),
    )

    print("\n" + "=" * 60)
    print("APP MODEL TRAINING COMPLETE")
    print("=" * 60)